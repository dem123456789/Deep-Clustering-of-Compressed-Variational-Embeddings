import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from modules import Cell
from utils import dict_to_device

device = config.PARAM['device']
   
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_info = self.make_encoder_info()
        self.encoder = self.make_encoder()
        
    def make_encoder_info(self):
        encoder_info = config.PARAM['model']['encoder_info']
        return encoder_info

    def make_encoder(self):
        encoder = nn.ModuleList([])
        for i in range(len(self.encoder_info)):
            encoder.append(Cell(self.encoder_info[i]))
        return encoder
        
    def forward(self, input):
        x = input
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_info = self.make_decoder_info()
        self.decoder = self.make_decoder()
        
    def make_decoder_info(self):
        decoder_info = config.PARAM['model']['decoder_info']
        return decoder_info

    def make_decoder(self):
        decoder = nn.ModuleList([])
        for i in range(len(self.decoder_info)):
            decoder.append(Cell(self.decoder_info[i]))
        return decoder

    def compression_loss_fn(self, input, output):
        if(config.PARAM['loss_mode']['compression'] == 'bce'):
            loss_fn = F.binary_cross_entropy
        elif(config.PARAM['loss_mode']['compression'] == 'mse'):
            loss_fn = F.mse_loss
        elif(config.PARAM['loss_mode']['compression'] == 'mae'):
            loss_fn = F.l1_loss
        else:
            raise ValueError('compression loss mode not supported') 
        if(config.PARAM['tuning_param']['compression'] > 0):
            loss = loss_fn(output['compression']['img'],input['img'],reduction='none').view(input['img'].size(0),-1).sum(dim=1)
            loss = loss.mean()
        else:
            loss = torch.tensor(0,device=device,dtype=torch.float32)
        return loss
        
    def forward(self, input):
        x = input
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)
        return x
        
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier_info = self.make_classifier_info()
        self.classifier = self.make_classifier()
        self.param = nn.ParameterDict({
            'mu': nn.Parameter(torch.zeros(config.PARAM['code_size'], config.PARAM['classes_size'])),
            'var': nn.Parameter(torch.ones(config.PARAM['code_size'], config.PARAM['classes_size'])),
            'odds': nn.Parameter(torch.ones(config.PARAM['classes_size'])/config.PARAM['classes_size'])
            })

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.bias.data.zero_()
                
    def make_classifier_info(self):
        classifier_info = config.PARAM['model']['classifier_info']
        return classifier_info
        
    def make_classifier(self):
        classifier = nn.ModuleList([])
        for i in range(len(self.classifier_info)):
            classifier.append(Cell(self.classifier_info[i]))
        return classifier

    def classification_loss_fn(self, input, output):
        if(config.PARAM['tuning_param']['classification'] > 0):
            q_c_z = output['classification']
            q_mu = output['compression']['param']['mu'].view(input['img'].size(0),-1,1)
            q_logvar = output['compression']['param']['logvar'].view(input['img'].size(0),-1,1)
            loss = torch.sum(q_c_z*0.5*torch.sum((q_mu-self.param['mu'])**2/self.param['var']+torch.exp(q_logvar)/self.param['var']-1+torch.log(self.param['var'])-q_logvar,dim=1),dim=1)            
            loss = loss + torch.sum(q_c_z*(torch.log(q_c_z)-F.log_softmax(self.param['odds'],dim=-1)),dim=1)
            loss = loss.mean()
        else:
            loss = torch.tensor(0,device=device,dtype=torch.float32)
        return loss
        
    def forward(self, input):
        z = input.view(input.size(0),-1,1)
        q_c_z = torch.exp(F.log_softmax(self.param['odds'],dim=-1)-torch.sum(0.5*torch.log(2*math.pi*self.param['var'])+(z-self.param['mu'])**2/(2*self.param['var']),dim=1))+1e-10
        x = q_c_z/torch.sum(q_c_z,dim=1,keepdim=True)
        return x
        
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.classifier = Classifier()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encoder_mean = Cell({'input_size':128*7*7,'output_size':config.PARAM['code_size'],'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none'})
        self.encoder_logvar = Cell({'input_size':128*7*7,'output_size':config.PARAM['code_size'],'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none'})
        self.decoder_in = Cell({'input_size':config.PARAM['code_size'],'output_size':128*7*7,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none'})

    def init_param(self, train_loader):
        with torch.no_grad():
            self.train(False)
            for i, input in enumerate(train_loader):
                for k in input:
                    input[k] = torch.stack(input[k],0)
                input = dict_to_device(input,device)
                output = self(input)
                z = output['compression']['code'].view(input['img'].size(0),-1)
                Z = torch.cat((Z,z),0) if i > 0 else z
            if(config.PARAM['init_param_mode']=='random'):
                C = torch.rand(Z.size(0),config.PARAM['classes_size'],device=device)
                nk = C.sum(dim=0,keepdim=True)+1e-10
                self.classifier.param['mu'].copy_(Z.t().matmul(C)/nk)
                self.classifier.param['var'].copy_((Z**2).t().matmul(C)/nk-2*self.classifier.param['mu']*Z.t().matmul(C)/nk+self.classifier.param['mu']**2)
            elif(config.PARAM['init_param_mode']=='kmeans'):
                from sklearn.cluster import KMeans
                C = Z.new_zeros(Z.size(0),config.PARAM['classes_size'])
                km = KMeans(n_clusters=config.PARAM['classes_size'],n_init=1,random_state=config.PARAM['randomGen']).fit(Z.cpu().numpy())
                C[torch.arange(C.size(0)),torch.tensor(km.labels_).long()] = 1
                nk = C.sum(dim=0,keepdim=True)+1e-10
                self.classifier.param['mu'].copy_(Z.t().matmul(C)/nk)
                self.classifier.param['var'].copy_((Z**2).t().matmul(C)/nk-2*self.classifier.param['mu']*Z.t().matmul(C)/nk+self.classifier.param['mu']**2)
            elif(config.PARAM['init_param_mode']=='gmm'):
                from sklearn.mixture import GaussianMixture
                gm = GaussianMixture(n_components=config.PARAM['classes_size'],covariance_type='diag',random_state=config.PARAM['randomGen']).fit(Z.cpu().numpy())
                self.classifier.param['mu'].copy_(torch.tensor(gm.means_.T).float().to(device))
                self.classifier.param['var'].copy_(torch.tensor(gm.covariances_.T).float().to(device))
            elif(config.PARAM['init_param_mode']=='bmm'):
                from bmm import BMM
                Z = torch.max(Z.view(-1,config.PARAM['num_levels'],config.PARAM['code_size']),dim=1)[1]
                bmm = BMM(n_comp=10,n_iter=300).fit(Z.cpu().numpy())
                bmmq = torch.tensor(bmm.q).float().to(device)
                bmmq[bmmq<=0.1] = 0.1
                bmmq[bmmq>=0.9] = 0.9
                self.classifier.param['codds'].copy_(torch.stack([1-bmmq,bmmq],dim=0))
            else:
                raise ValueError('Initialization method not supported')
        return
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            z = eps.mul(std).add_(mu)
        else:
            z = mu
        return z
        
    def forward(self, input):
        output = {'loss':torch.tensor(0,device=device,dtype=torch.float32),
            'compression':{'img':torch.tensor(0,device=device,dtype=torch.float32),'code':[],'param':None},
            'classification': torch.tensor(0,device=device,dtype=torch.float32)}

        img = input['img']
        encoded = self.encoder(img)
        mu = self.encoder_mean(encoded.view(img.size(0),-1,1,1))
        logvar = self.encoder_logvar(encoded.view(img.size(0),-1,1,1))
        output['compression']['code'] = self.reparameterize(mu,logvar)
        output['compression']['param'] = {'mu':mu,'logvar':logvar}

        if(config.PARAM['tuning_param']['compression'] > 0):
            todecode = self.decoder_in(output['compression']['code'])
            compression_output = self.decoder(todecode.view(encoded.size()))
            output['compression']['img'] = compression_output.view(input['img'].size())
        
        if(config.PARAM['tuning_param']['classification'] > 0):
            classification_output = self.classifier(output['compression']['code'])
            output['classification'] = classification_output

        output['loss']  = config.PARAM['tuning_param']['compression']*self.decoder.compression_loss_fn(input,output) + config.PARAM['tuning_param']['classification']*self.classifier.classification_loss_fn(input,output)
        return output
    
def cvade(model_TAG):
    model_TAG_list = model_TAG.split('_')
    init_dim = 1 if(model_TAG_list[1]=='MNIST') else 3  
    config.PARAM['code_size'] = int(model_TAG_list[3])
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder_info'] = [
        {'input_size':init_dim,'output_size':64,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},         
        {'input_size':64,'output_size':128,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'cell':'PixelShuffleCell','mode':'down','scale_factor':2},
        {'input_size':512,'output_size':128,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'cell':'PixelShuffleCell','mode':'down','scale_factor':2},
        {'input_size':512,'output_size':128,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':128,'output_size':128,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']}        
        ]
    config.PARAM['model']['decoder_info'] = [
        {'input_size':128,'output_size':128,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},      
        {'input_size':128,'output_size':512,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'cell':'PixelShuffleCell','mode':'up','scale_factor':2},
        {'input_size':128,'output_size':512,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'cell':'PixelShuffleCell','mode':'up','scale_factor':2},
        {'input_size':128,'output_size':64,'num_layer':1,'cell':'BasicCell','mode':'pass','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':64,'output_size':init_dim,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':config.PARAM['normalization'],'activation':'sigmoid'},        
        ]
    config.PARAM['model']['classifier_info'] = [
        {'cell':'none'},
        ]
    model = Model()
    return model
  