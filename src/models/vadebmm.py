import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from modules import Cell
from utils import *

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
            'codds': nn.Parameter(torch.ones(config.PARAM['num_levels'],config.PARAM['code_size'],config.PARAM['classes_size'])/config.PARAM['num_levels']),
            'odds': nn.Parameter(torch.ones(config.PARAM['classes_size'])/config.PARAM['classes_size'])
            })
                
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
            q_c_z = output['classification']['pred']
            q_y = output['compression']['param']['qy'].view(input['img'].size(0),config.PARAM['num_levels'],-1,1)
            loss = torch.sum(q_c_z*(((F.log_softmax(q_y,dim=1)-F.log_softmax(self.param['codds'],dim=0))*F.log_softmax(q_y,dim=1).exp()).sum(dim=1).sum(dim=1)),dim=1)
            loss = loss + torch.sum(q_c_z*(torch.log(q_c_z)-F.log_softmax(self.param['odds'],dim=-1)),dim=1)
            loss = loss.mean()
        else:
            loss = torch.tensor(0,device=device,dtype=torch.float32)
        return loss
        
    def forward(self, input):
        z = input.view(input.size(0),config.PARAM['num_levels'],-1,1)
        q_c_z = torch.exp(F.log_softmax(self.param['odds'],dim=-1)+(z*F.log_softmax(self.param['codds'],dim=0)).sum(dim=1).sum(dim=1))+1e-10
        x = q_c_z/torch.sum(q_c_z,dim=1,keepdim=True)
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.classifier = Classifier()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encoder_y = Cell({'input_size':256,'output_size':config.PARAM['code_size']*config.PARAM['num_levels'],'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none'})

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
            elif(config.PARAM['init_param_mode'] == 'bmm'):
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
    
    def reparameterize(self, logits, temperature):
        if self.training:
            z = gumbel_softmax(logits,tau=temperature,hard=True,sample=True,dim=1)
        else:
            z = gumbel_softmax(logits,tau=temperature,hard=True,sample=False,dim=1)
        z = z.view(z.size(0),-1,1,1)
        return z
        
    def forward(self, input):        
        output = {'loss':torch.tensor(0,device=device,dtype=torch.float32),
            'compression':{'img':torch.tensor(0,device=device,dtype=torch.float32),'code':[],'param':None},
            'classification': {}}

        img = input['img'].view(input['img'].size(0),-1,1,1)
        encoded = self.encoder(img)
        y = self.encoder_y(encoded)
        qy = y.view(y.size(0),config.PARAM['num_levels'],config.PARAM['code_size'])
        output['compression']['code'] = self.reparameterize(qy,config.PARAM['temperature'])
        output['compression']['param'] = {'qy':qy}

        if(config.PARAM['tuning_param']['compression'] > 0):
            compression_output = self.decoder(output['compression']['code'])
            output['compression']['img'] = compression_output.view(input['img'].size())
        
        if(config.PARAM['tuning_param']['classification'] > 0):
            classification_output = self.classifier(output['compression']['code'])
            output['classification']['pred'] = classification_output

        output['loss']  = config.PARAM['tuning_param']['compression']*self.decoder.compression_loss_fn(input,output) + config.PARAM['tuning_param']['classification']*self.classifier.classification_loss_fn(input,output)
        return output

def vadebmm(model_TAG):
    model_TAG_list = model_TAG.split('_')
    init_dim = 1 if(model_TAG_list[1]=='MNIST') else 3  
    config.PARAM['code_size'] = int(model_TAG_list[3])
    config.PARAM['model'] = {}
    config.PARAM['model']['encoder_info'] = [
        {'input_size':784,'output_size':512,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},        
        {'input_size':512,'output_size':256,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},       
        ]
    config.PARAM['model']['decoder_info'] = [
        {'input_size':config.PARAM['code_size']*config.PARAM['num_levels'],'output_size':256,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']},
        {'input_size':256,'output_size':512,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':config.PARAM['normalization'],'activation':config.PARAM['activation']}, 
        {'input_size':512,'output_size':784,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':config.PARAM['normalization'],'activation':'sigmoid'}, 
        ]
    config.PARAM['model']['classifier_info'] = [
        {'cell':'none'},
        ]
    model = Model()
    return model
    