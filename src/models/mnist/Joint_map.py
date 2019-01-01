import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import config
from modules import Quantize
from functions import pixel_unshuffle
from data import extract_patches_2d, reconstruct_from_patches_2d
from utils import RGB_to_L, L_to_RGB,dict_to_device

config.init()
device = config.PARAM['device']
code_size = 4

class Cell(nn.Module):
    def __init__(self, cell_info):
        super(Cell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        cell = nn.Conv2d(self.cell_info['input_size'],self.cell_info['output_size'],kernel_size=self.cell_info['kernel_size'],\
                stride=self.cell_info['stride'],padding=self.cell_info['padding'],dilation=self.cell_info['dilation'],bias=self.cell_info['bias'])
        return cell

    def forward(self, input, protocol):
        x = self.cell(input)
        return x

class EncoderCell(nn.Module):
    def __init__(self, encoder_cell_info):
        super(EncoderCell, self).__init__()
        self.encoder_cell_info = encoder_cell_info
        self.encoder_cell = self.make_encoder_cell()

    def make_encoder_cell(self):
        encoder_cell = nn.ModuleList([])
        for i in range(len(self.encoder_cell_info)):   
            encoder_cell.append(Cell(self.encoder_cell_info[i]))
        return encoder_cell

    def forward(self, input, protocol):
        x = pixel_unshuffle(input, protocol['jump_rate'])
        for i in range(len(self.encoder_cell)):          
            x = torch.tanh(self.encoder_cell[i](x,protocol))
        return x
        
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.encoder_info = self.make_encoder_info()
        self.encoder = self.make_encoder()
        
    def make_encoder_info(self):
        encoder_info = [        
        [{'input_size':512,'output_size':128,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False}],
        [{'input_size':512,'output_size':128,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False}],
        [{'input_size':512,'output_size':128,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False}],
        ]
        return encoder_info

    def make_encoder(self):
        encoder = nn.ModuleList([])
        for i in range(len(self.encoder_info)):
            encoder.append(EncoderCell(self.encoder_info[i]))
        return encoder
        
    def forward(self, input, protocol):
        x = L_to_RGB(input) if (protocol['mode'] == 'L') else input
        x = torch.tanh(self.conv0(x))
        for i in range(protocol['depth']):
            x = self.encoder[i](x, protocol)
        return x
               
class NormalEmbedding(nn.Module):
    def __init__(self):
        super(NormalEmbedding, self).__init__()
        self.conv_mean = nn.Conv2d(128, code_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_logvar = nn.Conv2d(128, code_size, kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, input, protocol):
        mu = self.conv_mean(input)
        logvar = self.conv_logvar(input)
        std = logvar.mul(0.5).exp()
        eps = mu.new_zeros(mu.size()).normal_()
        z = eps.mul(std).add_(mu)
        param = {'mu':mu,'var':logvar.exp()}
        return z, param

class DecoderCell(nn.Module):
    def __init__(self, decoder_cell_info):
        super(DecoderCell, self).__init__()
        self.decoder_cell_info = decoder_cell_info
        self.decoder_cell = self.make_decoder_cell()
        
    def make_decoder_cell(self):
        decoder_cell = nn.ModuleList([])
        for i in range(len(self.decoder_cell_info)):   
            decoder_cell.append(Cell(self.decoder_cell_info[i]))
        return decoder_cell

    def forward(self, input, protocol):
        x = input
        for i in range(len(self.decoder_cell)):          
            x = torch.tanh(self.decoder_cell[i](x,protocol))
        x = F.pixel_shuffle(x, protocol['jump_rate'])
        return x
        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv0 = nn.Conv2d(code_size, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.decoder_info = self.make_decoder_info()
        self.decoder = self.make_decoder()
        self.conv1 = nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0, bias=False)
        
    def make_decoder_info(self):
        decoder_info = [        
        [{'input_size':128,'output_size':512,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False}],
        [{'input_size':128,'output_size':512,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False}],
        [{'input_size':128,'output_size':512,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False}],
        ]
        return decoder_info

    def make_decoder(self):
        decoder = nn.ModuleList([])
        for i in range(len(self.decoder_info)):
            decoder.append(DecoderCell(self.decoder_info[i]))
        return decoder
        
    def forward(self, input, protocol):
        x = torch.tanh(self.conv0(input))
        for i in range(protocol['depth']):
            x = self.decoder[i](x, protocol)
        x = torch.tanh(self.conv1(x))
        x = RGB_to_L(x) if (protocol['mode'] == 'L') else x
        return x

class Codec(nn.Module):
    def __init__(self,classes_size):
        super(Codec, self).__init__()
        self.classes_size = classes_size
        self.encoder = Encoder()
        self.embedding = NormalEmbedding()
        self.decoder = Decoder()
        
    def compression_loss_fn(self, input, output, protocol):
        loss = 0
        if(protocol['tuning_param']['compression'] > 0):
            loss = loss + F.binary_cross_entropy(output['compression']['img'],input['img'],reduction='none').view(input['img'].size(0),-1).sum(dim=1)
        q_mu = output['compression']['param']['mu'].view(input['img'].size(0),-1,1)
        q_var = output['compression']['param']['var'].view(input['img'].size(0),-1,1)
        if(protocol['tuning_param']['classification'] > 0):
            q_c_z = output['classification']
            loss = loss + torch.sum(0.5*q_c_z*torch.sum(math.log(2*math.pi)+torch.log(input['param']['var'])+\
                q_var/input['param']['var'] + (q_mu-input['param']['mu'])**2/input['param']['var'], dim=1), dim=1)
            loss = loss + (-0.5*torch.sum(1+q_var+math.log(2*math.pi), 1)).squeeze(1)
        loss = loss.mean()
        return loss

class ClassifierCell(nn.Module):
    def __init__(self, classifier_cell_info):
        super(ClassifierCell, self).__init__()
        self.classifier_cell_info = classifier_cell_info
        self.classifier_cell = self.make_classifier_cell()
        
    def make_classifier_cell(self):
        classifier_cell = nn.ModuleList([])
        for i in range(len(self.classifier_cell_info)):
            classifier_cell.append(Cell(self.classifier_cell_info[i]))
        return classifier_cell
        
    def forward(self, input, protocol):
        x = input
        for i in range(len(self.classifier_cell)-1):          
            x = torch.tanh(self.classifier_cell[i](x,protocol))
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.classifier_cell[-1](x,protocol)
        x = x.view(x.size(0),x.size(1))
        return x
        
class Classifier(nn.Module):
    def __init__(self, classes_size):
        super(Classifier, self).__init__()
        self.classes_size = classes_size
        self.classifier_info = self.make_classifier_info()
        self.classifier = self.make_classifier()

    def make_classifier_info(self):
        classifier_info = [
                        {'input_size':code_size,'output_size':512,'kernel_size':1,'stride':1,'padding':0,'dilation':1,'bias':False},
                        {'input_size':512,'output_size':512,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False},
                        {'input_size':512,'output_size':self.classes_size,'kernel_size':1,'stride':1,'padding':0,'dilation':1,'bias':False}
                        ]
        return classifier_info

    def make_classifier(self):
        classifier = ClassifierCell(self.classifier_info)
        return classifier

    def classification_loss_fn(self, input, output, protocol):
        loss = 0
        if(protocol['tuning_param']['classification'] > 0): 
            q_c_z = output['classification']
            loss = loss + (q_c_z*(q_c_z.log()-input['param']['pi'].log())).sum(dim=1)
            loss = loss.mean()
        return loss
        
    def forward(self, input, protocol):
        x = self.classifier(input,protocol)
        x = F.log_softmax(x,dim=1).exp() + 10*np.finfo(np.float32).eps
        q_c_z = x/x.sum(dim=1,keepdim=True)
        return q_c_z

class Joint_map(nn.Module):
    def __init__(self,classes_size):
        super(Joint_map, self).__init__()
        self.classes_size = classes_size
        self.codec = Codec(classes_size)
        self.classifier = Classifier(classes_size)
        self.param = nn.ParameterDict({})
    
    def init_param_protocol(self,dataset,randomGen):
        protocol = {}
        protocol['tuning_param'] = config.PARAM['tuning_param'].copy()
        protocol['tuning_param']['classification'] = 1
        protocol['init_param_mode'] = 'random'
        protocol['classes_size'] = dataset.classes_size
        protocol['randomGen'] = randomGen
        protocol['loss'] = False
        return protocol

    def init_train_protocol(self,dataset):
        protocol = {}
        protocol['tuning_param'] = config.PARAM['tuning_param'].copy()
        protocol['metric_names'] = config.PARAM['metric_names'][:-1]
        protocol['topk'] = config.PARAM['topk']
        if(config.PARAM['balance']):
            protocol['classes_counts'] = dataset.classes_counts.expand(world_size,-1).to(device)
        protocol['loss'] = True
        return protocol 

    def init_test_protocol(self,dataset):
        protocol = {}
        protocol['tuning_param'] = config.PARAM['tuning_param'].copy()
        protocol['metric_names'] = config.PARAM['metric_names']
        protocol['topk'] = config.PARAM['topk']
        if(config.PARAM['balance']):
            protocol['classes_counts'] = dataset.classes_counts.expand(world_size,-1).to(device)
        protocol['loss'] = True
        return protocol
    
    def init_param(self,train_loader,protocol):
        with torch.no_grad(): 
            self.train(False)
            for i, input in enumerate(train_loader):
                input = self.collate(input)
                input = dict_to_device(input,device)
                protocol = self.update_protocol(input,protocol)
                output = self(input,protocol)
                z = output['compression']['code'].view(input['img'].size(0),-1)
                Z = torch.cat((Z,z),0) if i > 0 else z
                if(protocol['init_param_mode'] == 'random'):
                    c = output['classification']
                    C = torch.cat((C,c),0) if i > 0 else c
            if(protocol['init_param_mode'] == 'random'):
                nk = C.sum(dim=0,keepdim=True) + 10*np.finfo(np.float32).eps
                self.param['mu'] = nn.Parameter(Z.t().matmul(C)/nk)
                self.param['var'] = nn.Parameter((Z**2).t().matmul(C)/nk - 2*self.param['mu']*Z.t().matmul(C)/nk + self.param['mu']**2)
                self.param['pi'] = nn.Parameter(nk/C.size(0))
            elif(protocol['init_param_mode'] == 'kmeans'):
                from sklearn.cluster import KMeans
                C = C.new_zeros(C.size())
                km = KMeans(n_clusters=protocol['classes_size'], n_init=1, random_state=protocol['randomGen']).fit(Z)
                C[torch.arange(C.size(0)), torch.tensor(km.labels_).long()] = 1
                nk = C.sum(dim=0,keepdim=True) + 10*np.finfo(np.float32).eps
                self.param['mu'] = nn.Parameter(Z.t().matmul(C)/nk)
                self.param['var'] = nn.Parameter((Z**2).t().matmul(C)/nk - 2*self.param['mu']*Z.t().matmul(C)/nk + self.param['mu']**2)
                self.param['pi'] = nn.Parameter(nk/C.size(0))
            elif(protocol['init_param_mode'] == 'gmm'):
                from sklearn.mixture import GaussianMixture
                gm = GaussianMixture(n_components=protocol['classes_size'], covariance_type='diag').fit(Z)
                self.param['mu'] = nn.Parameter(torch.tensor(gm.means_.T).float().to(device))
                self.param['var'] = nn.Parameter(torch.tensor(gm.covariances_.T).float().to(device))
                self.param['pi'] = nn.Parameter(torch.tensor(gm.weights_).float().to(device))
            else:
                raise ValueError('Initialization method not supported')
        return
        
    def collate(self,input):
        for k in input:
            input[k] = torch.stack(input[k],0)
        return input

    def update_protocol(self,input,protocol):
        protocol['depth'] = config.PARAM['max_depth']
        protocol['jump_rate'] = config.PARAM['jump_rate']
        if(input['img'].size(1)==1):
            protocol['mode'] = 'L'
        elif(input['img'].size(1)==3):
            protocol['mode'] = 'RGB'
        else:
            raise ValueError('Wrong number of channel')
        return protocol 
    
    def loss_fn(self, input, output, protocol):
        compression_loss = self.codec.compression_loss_fn(input,output,protocol)
        classification_loss = self.classifier.classification_loss_fn(input,output,protocol)
        loss = protocol['tuning_param']['compression']*compression_loss + protocol['tuning_param']['classification']*classification_loss
        return loss

    def forward(self, input, protocol):
        output = {}
        input['param'] = self.param
        img = (input['img'] - 0.5) * 2
        encoded = self.codec.encoder(img,protocol)
        code, param = self.codec.embedding(encoded,protocol)
        output['compression'] = {'code':code, 'param':param}
        
        if(protocol['tuning_param']['compression'] > 0):
            compression_output = self.codec.decoder(code,protocol)
            compression_output = (compression_output + 1) * 0.5
            output['compression']['img'] = compression_output
       
        if(protocol['tuning_param']['classification'] > 0):
            classification_output = self.classifier(code,protocol)
            output['classification'] = classification_output
            
        if(protocol['loss']):
            output['loss'] = self.loss_fn(input,output,protocol)
        return output
        
        