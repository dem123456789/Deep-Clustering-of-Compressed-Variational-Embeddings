import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import config
from modules import Cell

device = config.PARAM['device']

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_info = self.make_encoder_info()
        self.encoder = self.make_encoder()
        
    def make_encoder_info(self):
        encoder_info = [
        {'input_size':1024,'output_size':500,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':config.PARAM['activation'],'raw':False},        
        # {'input_size':500,'output_size':500,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':config.PARAM['activation'],'raw':False},
        {'input_size':500,'output_size':500,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':config.PARAM['activation'],'raw':False},
        {'input_size':500,'output_size':2000,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':config.PARAM['activation'],'raw':False},       
        ]
        return encoder_info
        
    def make_encoder(self):
        encoder = nn.ModuleList([])
        for i in range(len(self.encoder_info)):
            encoder.append(Cell(self.encoder_info[i]))
        return encoder
            
    def forward(self, input, protocol):
        x = input.view(input.size(0),-1,1,1)
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_info = self.make_decoder_info()
        self.decoder = self.make_decoder()
        
    def make_decoder_info(self):
        decoder_info = [
        {'input_size':config.PARAM['code_size'],'output_size':2000,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':config.PARAM['activation'],'raw':False},
        {'input_size':2000,'output_size':500,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':config.PARAM['activation'],'raw':False}, 
        # {'input_size':500,'output_size':500,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':config.PARAM['activation'],'raw':False}, 
        {'input_size':500,'output_size':500,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':config.PARAM['activation'],'raw':False}, 
        {'input_size':500,'output_size':1024,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':'sigmoid','raw':False}, 
        ]
        return decoder_info

    def make_decoder(self):
        decoder = nn.ModuleList([])
        for i in range(len(self.decoder_info)):
            decoder.append(Cell(self.decoder_info[i]))
        return decoder
        
    def forward(self, input, protocol):
        x = input
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)
        return x

class vade(nn.Module):
    def __init__(self,classes_size):
        super(vade, self).__init__()
        self.classes_size = classes_size
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encoder_mean = Cell({'input_size':2000,'output_size':config.PARAM['code_size'],'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','raw':False})
        self.encoder_logvar = Cell({'input_size':2000,'output_size':config.PARAM['code_size'],'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','raw':False})
        self.param = nn.ParameterDict({
            'mu': nn.Parameter(torch.zeros(config.PARAM['code_size'], self.classes_size)),
            'var': nn.Parameter(torch.ones(config.PARAM['code_size'], self.classes_size)),
            'pi': nn.Parameter(torch.ones(self.classes_size)/self.classes_size)
            })

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            z = eps.mul(std).add_(mu)
        else:
            z = mu
        return z

    def classifier(self, input, protocol):        
        z = input.view(input.size(0),-1,1)
        q_c_z = torch.exp(torch.log(F.softmax(self.param['pi']),dim=-1) - torch.sum(0.5*torch.log(2*math.pi*self.param['var']) +\
            (z-self.param['mu'])**2/(2*self.param['var']),dim=1)) + 1e-10
        q_c_z = q_c_z/torch.sum(q_c_z,dim=1,keepdim=True)       
        return q_c_z

    def classification_loss_fn(self, input, output, protocol):
        loss = torch.tensor(0,device=device,dtype=torch.float32)
        if(protocol['tuning_param']['classification'] > 0): 
            q_c_z = output['classification']['classifier']
            q_mu = output['compression']['param']['mu'].view(input['img'].size(0),-1,1)
            q_logvar = output['compression']['param']['logvar'].view(input['img'].size(0),-1,1)
            loss = loss + torch.sum(0.5*q_c_z*torch.sum(math.log(2*math.pi)+torch.log(self.param['var'])+\
                 torch.exp(q_logvar)/self.param['var'] + (q_mu-self.param['mu'])**2/self.param['var'],dim=1),dim=1)
            loss = loss + (-0.5*torch.sum(1+q_logvar+math.log(2*math.pi), 1)).squeeze(1)
            # loss = loss + torch.sum(0.5*q_c_z*torch.sum(torch.log(self.param['var'])+\
            #      torch.exp(q_logvar)/self.param['var'] + (q_mu-self.param['mu'])**2/self.param['var'],dim=1),dim=1)
            # loss = loss + (-0.5*torch.sum(1+q_logvar, 1)).squeeze(1)
            loss = loss + torch.sum(q_c_z*(torch.log(q_c_z)-torch.log(F.softmax(self.param['pi'],dim=-1))),dim=1)
        return loss

    def compression_loss_fn(self, input, output, protocol):
        loss = torch.tensor(0,device=device,dtype=torch.float32)
        if(protocol['tuning_param']['compression'] > 0):
            loss = loss + F.binary_cross_entropy(output['compression']['img'],input['img'],reduction='none').view(input['img'].size(0),-1).sum(dim=1)
        return loss
    
    def loss_fn(self, input, output, protocol):
        compression_loss = self.compression_loss_fn(input,output,protocol)
        classification_loss = self.classification_loss_fn(input,output,protocol)
        loss = protocol['tuning_param']['compression']*compression_loss + protocol['tuning_param']['classification']*classification_loss
        loss = torch.mean(loss)
        return loss

    def forward(self, input, protocol):
        output = {'loss':torch.tensor(0,device=device,dtype=torch.float32),
            'compression':{'img':torch.tensor(0,device=device,dtype=torch.float32),'code':[],'param':None},
            'classification': {'classifier':torch.tensor(0,device=device,dtype=torch.float32),'code':[]}}
        
        img = input['img'].view(-1,1024).float()
        encoded = self.encoder(img,protocol)
        mu = self.encoder_mean(encoded)
        logvar = self.encoder_logvar(encoded)
        output['compression']['code'] = self.reparameterize(mu,logvar)
        output['classification']['code'] = self.reparameterize(mu,logvar)
        output['compression']['param'] = {'mu':mu,'logvar':logvar}

        if(protocol['tuning_param']['compression'] > 0):
            compression_output = self.decoder(output['compression']['code'],protocol)
            output['compression']['img'] = compression_output.view(input['img'].size())
        
        if(protocol['tuning_param']['classification'] > 0):
            classification_output = self.classifier(output['compression']['code'],protocol)
            output['classification']['classifier'] = classification_output
        
        output['loss'] = self.loss_fn(input,output,protocol)

        return output
