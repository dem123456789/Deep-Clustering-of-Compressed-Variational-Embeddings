import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import config
from modules import Cell
from bmm_implement import BMM

device = config.PARAM['device']

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape,device=device)
    return -torch.log(-torch.log(U+eps)+eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits+sample_gumbel(logits.size())
    return F.softmax(y/temperature,dim=-1)

def gumbel_softmax(logits, temperature, hard=False):
    y = gumbel_softmax_sample(logits,temperature)   
    if not hard:
        return y.view(-1,config.PARAM['code_size']*config.PARAM['num_level'],1,1)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1,shape[-1])
    y_hard.scatter_(1,ind.view(-1,1),1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard-y).detach()+y
    y_hard = y_hard.view(-1,config.PARAM['code_size']*config.PARAM['num_level'],1,1)
    return y_hard

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_info = self.make_encoder_info()
        self.encoder = self.make_encoder()
        
    def make_encoder_info(self):
        encoder_info = [
        {'input_size':1024,'output_size':512,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':config.PARAM['activation'],'raw':False},        
        # {'input_size':500,'output_size':500,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':config.PARAM['activation'],'raw':False},
        {'input_size':512,'output_size':256,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':config.PARAM['activation'],'raw':False},
        # {'input_size':256,'output_size':2000,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':config.PARAM['activation'],'raw':False},       
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
        {'input_size':config.PARAM['code_size']*config.PARAM['num_level'],'output_size':256,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':config.PARAM['activation'],'raw':False},
        # {'input_size':2000,'output_size':500,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':config.PARAM['activation'],'raw':False}, 
        # {'input_size':500,'output_size':500,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':config.PARAM['activation'],'raw':False}, 
        {'input_size':256,'output_size':512,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':config.PARAM['activation'],'raw':False}, 
        {'input_size':512,'output_size':1024,'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':'sigmoid','raw':False}, 
        ]
        return decoder_info

    def make_decoder(self):
        decoder = nn.ModuleList([])
        for i in range(len(self.decoder_info)):
            decoder.append(Cell(self.decoder_info[i]))
        return decoder
        
    def forward(self, input, protocol):
        # x = input
        x = input.view(input.size(0),-1,1,1)
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)
        return x

class vade_bmm(nn.Module):
    def __init__(self,classes_size):
        super(vade_bmm, self).__init__()
        self.classes_size = classes_size
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.encoder_y = Cell({'input_size':256,'output_size':config.PARAM['code_size']*config.PARAM['num_level'],'num_layer':1,'cell':'BasicCell','mode':'fc','normalization':'none','activation':'none','raw':False})

        self.param = nn.ParameterDict({
            'mean': nn.Parameter(torch.ones(config.PARAM['code_size'], self.classes_size)/config.PARAM['num_level']),
            'pi': nn.Parameter(torch.ones(self.classes_size)/self.classes_size)
            })

    def reparameterize(self, logits, temperature):
        if self.training:
            z = gumbel_softmax(logits,temperature,hard=True)
            # z_1 = gumbel_softmax(logits,0.5,hard=False)
            # z_2 = gumbel_softmax(logits,0.5,hard=True)
            # print('sample y, hard=F',z_1.view(100,32,2)[:,0,:])
            # a = torch.argmax(z_1.view(100,32,2),dim=2).float()[:,0]
            # b = torch.argmax(logits.view(100,32,2),dim=2).float()[:,0]
            # c = F.gumbel_softmax(logits.view(100,32,2)[:,0,:], tau=1., hard=True, eps=1e-10)[:,0]
            # e = torch.argmax(F.gumbel_softmax(logits.view(100,32,2)[:,0,:], tau=1., hard=False, eps=1e-10),dim=-1).float()
            # d = z_2.view(100,32,2)[:,0,0]
            # print('if argmax(sample y)==gumbel_softmax(qy, hard =T)?',torch.eq(a,d).sum())
            # print('if F.gumbel_softmax(qy, hard =T) == gumbel_softmax(qy, hard =T)?',torch.eq(c,d).sum())
            # print('if argmax(qy)==gumbel_softmax(qy, hard =T)?',torch.eq(b,d).sum())
            # print('if argmax(qy)==F.gumbel_softmax(sample y, hard =T)?',torch.eq(b,c).sum())
            # print('if argmax(F sample y)==F.gumbel_softmax(sample y, hard =T)?',torch.eq(e,c).sum())
        else:
            z = F.softmax(logits, dim=-1).reshape(logits.size(0),-1)
        return z

    def classifier(self, input, protocol):        
        z = input.view(input.size(0),config.PARAM['code_size'],config.PARAM['num_level'],1)#100x32x2x1
        z = z[:,:,0,:] #100x32x1
        p_y_c = torch.sigmoid(self.param['mean'])
        # p_y_c = self.param['mean']
        q_c_z = torch.exp(torch.log(self.param['pi'])+torch.sum(z*torch.log(p_y_c)+(1-z)*torch.log(1-p_y_c),dim=1))+1e-10
        q_c_z = q_c_z/q_c_z.sum(dim=1,keepdim=True) #NxH
        return q_c_z

    def classification_loss_fn(self, input, output, protocol):
        loss = torch.tensor(0,device=device,dtype=torch.float32)
        z = output['classification']['code'].view(input['img'].size(0),config.PARAM['code_size'],config.PARAM['num_level'],1)
        z = z[:,:,0,:] #100x32x1
        p_y_c = torch.sigmoid(self.param['mean'])
        # p_y_c = self.param['mean']
        if(protocol['tuning_param']['classification'] > 0): 
            q_c_z = output['classification']['classifier'] #NxH
            q_y = output['compression']['param']['qy'].view(input['img'].size(0),config.PARAM['code_size'],config.PARAM['num_level'],1)
            loss = loss + torch.sum(output['compression']['param']['qy']*torch.log(output['compression']['param']['qy']+1e-10),dim=1)
            loss = loss - torch.sum(q_c_z*torch.sum(z*torch.log(p_y_c)+(1-z)*torch.log(1-p_y_c),dim=1),dim=1)
            loss = loss + (q_c_z*(q_c_z.log()-self.param['pi'].log())).sum(dim=1)
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
            'classification':{'code':torch.tensor(0,device=device,dtype=torch.float32),'classifier':torch.tensor(0,device=device,dtype=torch.float32)}}

        img = input['img'].view(-1,1024).float()
        encoded = self.encoder(img,protocol)
        y = self.encoder_y(encoded)
        qy = y.view(y.size(0),config.PARAM['code_size'],config.PARAM['num_level'])
        output['compression']['code'] = self.reparameterize(qy,protocol['temperature'])
        output['classification']['code'] = gumbel_softmax(qy,protocol['temperature'],hard=True)
        output['compression']['param'] = {'qy':F.softmax(qy, dim=-1).reshape(y.size())}
        if(protocol['tuning_param']['compression'] > 0):
            compression_output = self.decoder(output['compression']['code'],protocol)
            output['compression']['img'] = compression_output.view(input['img'].size())
        
        if(protocol['tuning_param']['classification'] > 0):
            classification_output = self.classifier(output['classification']['code'],protocol)
            output['classification']['classifier'] = classification_output
        
        output['loss'] = self.loss_fn(input,output,protocol)

        return output
