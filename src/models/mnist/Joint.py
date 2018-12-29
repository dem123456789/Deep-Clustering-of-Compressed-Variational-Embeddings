import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import config
from modules import Quantize
from functions import pixel_unshuffle
from data import extract_patches_2d, reconstruct_from_patches_2d
from utils import RGB_to_L, L_to_RGB

config.init()
device = config.PARAM['device']
code_size = 1

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
        mode = protocol['mode']
        depth = protocol['depth']
        x = L_to_RGB(input) if (mode == 'L') else input
        x = torch.tanh(self.conv0(x))
        for i in range(depth):
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
        mode = protocol['mode']
        depth = protocol['depth']
        x = torch.tanh(self.conv0(input))
        for i in range(depth):
            x = self.decoder[i](x, protocol)
        x = torch.tanh(self.conv1(x))
        x = RGB_to_L(x) if (mode == 'L') else x
        return x

class Codec(nn.Module):
    def __init__(self):
        super(Codec, self).__init__()
        self.encoder = Encoder()
        self.embedding = NormalEmbedding()
        self.decoder = Decoder()
        
    def compression_loss_fn(self, input, output, protocol):
        loss = F.binary_cross_entropy(output['compression']['img'],input['img'])
        return loss
        
class Classifier(nn.Module):
    def __init__(self, classes_size):
        super(Classifier, self).__init__()
        self.classes_size = classes_size
        self.classifier_info = self.make_classifier_info()
        self.classifier = self.make_classifier(self.classifier_info)
        
    def make_classifier_info(self):
        classifier_info = {'input_size':code_size*4*4,'output_size':self.classes_size}
        return classifier_info
        
    def make_classifier(self, classifier_info):
        classifier = nn.ParameterDict([])       
        classifier['y'] = nn.Parameter(torch.ones(classifier_info['output_size'])/classifier_info['output_size'])
        classifier['mu'] = nn.Parameter(torch.zeros(classifier_info['input_size'], classifier_info['output_size']).normal_())
        classifier['var'] = nn.Parameter(torch.ones(classifier_info['input_size'], classifier_info['output_size']))
        return classifier
       
    def classification_loss_fn(self, input, output, protocol):
        z = output['compression']['code'].view(output['compression']['code'].size(0),-1,1)
        q_mu = output['compression']['param']['mu'].view(z.size(0),-1,1)
        q_var = output['compression']['param']['var'].view(z.size(0),-1,1)
        q_z_x = 1/torch.sqrt(2*math.pi*q_var)*torch.exp((z-q_mu)**2/(2*q_var))
        log_q_c_z = output['classification']
        KLD_mvn = 0.5*(torch.sum((q_var/self.classifier['var'] + (self.classifier['mu']-q_mu)**2/self.classifier['var'] - 1),dim=1) + torch.log(self.classifier['var'].prod(dim=0)/q_var.prod(dim=1)))
        KLD_categorical = log_q_c_z.exp()*(log_q_c_z-F.log_softmax(self.classifier['y'],dim=0)) 
        #loss = ((KLD_mvn*log_q_c_z.exp()).sum() + (KLD_categorical*q_z_x.prod(dim=1)).sum())/(input['img'].numel())
        loss = ((KLD_mvn*log_q_c_z.exp()).sum() + (KLD_categorical*q_z_x.prod(dim=1)).sum())/(input['img'].numel())
        if(torch.isnan(log_q_c_z).any()):
            print(output['compression']['code'][0,0])
            print(output['compression']['param']['mu'])
            print(output['compression']['param']['var'])
            print(self.classifier['mu'])
            print(self.classifier['var'])    
            print(KLD_mvn.sum())
            print(KLD_categorical.sum())
            print(loss)
            exit()  
        return loss
        
    def forward(self, input, protocol):
        x = input.view(input.size(0),-1,1)
        log_q_c_z = F.log_softmax(self.classifier['y'],dim=0) - torch.sum(0.5*torch.log(2*math.pi*self.classifier['var']) + (x-self.classifier['mu'])**2/(2*self.classifier['var']),dim=1)
        normalization_factor = torch.log(torch.exp(log_q_c_z-log_q_c_z.max(dim=1,keepdim=True)[0]).sum(dim=1,keepdim=True)) + log_q_c_z.max(dim=1,keepdim=True)[0]
        log_q_c_z = log_q_c_z - normalization_factor
        #q_c_z = torch.exp(log_q_c_z-log_q_c_z.max(dim=1,keepdim=True)[0])/torch.exp(log_q_c_z-log_q_c_z.max(dim=1,keepdim=True)[0]).sum(dim=1,keepdim=True)    
        return log_q_c_z
        
class Joint(nn.Module):
    def __init__(self,classes_size):
        super(Joint, self).__init__()
        self.classes_size = classes_size
        self.codec = Codec()
        self.classifier = Classifier(classes_size)

    def loss_fn(self, input, output, protocol):
        tuning_param = protocol['tuning_param']
        loss = tuning_param['compression']*self.codec.compression_loss_fn(input,output,protocol) +\
                tuning_param['classification']*self.classifier.classification_loss_fn(input,output,protocol)
        return loss

    def forward(self, input, protocol):
        tuning_param = protocol['tuning_param']
        loss = 0
        output = {}
        
        img = (input['img'] - 0.5) * 2
        encoded = self.codec.encoder(img,protocol)
        code, param = self.codec.embedding(encoded,protocol)
        compression_output = self.codec.decoder(code,protocol)
        compression_output = (compression_output + 1) * 0.5
        output['compression'] = {'code':code, 'param':param, 'img':compression_output}
        compression_loss = tuning_param['compression']*self.codec.compression_loss_fn(input,output,protocol)
        loss = loss + compression_loss
        
        if(tuning_param['classification'] > 0):
            classification_output = self.classifier(code,protocol)
            output['classification'] = classification_output
            classification_loss = tuning_param['classification']*self.classifier.classification_loss_fn(input,output,protocol)
            loss = loss + classification_loss
        output['loss'] = loss
        return output
        
        