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
        self.codec_prior_info = self.make_codec_prior_info()
        self.encoder = Encoder()
        self.embedding = NormalEmbedding()
        self.codec_prior = self.make_codec_prior()
        self.decoder = Decoder()
    
    def make_codec_prior_info(self):
        codec_prior_info = {'input_size':code_size*4*4,'output_size':self.classes_size}
        return codec_prior_info
    
    def make_codec_prior(self):
        codec_prior = nn.ParameterDict([])
        codec_prior['mu'] = nn.Parameter(torch.zeros(self.codec_prior_info['input_size'], self.codec_prior_info['output_size']))
        codec_prior['var'] = nn.Parameter(torch.ones(self.codec_prior_info['input_size'], self.codec_prior_info['output_size']))
        return codec_prior
        
    def compression_loss_fn(self, input, output, protocol):
        loss = 0
        if(protocol['tuning_param']['compression'] > 0):
            loss = loss + F.binary_cross_entropy(output['compression']['img'],input['img'],reduction='none').view(input['img'].size(0),-1).sum(dim=1,keepdim=True)
        q_mu = output['compression']['param']['mu'].view(input['img'].size(0),-1,1)
        q_var = output['compression']['param']['var'].view(input['img'].size(0),-1,1)
        KLD_mvn = 0.5*(torch.sum((q_var/self.codec_prior['var'] + (self.codec_prior['mu']-q_mu)**2/self.codec_prior['var'] - 1),dim=1) + torch.log(self.codec_prior['var'].prod(dim=0)/q_var.prod(dim=1)))
        if(protocol['tuning_param']['classification'] > 0):
            log_q_c_z = F.log_softmax(output['classification'],dim=1)
            loss = loss + KLD_mvn*log_q_c_z.exp()
        else:
            loss = loss + KLD_mvn
        loss = loss.sum()/input['img'].numel()
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
        self.classifier_prior_info = self.make_classifier_prior_info()
        self.classifier_prior = self.make_classifier_prior()
        
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

    def make_classifier_prior_info(self):
        classifier_prior_info = {'output_size':self.classes_size}
        return classifier_prior_info
        
    def make_classifier_prior(self):
        classifier_prior = nn.ParameterDict([])
        classifier_prior['y'] = nn.Parameter(torch.ones(self.classifier_prior_info['output_size']))
        return classifier_prior

    def classification_loss_fn(self, input, output, protocol):
        loss = 0
        if(protocol['tuning_param']['classification'] > 0):
            z = output['compression']['code'].view(input['img'].size(0),-1,1)
            q_mu = output['compression']['param']['mu'].view(input['img'].size(0),-1,1)
            q_var = output['compression']['param']['var'].view(input['img'].size(0),-1,1)
            q_z_x = 1/torch.sqrt(2*math.pi*q_var)*torch.exp((z-q_mu)**2/(2*q_var)) 
            log_q_c_z = F.log_softmax(output['classification'],dim=1)
            loss = loss +log_q_c_z.exp()*(log_q_c_z-F.log_softmax(self.classifier_prior['y'],dim=0))*q_z_x.prod(dim=1)
            loss = loss.sum()/input['img'].numel()
        return loss
        
    def forward(self, input, protocol):
        x = self.classifier(input,protocol)
        return x

class Joint(nn.Module):
    def __init__(self,classes_size):
        super(Joint, self).__init__()
        self.classes_size = classes_size
        self.codec = Codec(classes_size)
        self.classifier = Classifier(classes_size)

    def loss_fn(self, input, output, protocol):
        compression_loss = self.codec.compression_loss_fn(input,output,protocol)
        classification_loss = self.classifier.classification_loss_fn(input,output,protocol)
        loss = protocol['tuning_param']['compression']*compression_loss + protocol['tuning_param']['classification']*classification_loss
        return loss

    def forward(self, input, protocol):
        output = {}
        
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
            
        output['loss'] = self.loss_fn(input,output,protocol)
        return output
        
        