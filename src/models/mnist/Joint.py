import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config
from modules import ConvLSTMCell, Sign
from functions import pixel_unshuffle
from data import extract_patches_2d, reconstruct_from_patches_2d
from utils import RGB_to_L, L_to_RGB

config.init()
device = config.PARAM['device']
encoder_channels = [(128,128,3,3),(512,128,3,3),(512,128,3,3),(512,128,3,1),(512,128,3,1),(512,128,3,1),(512,128,3,1),(512,128,3,1)]
decoder_channels = [(128,512,3,1),(128,512,3,1),(128,512,3,1),(128,512,3,1),(128,512,3,1),(128,512,3,3),(128,512,3,3),(128,128,3,3)]
code_channels = [1,2,8,32]

class Cell(nn.Module):
    def __init__(self, cell_info):
        super(Cell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell(cell_info)
        
    def make_cell(self, cell_info):
        cell = nn.Conv2d(self.cell_info['input_size'],self.cell_info['output_size'],self.cell_info['kernel_size'],\
                self.cell_info['stride'],self.cell_info['padding'],self.cell_info['dilation'],self.cell_info['bias'])
        return cell

    def forward(self, input, protocol):
        x = self.cell(input)
        return x
    
class EncoderCell(nn.Module):
    def __init__(self, encoder_cell_info):
        super(EncoderCell, self).__init__()
        self.encoder_cell_info = encoder_cell_info
        self.encoder_cell = self.make_encoder_cell(self.encoder_cell_info)
        
    def make_encoder_cell(self, encoder_cell_info):
        encoder_cell = nn.ModuleList([])
        for i in range(len(encoder_cell_info)):   
            encoder_cell.append(Cell(encoder_cell_info[i]))
        return encoder_cell

    def forward(self, input, protocol):
        jump_rate = protocol['jump_rate']
        x = pixel_unshuffle(input, jump_rate)
        for i in range(self.encoder_cell):          
            x = F.relu(self.encoder_cell[i](x))
        return x
        
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.encoder_info = self.make_encoder_info()
        self.encoder = self.make_encoder(self.encoder_info)
        
    def make_encoder_info(self):
        encoder_info = [        
        {'input_size':512,'output_size':128,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False},
        {'input_size':512,'output_size':128,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False},
        {'input_size':512,'output_size':128,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False},
        ]
        return encoder_info

    def make_encoder(self, encoder_info):
        encoder = nn.ModuleList([])
        for i in range(len(encoder_info)):
            encoder.append(EncoderCell(encoder_info[i]))
        return encoder
        
    def forward(self, input, protocol):
        mode = protocol['mode']
        depth = protocol['depth']
        x = L_to_RGB(input) if (mode == 'L') else input
        x = F.relu(self.conv0(x))
        for i in range(depth):
            x = self.encoder[i](x, protocol['encoder'][i])
        return x

class NormalEmbedding(nn.Module):
    def __init__(self):
        super(NormalEmbedding, self).__init__()
        self.conv_mean = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_logvar = nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, input, protocol):
        mu = self.conv_mean(x)
        logvar = self.conv_logvar(x)
        std = logvar.mul(0.5).exp_()
        eps = mu.new_zeros(mu.size()).normal_()
        x = eps.mul(std).add_(mu)
        return x
        
class DecoderCell(nn.Module):
    def __init__(self, decoder_cell_info):
        super(DecoderCell, self).__init__()
        self.decoder_cell_info = decoder_cell_info
        self.decoder_cell_info = self.make_decoder_cell(self.decoder_cell_info)
        
    def make_decoder_cell(self, decoder_cell_info):
        decoder_cell = nn.ModuleList([])
        for i in range(len(decoder_cell_info)):   
            decoder_cell.append(Cell(decoder_cell_info[i]))
        return decoder_cell

    def forward(self, input, protocol):
        jump_rate = protocol['jump_rate']
        for i in range(self.encoder_cell):          
            x = F.relu(self.encoder_cell[i](x))
        x = F.pixel_shuffle(input, jump_rate)
        return x
        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder_info = self.make_decoder_info()
        self.decoder = self.make_decoder(self.decoder_info)
        self.conv0 = nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0, bias=False)
        
    def make_decoder_info(self):
        decoder_info = [        
        {'input_size':128,'output_size':512,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False},
        {'input_size':128,'output_size':512,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False},
        {'input_size':128,'output_size':512,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False},
        ]
        return encoder_info

    def make_encoder(self, decoder_info):
        decoder = nn.ModuleList([])
        for i in range(len(encoder_info)):
            decoder.append(DecoderCell(decoder_info[i]))
        return decoder
        
    def forward(self, input, protocol):
        mode = protocol['mode']
        depth = protocol['depth']
        for i in range(depth):
            x = self.decoder[i](x, protocol['encoder'][i])
        x = F.sigmoid(self.conv0(x))
        x = RGB_to_L(x) if (mode == 'L') else x
        return x

class Codec(nn.Module):
    def __init__(self):
        super(Codec, self).__init__()
        self.encoder = Encoder()
        self.embedding = NormalEmbedding()
        self.decoder = Decoder()
        
    def compression_loss_fn(self, output, target):
        loss = F.l1_loss(output,target)
        return loss
        
class Classifier(nn.Module):
    def __init__(self, classes_size):
        super(Classifier, self).__init__()
        self.classes_size = classes_size
        self.classifier_info = self.make_classifier_info(classes_size)
        self.classifier = self.make_classifier(self.classifier_info)
        
    def make_classifier_info(self, classes_size):
        classifier_info = {'input_size':32,'output_size':classes_size}
        return classifier_info
        
    def make_classifier(self, classifier_info):
        classifier = nn.ParameterDict([])
        classifier['pi'] = nn.Parameter(torch.ones(classifier_info['output_size'])/classifier_info['output_size'])
        classifier['mu'] = nn.Parameter(torch.zeros(classifier_info['input_size'], classifier_info['output_size']))
        classifier['var'] = nn.Parameter(torch.zeros(classifier_info['input_size'], classifier_info['output_size']))
        return classifier
       
    def classification_loss_fn(self, output, target, protocol):
        return loss
        
    def forward(self, input, protocol):
        return x
        
class Joint(nn.Module):
    def __init__(self,classes_size):
        super(Joint, self).__init__()
        self.classes_size = classes_size
        self.codec = Codec()
        self.classifier = Classifier(classes_size)

	def loss_function(self, recon_x, x, z, mean, logvar):
		N = z.size()[0]
		D = z.size()[1]
		K = self.n_classes

		Z = z.unsqueeze(2).expand(N,D,K)
		z_mean_t = mean.unsqueeze(2).expand(N,D,K)
		z_log_var = logvar
		z_log_var_t = logvar.unsqueeze(2).expand(N,D,K)
		u_tensor3 = self.mu_k.unsqueeze(0).expand(N,D,K)
		lambda_tensor3 = self.var_k.unsqueeze(0).expand(N,D,K)
		theta_tensor2 = self.pi_k.unsqueeze(0).expand(N,K)
		

		p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
			(Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-10
		gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)
		
		BCE = -torch.sum(x*torch.log(torch.clamp(recon_x, min=1e-10))+(1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)), 1)

		logpzc = torch.sum(0.5*gamma*torch.sum(math.log(2*math.pi)+torch.log(lambda_tensor3)+\
			torch.exp(z_log_var_t)/lambda_tensor3 + (z_mean_t-u_tensor3)**2/lambda_tensor3, dim=1), dim=1)
		qentropy = -0.5*torch.sum(1+z_log_var+math.log(2*math.pi), 1)
		logpc = -torch.sum(torch.log(theta_tensor2)*gamma, 1)
		logqcx = torch.sum(torch.log(gamma)*gamma, 1)

		loss = torch.mean(BCE + logpzc + qentropy + logpc + logqcx)

		return loss

	def forward(self, input, protocol):
        tuning_param = protocol['tuning_param']
        num_iter = protocol['num_iter']
        loss = 0
        output = {}
        
        compression_loss = 0
        compression_output = 0
        compression_target = input['img']
        encoded = self.codec.encoder(input['img'],protocol)
        code = self.codec.embedding(encoded,protocol)
        compression_output = self.codec.decoder(code,protocol)
        compression_loss = compression_loss + self.codec.compression_loss_fn(compression_output,compression_target)
        output['compression'] = compression_output
        loss = loss + tuning_param['compression']*compression_loss
        
        if(tuning_param['classification'] > 0):
            classification_loss = 0        
            classification_output = 0
            classification_target = input['label']
            classification_output = self.classifier(code,protocol)
            loss = loss + tuning_param['classification']*classification_loss
            output['classification'] = classification_output
        output['loss'] = loss
		return output
        
        