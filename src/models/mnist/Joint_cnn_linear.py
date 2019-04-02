import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import config
from modules import Quantize
from torch.autograd import Variable
from functions import pixel_unshuffle
from data import extract_patches_2d, reconstruct_from_patches_2d
from utils import RGB_to_L, L_to_RGB,dict_to_device

config.init()
device = config.PARAM['device']
code_size = 128
z_dim = 10

class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
	def forward(self, input, size=128):
		return input.view(input.size(0), size, 4, 4)

def buildCell(cell_info):
	cell = nn.Conv2d(cell_info['input_size'], cell_info['output_size'], kernel_size=cell_info['kernel_size'],
		stride=cell_info['stride'], padding=cell_info['padding'], dilation=cell_info['dilation'], bias=cell_info['bias'])
	return cell

def buildNetwork(cell_info):
	net = nn.ModuleList([])
	for i in range(len(cell_info)):   
		net.append(buildCell(cell_info[i]))
	return net 

class Joint_cnn_linear(nn.Module):
	def __init__(self,classes_size):
		super(Joint_cnn_linear, self).__init__()
		self.classes_size = classes_size
		self.conv01 = nn.Conv2d(3, 128, kernel_size=1, stride=1, padding=0, bias=False)
		self.encoder = buildNetwork(cell_info = [        
		{'input_size':512,'output_size':128,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False},
		{'input_size':512,'output_size':128,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False},
		{'input_size':512,'output_size':128,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False},
		])
		self.conv12 = nn.Conv2d(128, code_size, kernel_size=1, stride=1, padding=0, bias=False)
		self.flatten = Flatten()
		self.unflatten = UnFlatten()

		self.fc_mean = nn.Linear(2048, z_dim)
		self.fc_logvar = nn.Linear(2048, z_dim)
		self.fc0= nn.Linear(z_dim, 2048)

		self.decoder = buildNetwork(cell_info = [        
		{'input_size':128,'output_size':512,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False},
		{'input_size':128,'output_size':512,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False},
		{'input_size':128,'output_size':512,'kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':False},
		])
		self.conv21 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)
		self.conv10 = nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0, bias=False)

		self.param = nn.ParameterDict({
			'mu': nn.Parameter(torch.zeros(z_dim, self.classes_size)),
			'var': nn.Parameter(torch.ones(z_dim, self.classes_size)),
			'pi': nn.Parameter(torch.ones(self.classes_size)/self.classes_size)
			})

	def encode(self, input, protocol):
		x = L_to_RGB(input) if (protocol['mode'] == 'L') else input
		x = torch.tanh(self.conv01(x))
		for i in range(protocol['depth']):
			x = pixel_unshuffle(x, protocol['jump_rate'])
			x = torch.tanh(self.encoder[i](x))
		x = torch.tanh(self.conv12(x))
		return x

	def decode(self, input, protocol):
		x = torch.tanh(self.conv21(input))
		for i in range(protocol['depth']):
			x = torch.tanh(self.decoder[i](x))
			x = F.pixel_shuffle(x, protocol['jump_rate'])
		x = torch.tanh(self.conv10(x))
		x = RGB_to_L(x) if (protocol['mode'] == 'L') else x
		return x

	def reparameterize(self, mu, logvar):
		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = Variable(std.data.new(std.size()).normal_())
			  # num = np.array([[ 1.096506  ,  0.3686553 , -0.43172026,  1.27677995,  1.26733758,
			  #       1.30626082,  0.14179629,  0.58619505, -0.76423112,  2.67965817]], dtype=np.float32)
			  # num = np.repeat(num, mu.size()[0], axis=0)
			  # eps = Variable(torch.from_numpy(num))
			return eps.mul(std).add_(mu)
		else:
			return mu

	def classifier(self, input, protocal):
		z = input['code'].view(input['code'].size(0),-1,1)
		q_c_z = torch.exp(self.param['pi'].log() - torch.sum(0.5*torch.log(2*math.pi*self.param['var']) +\
			(z-self.param['mu'])**2/(2*self.param['var']),dim=1)) + 1e-10
		# + 10*np.finfo(np.float32).eps
		q_c_z = q_c_z/q_c_z.sum(dim=1,keepdim=True)
		return q_c_z
		
	def classification_loss_fn(self, input, output, protocol):
		loss = 0
		if(protocol['tuning_param']['classification'] > 0): 
			q_c_z = output['classification']
			loss = loss + (q_c_z*(q_c_z.log()-self.param['pi'].log())).sum(dim=1)
			# loss = loss.sum()/input['img'].numel()
		return loss
	 
	def compression_loss_fn(self, input, output, protocol):
		loss = 0
		if(protocol['tuning_param']['compression'] > 0):
			loss = loss + F.binary_cross_entropy(output['compression']['img'],input['img'],reduction='none').view(input['img'].size(0),-1).sum(dim=1)
		q_mu = output['compression']['param']['mu'].view(input['img'].size(0),-1,1)
		q_logvar = output['compression']['param']['logvar'].view(input['img'].size(0),-1,1)
		if(protocol['tuning_param']['classification'] > 0):
			q_c_z = output['classification']
			loss = loss + torch.sum(0.5*q_c_z*torch.sum(math.log(2*math.pi)+torch.log(self.param['var'])+\
				torch.exp(q_logvar)/self.param['var'] + (q_mu-self.param['mu'])**2/self.param['var'], dim=1), dim=1)
			loss = loss + (-0.5*torch.sum(1+q_logvar+math.log(2*math.pi), 1)).squeeze(1)
		# loss = loss.sum()/input['img'].numel()
		return loss

	def loss_fn(self, input, output, protocol):
		compression_loss = self.compression_loss_fn(input,output,protocol)
		classification_loss = self.classification_loss_fn(input,output,protocol)
		loss = protocol['tuning_param']['compression']*compression_loss + protocol['tuning_param']['classification']*classification_loss
		loss = torch.mean(loss)
		return loss

	def forward(self, input, protocol):
		output = {}
		
		img = (input['img'] - 0.5) * 2
		encoded = self.flatten(self.encode(img,protocol))
		
		mu = self.fc_mean(encoded)
		logvar = self.fc_logvar(encoded)
		code = self.reparameterize(mu, logvar)

		param = {'mu':mu,'logvar':logvar}
		output['compression'] = {'code':code, 'param':param}

		if(protocol['tuning_param']['compression'] > 0):
			compression_output = self.decode(self.unflatten(self.fc0(code)),protocol)
			compression_output = (compression_output + 1) * 0.5
			output['compression']['img'] = compression_output
	   
		if(protocol['tuning_param']['classification'] > 0):
			classification_output = self.classifier({'code':code},protocol)
			output['classification'] = classification_output
		output['loss'] = self.loss_fn(input,output,protocol)
		return output
		
