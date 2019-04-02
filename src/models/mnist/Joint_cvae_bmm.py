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
from bmm_implement import BMM


config.init()
device = config.PARAM['device']
z_dim=32
z_category = 2
classes_size = 10

def sample_gumbel(shape, eps=1e-20):
	U = torch.rand(shape)
	if torch.cuda.is_available():
		U = U.cuda()
	return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
	y = logits + sample_gumbel(logits.size())
	return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
	"""
	ST-gumple-softmax
	input: [*, n_class]
	return: flatten --> [*, n_class] an one-hot vector
	"""
	y = gumbel_softmax_sample(logits, temperature)
	
	if not hard:
		return y.view(-1, z_dim * z_category)

	shape = y.size()
	_, ind = y.max(dim=-1)
	y_hard = torch.zeros_like(y).view(-1, shape[-1])
	y_hard.scatter_(1, ind.view(-1, 1), 1)
	y_hard = y_hard.view(*shape)
	# Set gradients w.r.t. y_hard gradients w.r.t. y
	y_hard = (y_hard - y).detach() + y
	return y_hard.view(-1, z_dim * z_category)

def buildNetwork(layers):
	net = []
	for i in range(1, len(layers)):
		net.append(nn.Linear(layers[i-1], layers[i]))
		net.append(nn.ReLU())

	return nn.Sequential(*net)

class Joint_cvae_bmm(nn.Module):
	def __init__(self,classes_size):
		super(Joint_cvae_bmm, self).__init__()
		self.classes_size = classes_size
		self.temp = config.PARAM['temp']

		self.encoder = buildNetwork([1024,500,500,2000])
		self.enc_q = nn.Linear(2000, z_dim*z_category)
		self.decoder = buildNetwork([z_dim*z_category,2000,500,500])
		self.h0 = nn.Linear(500,1024)

		self.param = nn.ParameterDict({
			'mean': nn.Parameter(torch.ones(z_dim, self.classes_size)/z_category),
			'pi': nn.Parameter(torch.ones(self.classes_size)/self.classes_size)
			})

	def decode(self, input):
		x = self.decoder(input)
		x = torch.tanh(self.h0(x))
		return x

	def reparameterize(self, logits, temperature):
		if self.training:
			return gumbel_softmax(logits, temperature, hard=False)
		else:
			return gumbel_softmax(logits, temperature, hard=True)

	def classifier(self, input, protocol):		
		# z = torch.argmax(input['code'].view(input['code'].size(0),z_dim,z_category,1),dim=2) #NxDx2x1
		# z = z.float().view(input['code'].size(0),z_dim,1)
		# z = input['code'].view(input['code'].size(0),-1,1)
		z = input['code'].view(input['code'].size(0),z_dim,z_category,1) #Nx(Dx2)x1
		q_c_z = torch.exp(torch.log(self.param['pi'])+torch.sum(z[:,:,1,:]*torch.log(input['param']['mean'])+z[:,:,0,:]*torch.log(1-input['param']['mean']),dim=1))+1e-10
		q_c_z = q_c_z/q_c_z.sum(dim=1,keepdim=True) #NxH
		if (torch.isnan(q_c_z).sum() and not config.PARAM['printed']):
			print(torch.isnan(torch.log(self.param['pi'])),
				torch.isnan(z[:,:,1,:]*torch.log(input['param']['mean'])))
		return q_c_z

	def classification_loss_fn(self, input, output, protocol):
		loss = 0
		if(protocol['tuning_param']['classification'] > 0): 
			q_c_z = output['classification'] #NxH
			loss = loss + (q_c_z*(q_c_z.log()-self.param['pi'].log())).sum(dim=1)
			# loss = loss.sum()/input['img'].numel()
		return loss

	def compression_loss_fn(self, input, output, protocol):
		loss = 0
		if(protocol['tuning_param']['compression'] > 0):
			loss = loss + F.binary_cross_entropy(output['compression']['img'],input['img'],reduction='none').view(input['img'].size(0),-1).sum(dim=1)
		q_y = output['compression']['param']['qy'].view(input['img'].size(0),z_dim,z_category,1) #NxDx2x1
		# z = output['compression']['code'].view(output['compression']['code'].size(0),z_dim,z_category,1)
		if(protocol['tuning_param']['classification'] > 0):
			q_c_z = output['classification'] #NxH
			loss = loss - torch.sum(q_c_z*torch.sum(q_y[:,:,1,:]*torch.log(input['param']['mean']*z_category)+q_y[:,:,0,:]*torch.log((1-input['param']['mean'])*z_category),dim=1),dim=1)
			if (torch.isnan(loss).sum() and not config.PARAM['printed']):
				print(torch.isnan(q_y[:,:,1,:]*torch.log(input['param']['mean'])).sum())
			loss = loss + torch.sum(output['compression']['param']['qy']*torch.log(output['compression']['param']['qy']*z_category+1e-10), 1)
		# loss = loss.sum()/input['img'].numel()
		return loss
	
	def loss_fn(self, input, output, protocol):
		compression_loss = self.compression_loss_fn(input,output,protocol)
		classification_loss = self.classification_loss_fn(input,output,protocol)
		loss = protocol['tuning_param']['compression']*compression_loss + protocol['tuning_param']['classification']*classification_loss
		loss = torch.mean(loss)
		if (torch.isnan(loss) and not config.PARAM['printed']):
			if (torch.isnan(compression_loss).sum()):
				print('compression_loss is nan')
			if (torch.isnan(classification_loss).sum()):
				print('classification_loss is nan')
			print(input['param']['mean'])
			print(output['classification'])
			config.PARAM['printed'] = True
		loss = torch.mean(loss)
		return loss 

	def forward(self, input, protocol):
		output = {}
		
		input['param'] = {'mean':torch.sigmoid(self.param['mean']),'pi':self.param['pi']}
		# input['param'] = {'mean':self.param['mean'],'pi':self.param['pi']}

		img = (input['img'].view(-1,1024) - 0.5) * 2
		h = self.encoder(img)
		q = self.enc_q(h)
		qy = q.view(q.size(0),z_dim,z_category)

		code = self.reparameterize(qy,self.temp)
		param = {'qy':F.softmax(qy, dim=-1).reshape(*q.size())} #(p(z_k=0),p(z_k=1)) k=1,2,...,32

		output['compression'] = {'code':code, 'param':param}

		if(protocol['tuning_param']['compression'] > 0):
			compression_output = self.decode(code)
			compression_output = (compression_output + 1) * 0.5
			output['compression']['img'] = compression_output.view(input['img'].size())
		if(protocol['tuning_param']['classification'] > 0):
			classification_output = self.classifier({'code':code,'param':input['param']},protocol)
			output['classification'] = classification_output
		output['loss'] = self.loss_fn(input,output,protocol)

		return output
