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
z_dim = 10

class CoderCell(nn.Module):
  def __init__(self, coder_cell_info):
	  super(CoderCell, self).__init__()
	  self.coder_cell_info = coder_cell_info
	  self.coder_cell = self.make_coder_cell()

  def make_coder_cell(self):
	  coder_cell = nn.ModuleList([])
	  for i in range(1,len(self.coder_cell_info)):   
		  coder_cell.append(nn.Linear(self.coder_cell_info[i-1], self.coder_cell_info[i]))
		  coder_cell.append(nn.ReLU())
	  return coder_cell

  def forward(self, input):
	  x = input
	  for i in range(len(self.coder_cell)):
		  x = self.coder_cell[i](x)
	  return x
		
class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.h0 = nn.Linear(1024,500)
		self.encoder_info = [500, 500, 2000]
		self.encoder = CoderCell(self.encoder_info)
		
	def forward(self, input, protocol):
		x = torch.relu(self.h0(input))
		x = self.encoder(x)
		return x
			   
class NormalEmbedding(nn.Module):
	def __init__(self):
		super(NormalEmbedding, self).__init__()
		self.enc_mean = nn.Linear(2000, z_dim)
		self.enc_logvar = nn.Linear(2000, z_dim)
		
	def forward(self, input, protocol):
		mu = self.enc_mean(input)
		logvar = self.enc_logvar(input)
		std = logvar.mul(0.5).exp()
		eps = mu.new_zeros(mu.size()).normal_()
		z = eps.mul(std).add_(mu)
		param = {'mu':mu,'var':logvar.exp()}
		return z, param

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self.h0 = nn.Linear(z_dim, 2000)
		self.decoder_info = [2000, 500, 500]
		self.decoder = CoderCell(self.decoder_info)
		self.h1 = nn.Linear(500,1024)
		
	def forward(self, input, protocol):
		x = torch.relu(self.h0(input))
		x = self.decoder(x)
		x = torch.tanh(self.h1(x))
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

	def classification_loss_fn(self, input, output, protocol):
		loss = 0
		if(protocol['tuning_param']['classification'] > 0): 
			q_c_z = output['classification']
			loss = loss + (q_c_z*(q_c_z.log()-input['param']['pi'].log())).sum(dim=1)
			loss = loss.sum()/input['img'].numel()
		return loss
		
	def forward(self, input, protocol):
		z = input['code'].view(input['code'].size(0),-1,1)
		
		q_c_z = torch.exp(torch.log(input['param']['pi']) - torch.sum(0.5*torch.log(2*math.pi*input['param']['var']) +\
			(z-input['param']['mu'])**2/(2*input['param']['var']),dim=1))+ 1e-10
		q_c_z = q_c_z/q_c_z.sum(dim=1,keepdim=True)

		return q_c_z

class Joint_linear(nn.Module):
	def __init__(self,classes_size):
		super(Joint_linear, self).__init__()
		self.classes_size = classes_size
		self.codec = Codec(classes_size)
		self.classifier = Classifier(classes_size)
		self.create_gmmparam()

	def create_gmmparam(self):
		self.param = nn.ParameterDict({
			'mu': torch.zeros(z_dim, self.classes_size),
			'var': torch.ones(z_dim, self.classes_size),
			'pi': torch.ones(self.classes_size)/self.classes_size
			})
		
	def init_param_protocol(self,dataset,randomGen):
		protocol = {}
		protocol['tuning_param'] = config.PARAM['tuning_param'].copy()
		protocol['tuning_param']['classification'] = 0
		protocol['init_param_mode'] = 'gmm'
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
				z = output['compression']['code']
				Z = torch.cat((Z,z),0) if i > 0 else z
			if(protocol['init_param_mode'] == 'random'):
				C = torch.rand(Z.size(0), protocol['classes_size'],device=device)
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
				gm = GaussianMixture(n_components=protocol['classes_size'], covariance_type='diag', random_state=protocol['randomGen']).fit(Z)
				self.param['mu'] = nn.Parameter(torch.tensor(gm.means_.T).float().to(device))
				self.param['var'] = nn.Parameter(torch.tensor(gm.covariances_.T).float().to(device))
				self.param['pi'] = nn.Parameter(torch.tensor(gm.weights_).float().to(device))
				self.param['mu'].data.copy_(torch.tensor(gm.means_.T).float().to(device))
				self.param['var'].data.copy_(torch.tensor(gm.covariances_.T).float().to(device))
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
		img = img.view(-1,1024)
		encoded = self.codec.encoder(img,protocol)
		code, param = self.codec.embedding(encoded,protocol)
		output['compression'] = {'code':code, 'param':param}
		
		if(protocol['tuning_param']['compression'] > 0):
			compression_output = self.codec.decoder(code,protocol)
			compression_output = (compression_output + 1) * 0.5
			output['compression']['img'] = compression_output.view(input['img'].size())
	   
		if(protocol['tuning_param']['classification'] > 0):
			classification_output = self.classifier({'code':code, 'param':input['param']},protocol)
			output['classification'] = classification_output
		output['loss'] = self.loss_fn(input,output,protocol)
		return output
		
		
