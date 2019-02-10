
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
z_dim=10

def buildNetwork(layers):
	net = []
	for i in range(1, len(layers)):
		net.append(nn.Linear(layers[i-1], layers[i]))
		net.append(nn.ReLU())

	return nn.Sequential(*net)

class Joint_vade(nn.Module):
	def __init__(self,classes_size):
		super(Joint_vade, self).__init__()
		self.classes_size = classes_size
		self.encoder = buildNetwork([1024,500,500,2000])
		self.enc_mean = nn.Linear(2000, z_dim)
		self.enc_logvar = nn.Linear(2000, z_dim)

		self.decoder = buildNetwork([z_dim,2000,500,500])
		self.h0 = nn.Linear(500,1024)

		self.param = nn.ParameterDict({
			'mu': nn.Parameter(torch.zeros(z_dim, self.classes_size)),
			'var': nn.Parameter(torch.ones(z_dim, self.classes_size)),
			'pi': nn.Parameter(torch.ones(self.classes_size)/self.classes_size)
			})

	def create_gmmparam(self):
		self.theta_p = nn.Parameter(torch.ones(self.classes_size)/self.classes_size) #requires_grad=True
		self.mu_p = nn.Parameter(torch.zeros(z_dim, self.classes_size))
		self.var_p = nn.Parameter(torch.ones(z_dim, self.classes_size))

	def decode(self, input):
		x = self.decoder(input)
		x = torch.sigmoid(self.h0(x))
		return x

	def classifier(self, input, protocol):
		
		z = input['code'].view(input['code'].size(0),-1,1)
		q_c_z = torch.exp(torch.log(self.param['pi']) - torch.sum(0.5*torch.log(2*math.pi*self.param['var']) +\
			(z-self.param['mu'])**2/(2*self.param['var']),dim=1)) + 1e-20
		if not config.PARAM['printed']:
			print(10*np.finfo(np.float32).eps)
			config.PARAM['printed'] = True
		q_c_z = q_c_z/q_c_z.sum(dim=1,keepdim=True) #Nx1
		
		return q_c_z

	def init_param_protocol(self,dataset,randomGen):
		protocol = {}
		protocol['tuning_param'] = config.PARAM['tuning_param'].copy()
		protocol['tuning_param']['classification'] = 0
		protocol['init_param_mode'] = 'kmeans'
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
				self.param['mu'].data.copy_(Z.t().matmul(C)/nk)
				self.param['var'].data.copy_((Z**2).t().matmul(C)/nk - 2*self.param['mu']*Z.t().matmul(C)/nk + self.param['mu']**2)
			elif(protocol['init_param_mode'] == 'kmeans'):
				from sklearn.cluster import KMeans
				C = torch.rand(Z.size(0), protocol['classes_size'],device=device)
				C = C.new_zeros(C.size())
				km = KMeans(n_clusters=protocol['classes_size'], n_init=1, random_state=protocol['randomGen']).fit(Z)
				C[torch.arange(C.size(0)), torch.tensor(km.labels_).long()] = 1
				nk = C.sum(dim=0,keepdim=True) + 10*np.finfo(np.float32).eps
				self.param['mu'].data.copy_(Z.t().matmul(C)/nk)
				self.param['var'].data.copy_((Z**2).t().matmul(C)/nk - 2*self.param['mu']*Z.t().matmul(C)/nk + self.param['mu']**2)
			elif(protocol['init_param_mode'] == 'gmm'):
				from sklearn.mixture import GaussianMixture
				gm = GaussianMixture(n_components=protocol['classes_size'], covariance_type='diag', random_state=protocol['randomGen']).fit(Z)
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

	def loss_fn_base(self, input, output, protocol):
		N = output['compression']['code'].size()[0]
		D = output['compression']['code'].size()[1]
		K = self.classes_size

		Z = output['compression']['code'].unsqueeze(2).expand(N, D, K) # NxDxK
		z_mean_t = output['compression']['param']['mu'].unsqueeze(2).expand(N, D, K)
		z_log_var_t = output['compression']['param']['logvar'].unsqueeze(2).expand(N, D, K)

		u_tensor3 = self.mu_p.unsqueeze(0).expand(N,D,K) # NxDxK
		lambda_tensor3 = self.var_p.unsqueeze(0).expand(N,D,K)
		theta_tensor2 = self.theta_p.unsqueeze(0).expand(N, K) # NxK

		p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
			(Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-10 # NxK
		gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True) # NxK

		BCE = -torch.sum(input['img'].view(-1,1024)*torch.log(torch.clamp(output['compression']['img'].view(-1,1024), min=1e-10))+
			(1-input['img'].view(-1,1024))*torch.log(torch.clamp(1-output['compression']['img'].view(-1,1024), min=1e-10)), 1)
		logpzc = torch.sum(0.5*gamma*torch.sum(math.log(2*math.pi)+torch.log(lambda_tensor3)+
			torch.exp(z_log_var_t)/lambda_tensor3 + (z_mean_t-u_tensor3)**2/lambda_tensor3, dim=1), dim=1)
		
		qentropy = -0.5*torch.sum(1+output['compression']['param']['logvar']+math.log(2*math.pi), 1)

		logpc = -torch.sum(torch.log(theta_tensor2)*gamma, 1)
		logqcx = torch.sum(torch.log(gamma)*gamma, 1)

		loss = torch.mean(BCE+logpzc+qentropy+logpc+logqcx)
		return loss

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
		
		img = input['img'].view(-1,1024)
		h = self.encoder(img)
		mu = self.enc_mean(h)
		logvar = self.enc_logvar(h)

		std = logvar.mul(0.5).exp_()
		eps = mu.new_zeros(mu.size()).normal_()
		code = eps.mul(std).add_(mu)
		param = {'mu':mu,'logvar':logvar}

		output['compression'] = {'code':code, 'param':param}

		compression_output = self.decode(code)
		output['compression']['img'] = compression_output.view(input['img'].size())
		
		if(protocol['tuning_param']['classification'] > 0):
			classification_output = self.classifier({'code':code},protocol)
			output['classification'] = classification_output
		
		output['loss'] = self.loss_fn(input,output,protocol)

		return output
