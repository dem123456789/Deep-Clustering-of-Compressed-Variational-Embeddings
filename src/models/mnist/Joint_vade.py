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
z_dim=10

def buildNetwork(layers):
	net = []
	for i in range(1, len(layers)):
		net.append(nn.Linear(layers[i-1], layers[i]))
		net.append(nn.ReLU())

	return nn.Sequential(*net)

class Joint_vade(nn.Module):
	def __init__(self,classes_size,binary=True):
		super(Joint_vade, self).__init__()
		self.classes_size = classes_size
		self.encoder = buildNetwork([1024]+[500,500,2000])
		self.decoder = buildNetwork([z_dim]+[2000,500,500])
		self.enc_mean = nn.Linear([500,500,2000][-1], z_dim)
		self.enc_logvar = nn.Linear([500,500,2000][-1], z_dim)
		self._dec_act = None
		if binary:
			self._dec_act = nn.Sigmoid()

		self.h0 = nn.Linear(500,1024)

		self.param = nn.ParameterDict({
			'mu': nn.Parameter(torch.zeros(z_dim, self.classes_size)),
			'var': nn.Parameter(torch.ones(z_dim, self.classes_size)),
			'pi': nn.Parameter(torch.ones(self.classes_size)/self.classes_size)
			})

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

	def decode(self, input):
		h = self.decoder(input)
		x = self.h0(h)
		if self._dec_act is not None:
			x = self._dec_act(x)
		return x
	
	def classifier(self, input, protocol):		
		z = input['code'].view(input['code'].size(0),-1,1)
		q_c_z = torch.exp(torch.log(self.param['pi']) - torch.sum(0.5*torch.log(2*math.pi*self.param['var']) +\
			(z-self.param['mu'])**2/(2*self.param['var']),dim=1)) + 1e-10
		q_c_z = q_c_z/torch.sum(q_c_z,dim=1,keepdim=True) #Nx1		
		return q_c_z

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
			loss = loss + torch.sum(q_c_z*(torch.log(q_c_z)-torch.log(self.param['pi'])),1)
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
			loss = loss + torch.sum(0.5*q_c_z*torch.sum(math.log(2*math.pi)+torc.log(self.param['var'])+\
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
		
		img = input['img'].view(-1,1024).float()
		h = self.encoder(img)
		mu = self.enc_mean(h)
		logvar = self.enc_logvar(h)
		code = self.reparameterize(mu,logvar)
		param = {'mu':mu,'logvar':logvar}

		output['compression'] = {'code':code, 'param':param}

		compression_output = self.decode(code)
		# for name, param in self.decoder.named_parameters():
		# 	if param.requires_grad:
		# 		print(name, param.data,param.grad)
		output['compression']['img'] = compression_output.view(input['img'].size())
		
		if(protocol['tuning_param']['classification'] > 0):
			classification_output = self.classifier({'code':code},protocol)
			output['classification'] = classification_output
		
		output['loss'] = self.loss_fn(input,output,protocol)

		return output
