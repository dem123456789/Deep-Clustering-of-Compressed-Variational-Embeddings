import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch.autograd import Variable
import math

def buildNetwork(layers, activation="relu", dropout=0):
	net = []
	for i in range(1, len(layers)):
		net.append(nn.Linear(layers[i-1], layers[i]))
		if activation=="relu":
			net.append(nn.ReLU())
		elif activation=="sigmoid":
			net.append(nn.Sigmoid())
		if dropout > 0:
			net.append(nn.Dropout(dropout))
	return nn.Sequential(*net)

class VaDE(nn.Module):
	def __init__(self, input_dim=784, z_dim=10, n_centroids=10, binary=True,
				 encodeLayer=[500,500,2000], decodeLayer=[2000,500,500]):
		super(self.__class__, self).__init__()
		self.z_dim = z_dim
		self.n_centroids = n_centroids
		self.encoder = buildNetwork([input_dim] + encodeLayer)
		self.decoder = buildNetwork([z_dim] + decodeLayer)
		self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
		self._enc_log_sigma = nn.Linear(encodeLayer[-1], z_dim)
		self._dec = nn.Linear(decodeLayer[-1], input_dim)
		self._dec_act = None
		if binary:
			self._dec_act = nn.Sigmoid()
		
		self.create_gmmparam(n_centroids, z_dim)
	
	def create_gmmparam(self, n_centroids, z_dim):
		self.theta_p = nn.Parameter(torch.ones(n_centroids)/n_centroids)
		self.u_p = nn.Parameter(torch.zeros(z_dim, n_centroids))
		self.lambda_p = nn.Parameter(torch.ones(z_dim, n_centroids))
	
	def initialize_gmm(self, dataloader):
		# use_cuda = torch.cuda.is_available()
		# if use_cuda:
		# 	self.cuda()
		self.eval()
		data = []
		for batch_idx, (inputs, _) in enumerate(dataloader):
			inputs = inputs.view(-1, 784)
			z, outputs, mu, logvar = self.forward(inputs)
			data.append(z.data.cpu().numpy())
		data = np.concatenate(data)
		gmm = GaussianMixture(n_components=self.n_centroids,covariance_type='diag')
		gmm.fit(data)
		self.u_p.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
		self.lambda_p.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))
	
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
			
	def decode(self, z):
		h = self.decoder(z)
		x = self._dec(h)
		if self._dec_act is not None:
			x = self._dec_act(x)
		return x

	def get_gamma(self, z, z_mean, z_log_var):
		
		Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], self.n_centroids) # NxDxK
		z_mean_t = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], self.n_centroids)
		z_log_var_t = z_log_var.unsqueeze(2).expand(z_log_var.size()[0], z_log_var.size()[1], self.n_centroids)
		u_tensor3 = self.u_p.unsqueeze(0).expand(z.size()[0], self.u_p.size()[0], self.u_p.size()[1]) # NxDxK
		lambda_tensor3 = self.lambda_p.unsqueeze(0).expand(z.size()[0], self.lambda_p.size()[0], self.lambda_p.size()[1])
		theta_tensor2 = self.theta_p.unsqueeze(0).expand(z.size()[0], self.n_centroids) # NxK

		p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
			(Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-10 # NxK
		gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

		return gamma

	def loss_function(self, recon_x, x, z, z_mean, z_log_var):
		Z = z.unsqueeze(2).expand(z.size()[0], z.size()[1], self.n_centroids) # NxDxK
		z_mean_t = z_mean.unsqueeze(2).expand(z_mean.size()[0], z_mean.size()[1], self.n_centroids)
		z_log_var_t = z_log_var.unsqueeze(2).expand(z_log_var.size()[0], z_log_var.size()[1], self.n_centroids)
		u_tensor3 = self.u_p.unsqueeze(0).expand(z.size()[0], self.u_p.size()[0], self.u_p.size()[1]) # NxDxK
		lambda_tensor3 = self.lambda_p.unsqueeze(0).expand(z.size()[0], self.lambda_p.size()[0], self.lambda_p.size()[1])
		theta_tensor2 = self.theta_p.unsqueeze(0).expand(z.size()[0], self.n_centroids) # NxK

		p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
			(Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-10 # NxK
		gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True) # NxK
		
		BCE = -torch.sum(x*torch.log(torch.clamp(recon_x, min=1e-10))+
			(1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)), 1)
		logpzc = torch.sum(0.5*gamma*torch.sum(math.log(2*math.pi)+torch.log(lambda_tensor3)+\
			torch.exp(z_log_var_t)/lambda_tensor3 + (z_mean_t-u_tensor3)**2/lambda_tensor3, dim=1), dim=1)
		qentropy = -0.5*torch.sum(1+z_log_var+math.log(2*math.pi), 1)
		logpc = -torch.sum(torch.log(theta_tensor2)*gamma, 1)
		logqcx = torch.sum(torch.log(gamma)*gamma, 1)

	# Normalise by same number of elements as in reconstruction
		loss = torch.mean(BCE + logpzc + qentropy + logpc + logqcx)

		loss = loss / x.size()[0]

		return loss

	def forward(self, x):
		h = self.encoder(x)
		mu = self._enc_mu(h)
		logvar = self._enc_log_sigma(h)
		z = self.reparameterize(mu, logvar)
		return z, self.decode(z), mu, logvar

