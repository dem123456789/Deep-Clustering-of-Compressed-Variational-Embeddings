import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from sklearn.mixture import GaussianMixture
from utils import *
import math


class VAE(nn.Module):

	def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,n_classes):

		super(self.__class__, self).__init__()

		assert type(encoder_layer_sizes) == list
		assert type(latent_size) == int
		assert type(decoder_layer_sizes) == list

		self.latent_size = latent_size
		self.n_classes = n_classes

		self.encoder = Encoder(encoder_layer_sizes, latent_size)
		self.decoder = Decoder(decoder_layer_sizes, latent_size)

		self.create_gmmparam(n_classes, latent_size)

	def inference(self, n=1):

		batch_size = n
		z = torch.randn([batch_size, self.latent_size])

		recon_x = self.decoder(z)

		return recon_x

	def create_gmmparam(self, K, D):
		self.pi_k = nn.Parameter(torch.ones(K)/K)
		self.mu_k = nn.Parameter(torch.zeros(D, K))
		self.var_k = nn.Parameter(torch.ones(D, K))

	def initialize_gmm(self, dataloader):
		# use_cuda = torch.cuda.is_available()
		# if use_cuda:
		# 	self.cuda()
		self.eval()
		data = []
		for batch_idx, (inputs, _) in enumerate(dataloader):
			inputs = inputs.view(-1, 784)
			outputs, mu, logvar, z = self.forward(inputs)
			data.append(z.data.cpu().numpy())
		data = np.concatenate(data)
		
		gmm = GaussianMixture(n_components=self.n_classes,covariance_type='diag')
		gmm.fit(data)
		self.mu_k.data.copy_(torch.from_numpy(gmm.means_.T.astype(np.float32)))
		self.var_k.data.copy_(torch.from_numpy(gmm.covariances_.T.astype(np.float32)))
	
	# def EM_q_c_z(self,z):

	# 	N = z.size()[0]
	# 	D = z.size()[1]
	# 	K = self.n_classes
	# 	z = z.unsqueeze(2).expand(N,D,K) #NxDxK
	# 	pi_k = self.pi_k # 1xK
	# 	mu_k = self.mu_k # DxK
	# 	var_k = self.var_k # DxK

	# 	q_c_z = torch.exp(torch.log(pi_k)-0.5*torch.sum((z-mu_k)**2 / var_k +torch.log(2*math.pi*var_k),dim=1))

	# 	qcz = q_c_z / torch.sum(q_c_z, dim=1, keepdim=True) #NxK

	# 	return qcz

	def get_gamma(self, z):

		# from p(c=k),p(z|c) mean and variance and observation z to q(c|z)
		N = z.size()[0]
		D = z.size()[1]
		K = self.n_classes

		Z = z.unsqueeze(2).expand(N,D,K) # NxDxK
		
		u_tensor3 = self.mu_k.unsqueeze(0).expand(N,D,K) # NxDxK
		lambda_tensor3 = self.var_k.unsqueeze(0).expand(N,D,K)
		theta_tensor2 = self.pi_k.unsqueeze(0).expand(N,K) # NxK

		p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
			(Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-10 # NxK
		gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True)

		return gamma

	def loss_function(self, recon_x, x, z, mean, logvar):
		N = z.size()[0]
		D = z.size()[1]
		K = self.n_classes #NxK

		Z = z.unsqueeze(2).expand(N,D,K) # NxDxK
		z_mean_t = mean.unsqueeze(2).expand(N,D,K)
		z_log_var = logvar
		z_log_var_t = logvar.unsqueeze(2).expand(N,D,K)
		u_tensor3 = self.mu_k.unsqueeze(0).expand(N,D,K) # NxDxK
		lambda_tensor3 = self.var_k.unsqueeze(0).expand(N,D,K)
		theta_tensor2 = self.pi_k.unsqueeze(0).expand(N,K) # NxK
		
		#gamma = self.EM_q_c_z(z) #NxK

		p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
			(Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-10 # NxK
		gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True) # NxK
		
		BCE = -torch.sum(x*torch.log(torch.clamp(recon_x, min=1e-10))+(1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)), 1)

		logpzc = torch.sum(0.5*gamma*torch.sum(math.log(2*math.pi)+torch.log(lambda_tensor3)+\
			torch.exp(z_log_var_t)/lambda_tensor3 + (z_mean_t-u_tensor3)**2/lambda_tensor3, dim=1), dim=1)
		qentropy = -0.5*torch.sum(1+z_log_var+math.log(2*math.pi), 1)
		logpc = -torch.sum(torch.log(theta_tensor2)*gamma, 1)
		logqcx = torch.sum(torch.log(gamma)*gamma, 1)

	# Normalise by same number of elements as in reconstruction
		loss = torch.mean(BCE + logpzc + qentropy + logpc + logqcx)

		return loss

	# def loss_function(self, recon_x, x, z, mean, logvar):

	# 	N = z.size()[0]
	# 	D = z.size()[1]
	# 	K = self.n_classes #NxK
	# 	q_c_z = self.EM_q_c_z(z) #NxK
	# 	logvar_k= torch.log(self.var_k) #DxK
	# 	mu_k = self.mu_k #DxK
	# 	pi_k = self.pi_k #1xK
	# 	#logvar  NxD

	   
	# 	BCE = F.binary_cross_entropy(recon_x, x) #average of N*784

	# 	KLD = torch.zeros(N)

	# 	for i in range(K):
	# 		KLD_c = -0.5 * torch.sum(1 + (logvar-logvar_k[:,i])- (mean-mu_k[:,i]).pow(2) / logvar_k[:,i].exp() - logvar.exp()/logvar_k[:,i].exp(),dim=1)
	# 		KLD = KLD + q_c_z[:,i] * KLD_c
		
	# 	KLD = torch.sum(KLD)
	# 	KLD /= N *784
			
	# 	log_ratio = torch.log(pi_k+1e-20)-torch.log(q_c_z+1e-20)
	# 	KLD_2 = torch.sum(q_c_z * log_ratio)
	# 	KLD_2 /= N*784

	# 	return BCE+KLD+KLD_2

	def forward(self, x):

		batch_size = x.size(0)

		means, logvars = self.encoder(x)

		std = torch.exp(0.5 * logvars)
		eps = torch.randn([batch_size, self.latent_size])
		z = eps * std + means

		recon_x = self.decoder(z)

		return recon_x, means, logvars, z


class Encoder(nn.Module):

	def __init__(self, layer_sizes, latent_size):

		super().__init__()

		self.MLP = nn.Sequential()

		for i, (in_size, out_size) in enumerate( zip(layer_sizes[:-1], layer_sizes[1:]) ):
			self.MLP.add_module(name="L%i"%(i), module=nn.Linear(in_size, out_size))
			self.MLP.add_module(name="A%i"%(i), module=nn.ReLU())


		self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
		self.linear_logvars = nn.Linear(layer_sizes[-1], latent_size)

	def forward(self, x):

		x = self.MLP(x)

		means = self.linear_means(x)
		logvars = self.linear_logvars(x)

		return means, logvars


class Decoder(nn.Module):

	def __init__(self, layer_sizes, latent_size):

		super().__init__()

		self.MLP = nn.Sequential()

		input_size = latent_size

		for i, (in_size, out_size) in enumerate( zip([input_size]+layer_sizes[:-1], layer_sizes)):
			self.MLP.add_module(name="L%i"%(i), module=nn.Linear(in_size, out_size))
			if i+1 < len(layer_sizes):
				self.MLP.add_module(name="A%i"%(i), module=nn.ReLU())
			else:
				self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

	def forward(self, z):

		x = self.MLP(z)

		return x

# def sample_gumbel(shape, eps=1e-20):
# 	U = torch.rand(shape)
# 	if torch.cuda.is_available():
# 		U = U.cuda()
# 	return -torch.log(-torch.log(U + eps) + eps)


# def gumbel_softmax_sample(logits, temperature):
# 	y = logits + sample_gumbel(logits.size())
# 	return F.softmax(y / temperature, dim=-1)


# def gumbel_softmax(logits, temperature, latent_dim, categorical_dim, hard=False):
# 	"""
# 	ST-gumple-softmax
# 	input: [*, n_class]
# 	return: flatten --> [*, n_class] an one-hot vector
# 	"""
# 	y = gumbel_softmax_sample(logits, temperature)
	
# 	if not hard:
# 		return y.view(-1, latent_dim * categorical_dim)

# 	shape = y.size()
# 	_, ind = y.max(dim=-1)
# 	y_hard = torch.zeros_like(y).view(-1, shape[-1])
# 	y_hard.scatter_(1, ind.view(-1, 1), 1)
# 	y_hard = y_hard.view(*shape)
# 	# Set gradients w.r.t. y_hard gradients w.r.t. y
# 	y_hard = (y_hard - y).detach() + y
# 	return y_hard.view(-1, latent_dim * categorical_dim)

# class Classifier(nn.Module):
	
# 	def __init__(self, latent_size, n_classes):
# 		super().__init__()

# 		self.latent_size = latent_size
# 		self.n_classes = n_classes

# 		self.fc1 = nn.Linear(latent_size, 512)
# 		self.fc2 = nn.Linear(512, 2000)
# 		self.fc3 = nn.Linear(2000, n_classes)

# 		# self.fc4 = nn.Linear(n_classes, 2000)
# 		# self.fc5 = nn.Linear(2000, 512)
# 		# self.fc6 = nn.Linear(512, latent_size*n_classes)

# 		self.relu = nn.ReLU()

# 	def encode(self, z):
# 		h1 = self.fc1(z)
# 		h2 = self.fc2(h1)
# 		h3 = self.relu(self.fc3(h2))
# 		return h3

# 	# def decode(self, c):
# 	# 	h4 = self.fc4(c)
# 	# 	h5 = self.fc5(h4)
# 	# 	h_mean = self.relu(self.fc6(h5))
# 	# 	h_logvar = self.relu(self.fc6(h5))

# 	# 	return h_mean, h_logvar

# 	def forward(self, z, temp, hard):
# 		q = self.encode(z)
# 		q_y = q.view(q.size(0), 1, self.n_classes)
# 		c = gumbel_softmax(q_y, temp, 1, self.n_classes, hard)

# 		return F.softmax(q_y, dim=-1).reshape(*q.size()), torch.argmax(c,dim=1)

		
