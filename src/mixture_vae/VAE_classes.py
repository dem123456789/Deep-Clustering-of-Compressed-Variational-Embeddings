import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import *

class VAE(nn.Module):

	def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,n_classes):

		super().__init__()

		assert type(encoder_layer_sizes) == list
		assert type(latent_size) == int
		assert type(decoder_layer_sizes) == list

		self.latent_size = latent_size
		self.n_classes = n_classes

		self.encoder = Encoder(encoder_layer_sizes, latent_size)
		self.classifier = Classifier(latent_size,n_classes)
		self.decoder = Decoder(decoder_layer_sizes, latent_size)

	def forward(self, x, temp, hard):

		batch_size = x.size(0)
		means, logvars = self.encoder(x)

		# z=([])

		# for i, (means_split,log_var_split) in enumerate(zip(torch.chunk(means, 10, dim=1),torch.chunk(log_var,10,dim=1))):
		# 	std = torch.exp(0.5 * log_var_split)
		# 	eps = torch.randn([batch_size, self.latent_size])
		# 	z = torch.cat((z,eps * std + means_split),0)

		std = torch.exp(0.5 * logvars)
		eps = torch.randn([batch_size, self.latent_size])
		z = eps * std + means

		qy, c = self.classifier(z,temp,hard)

		# #sample l z

		# for i in range(20):
		# 	eps = torch.randn([batch_size, self.latent_size])
		# 	z = eps * std + means
		# 	qy_sp, c_sp = self.classifier(z,temp,hard)
		# 	torch.cat(qy,qy_sp,dim=0)
		# 	torch.cat(c,c,dim=0)

		# z_c = torch.zeros(batch_size, self.latent_size)
		# means_c = torch.zeros(batch_size, self.latent_size)
		# logvars_c = torch.zeros(batch_size, self.latent_size)

		# for i in range(batch_size):
		# 	z_c[i,] = torch.chunk(z[i,], 10)[c[i]]
		# 	means_c[i,] = torch.chunk(means[i,], 10)[c[i]]
		# 	logvars_c[i,] = torch.chunk(logvars[i,], 10)[c[i]]

		# # for i, (means_split,log_var_split) in enumerate(zip(torch.chunk(recon_mean, 10, dim=1),torch.chunk(recon_logvar,10,dim=1))):
		# # 	std = torch.exp(0.5 * log_var_split)
		# # 	eps = torch.randn([batch_size, self.latent_size])
		# # 	recon_z = torch.cat((z,eps * std + means_split),0)
		# # 	recon_z = recon_z.append(eps * std + means_split)
		
		recon_x = self.decoder(z)

		return recon_x, means, logvars, z, qy, c

	def inference(self, n=1):

		batch_size = n
		z = torch.randn([batch_size, self.latent_size])

		recon_x = self.decoder(z)

		return recon_x

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

def sample_gumbel(shape, eps=1e-20):
	U = torch.rand(shape)
	if torch.cuda.is_available():
		U = U.cuda()
	return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
	y = logits + sample_gumbel(logits.size())
	return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, latent_dim, categorical_dim, hard=False):
	"""
	ST-gumple-softmax
	input: [*, n_class]
	return: flatten --> [*, n_class] an one-hot vector
	"""
	y = gumbel_softmax_sample(logits, temperature)
	
	if not hard:
		return y.view(-1, latent_dim * categorical_dim)

	shape = y.size()
	_, ind = y.max(dim=-1)
	y_hard = torch.zeros_like(y).view(-1, shape[-1])
	y_hard.scatter_(1, ind.view(-1, 1), 1)
	y_hard = y_hard.view(*shape)
	# Set gradients w.r.t. y_hard gradients w.r.t. y
	y_hard = (y_hard - y).detach() + y
	return y_hard.view(-1, latent_dim * categorical_dim)

class Classifier(nn.Module):
	
	def __init__(self, latent_size, n_classes):
		super().__init__()

		self.latent_size = latent_size
		self.n_classes = n_classes

		self.fc1 = nn.Linear(latent_size, 512)
		self.fc2 = nn.Linear(512, 2000)
		self.fc3 = nn.Linear(2000, n_classes)

		# self.fc4 = nn.Linear(n_classes, 2000)
		# self.fc5 = nn.Linear(2000, 512)
		# self.fc6 = nn.Linear(512, latent_size*n_classes)

		self.relu = nn.ReLU()

	def encode(self, z):
		h1 = self.fc1(z)
		h2 = self.fc2(h1)
		h3 = self.relu(self.fc3(h2))
		return h3

	# def decode(self, c):
	# 	h4 = self.fc4(c)
	# 	h5 = self.fc5(h4)
	# 	h_mean = self.relu(self.fc6(h5))
	# 	h_logvar = self.relu(self.fc6(h5))

	# 	return h_mean, h_logvar

	def forward(self, z, temp, hard):
		q = self.encode(z)
		q_y = q.view(q.size(0), 1, self.n_classes)
		c = gumbel_softmax(q_y, temp, 1, self.n_classes, hard)

		return F.softmax(q_y, dim=-1).reshape(*q.size()), torch.argmax(c,dim=1)

		
