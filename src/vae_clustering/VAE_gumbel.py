import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

def sample_gumbel(shape, eps=1e-20):
	U = torch.rand(shape)
	if torch.cuda.is_available():
		U = U.cuda()
	return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
	y = logits + sample_gumbel(logits.size())
	return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard, latent_dim,categorical_dim):
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


class VAE_gumbel(nn.Module):
	def __init__(self, temp, latent_dim, categorical_dim):
		super(VAE_gumbel, self).__init__()

		self.fc1 = nn.Linear(784, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, latent_dim * categorical_dim)

		self.fc4 = nn.Linear(latent_dim * categorical_dim, 256)
		self.fc5 = nn.Linear(256, 512)
		self.fc6 = nn.Linear(512, 784)

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

	def encode(self, x):
		h1 = self.relu(self.fc1(x))
		h2 = self.relu(self.fc2(h1))
		return self.relu(self.fc3(h2))

	def decode(self, z):
		h4 = self.relu(self.fc4(z))
		h5 = self.relu(self.fc5(h4))
		return self.sigmoid(self.fc6(h5))

	def forward(self, x, temp, hard, latent_dim, categorical_dim):
		q = self.encode(x.view(-1, 784))
		q_y = q.view(q.size(0), latent_dim, categorical_dim)
		z = gumbel_softmax(q_y, temp, hard, latent_dim, categorical_dim)
		return self.decode(z), F.softmax(q_y, dim=-1).reshape(*q.size()), z
