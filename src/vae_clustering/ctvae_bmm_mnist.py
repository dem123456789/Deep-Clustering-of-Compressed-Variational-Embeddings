from __future__ import print_function

import torch
import torch.utils.data

from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np

from utils import *

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, qy, categorical_dim):
	BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum') / x.shape[0]

	log_ratio = torch.log(qy * categorical_dim + 1e-20)
	KLD = torch.sum(qy * log_ratio, dim=-1).mean()

	loss = (BCE+KLD) / recon_x.shape[0]

	return loss

def train_model_cvae(model, data_loader, epochs,
	latent_dim,
	categorical_dim,
	device,
	SNE_n_iter,
	anneal,
	log_interval,
	temp,
	hard,
	temp_min=0.5,
	ANNEAL_RATE = 0.00003):

	optimizer = optim.Adam(model.parameters(), lr=1e-3)

	acc_bmm = ([])

	for epoch in range(1, epochs + 1):
		model.train()

		if anneal:
			epoch_lr = adjust_learning_rate(1e-3, optimizer, epoch)

		total_loss = 0
		total_psnr = 0

		total_z = ([])
		total_labels = []

		temp = temp

		for batch_idx, (data, label) in enumerate(data_loader):
			data = data.to(device)

			optimizer.zero_grad()

			recon_batch, qy, z = model(data, temp, hard, latent_dim, categorical_dim)
			loss = loss_function(recon_batch, data, qy, categorical_dim)

			z = z.detach().numpy().reshape((len(data),latent_dim,categorical_dim))
			z = np.argmax(z, axis=2)

			psnr = PSNR(recon_batch,data.view(-1,784),1.0)

			loss.backward()
			total_loss += loss.item() * len(data)
			total_psnr += psnr.item() * len(data)
			optimizer.step()

			if batch_idx % 100 == 1:
				temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)

			total_z.append(z)
			total_labels = np.concatenate((total_labels, label.detach().numpy()))
		
		total_z = np.concatenate(total_z)
		bmm_z_pred = cluster_sample(total_z,total_labels,z_type=True)
		acc_h2 = cluster_acc(bmm_z_pred,total_labels)

		acc_bmm = np.append(acc_bmm,acc_h2)

		print('====> Epoch: {} lr:{} Average loss: {:.4f} Average psnr: {:.4f}'.format(
			epoch, epoch_lr, 
			total_loss/len(data_loader.dataset),
			total_psnr/len(data_loader.dataset)))

		# if epoch % log_interval == 0:
		# 	visualization(total_z, bmm_z_pred, total_labels, SNE_n_iter, epoch, 3)

	M = 64 * latent_dim
	np_y = np.zeros((M, categorical_dim), dtype=np.float32)
	np_y[range(M), np.random.choice(categorical_dim, M)] = 1
	np_y = np.reshape(np_y, [M // latent_dim, latent_dim, categorical_dim])
	sample = torch.from_numpy(np_y).view(M // latent_dim, latent_dim * categorical_dim)
	# if args.cuda:
	#   sample = sample.cuda()
	sample = model.decode(sample).cpu()
	save_image(sample.data.view(M // latent_dim, 1, 28, 28),
				   './sample_' + 'cvae' + str(epoch) + '.png')

	return acc_bmm