from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from utils import *

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):

	BCE = F.binary_cross_entropy(recon_x, x) # change to L-1 loss
	#BCE = (recon_x-x).abs().mean() # L-1 loss

	# see Appendix B from VAE paper:
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# https://arxiv.org/abs/1312.6114
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	# Normalise by same number of elements as in reconstruction
	KLD /= x.size()[0] * 784

	return BCE + KLD # get the mean here

def train_model(model, data_loader, epochs, 
	latent_size, 
	device, 
	SNE_n_iter,
	log_interval, 
	anneal):

	optimizer = optim.Adam(model.parameters(), lr=1e-3)

	acc_gmm = ([])

	for epoch in range(1, epochs + 1):
		model.train()

		if anneal:
			epoch_lr = adjust_learning_rate(1e-3, optimizer, epoch)

		total_loss = 0
		total_psnr = 0

		total_z = np.empty((0,latent_size))
		total_labels = []

		for batch_idx, (data, label) in enumerate(data_loader):
			data = data.to(device)
			optimizer.zero_grad()
			data = data.view(-1, 784)
			recon_batch, mu, logvar, z = model(data)
			loss = loss_function(recon_batch, data, mu, logvar)
		#psnr
			psnr = PSNR(recon_batch,data,1.0)
			loss.backward()

			total_loss += loss.item()*len(data)
			total_psnr += psnr.item()*len(data)
			optimizer.step()

		  #   if batch_idx % args.log_interval == 0:
			 #    print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tPSNR:{:.6f}'.format(
				# epoch, batch_idx * len(data), len(data_loader.dataset),
				# 100. * batch_idx / len(data_loader),
				# loss.item()*len(data), psnr.item()))
			total_z = np.row_stack((total_z,z.detach().numpy()))
			total_labels = np.concatenate((total_labels, label.detach().numpy()))
		
		gmm_z_pred = cluster_sample(total_z,total_labels)
		acc_h2 = cluster_acc(gmm_z_pred,total_labels)

		acc_gmm = np.append(acc_gmm,acc_h2)

		print('====> Epoch: {} lr:{} Average loss: {:.4f} Average psnr: {:.4f}'.format(
			epoch, epoch_lr, 
			total_loss/len(data_loader.dataset),
			total_psnr/len(data_loader.dataset)))

		# if epoch % log_interval == 0:
		# 	visualization(total_z, gmm_z_pred, total_labels, SNE_n_iter, epoch, 1)

		# if epoch == epochs:
		# 	km_score, km_rf_pred = rf_Iteration(total_z,km_z_pred,RF_n_iter)
		# 	gmm_score, gmm_rf_pred = rf_Iteration(total_z,gmm_z_pred,RF_n_iter)

	# acc_km_rf = cluster_acc(km_rf_pred, total_labels) 
	# acc_gmm_rf = cluster_acc(gmm_rf_pred, total_labels)

	with torch.no_grad():
		sample = model.inference(n=64).cpu()
		save_image(sample.view(64, 1, 28, 28), 
			'./sample_' + 'vae'+ str(epoch) + '.png')

	return acc_gmm