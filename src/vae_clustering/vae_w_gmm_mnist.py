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

def train_model_vade(model, data_loader, 
	epochs, 
	latent_size, 
	device, 
	SNE_n_iter, 
	anneal, log_interval):

	model.initialize_gmm(data_loader)
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
	
	acc_vade = ([])

	for epoch in range (1, epochs+1):
		model.train()

		if anneal:
			epoch_lr = adjust_learning_rate(1e-3, optimizer, epoch)
	
		total_loss = 0
		total_psnr = 0
		
		total_z = np.empty((0,latent_size))
		total_labels = []
		pred_labels = []

		for batch_idx, (data, label) in enumerate(data_loader):
			data = data.to(device)
			optimizer.zero_grad()
			data = data.view(-1, 784)

			z, recon_batch, mu, logvar = model(data)
			loss = model.loss_function(recon_batch, data, z, mu, logvar)
		#psnr
			psnr = PSNR(recon_batch,data,1.0)
		
			loss.backward()
			total_loss += loss.item()*len(data)
			total_psnr += psnr.item()*len(data)
			optimizer.step()
		# if batch_idx % args.log_interval == 0:
		# 	print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tPSNR:{:.6f}'.format(
		# 		epoch, batch_idx * len(data), len(data_loader.dataset),
		# 		100. * batch_idx / len(data_loader),
		# 		loss.item() / len(data),psnr.item()))

			gamma = model.get_gamma(z, mu, logvar).data.cpu().numpy()

			total_z = np.row_stack((total_z,z.detach().numpy()))
			total_labels = np.concatenate((total_labels, label.detach().numpy()))
			pred_labels.append(np.argmax(gamma,axis=1))

		pred_labels = np.concatenate(pred_labels)
		print('====> Epoch: {} lr: {} Average loss: {:.4f} Average psnr: {:.4f}'.format(
			epoch, epoch_lr, 
			total_loss / len(data_loader.dataset),
			total_psnr / len(data_loader.dataset)))

		acc_h = cluster_acc(pred_labels,total_labels)
		acc_vade = np.append(acc_vade,acc_h)

		# if epoch % log_interval == 0:
		# 	visualization(total_z, pred_labels, total_labels, SNE_n_iter, epoch, 2)
	
	with torch.no_grad():
		sample = torch.randn(64, latent_size).to(device)
		sample = model.decode(sample).cpu()
		save_image(sample.view(64, 1, 28, 28),
			'./sample_' + 'vae_gmm' + str(epoch)+'.png') 

	return acc_vade