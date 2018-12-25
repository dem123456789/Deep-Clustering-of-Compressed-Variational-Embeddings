from __future__ import print_function
import argparse
import torch
import sklearn
import torch.utils.data
import os
import time 
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from utils import *
from VAE_classes import VAE

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='VAE MNIST Example')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
					help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
					help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
					help='how many batches to wait before logging training status')
parser.add_argument("--encoder_layer_sizes", type=list, default=[784, 500,500,2000])
parser.add_argument("--decoder_layer_sizes", type=list, default=[2000,500,500, 784])
parser.add_argument("--latent_size", type=int, default=20)
# parser.add_argument('--hard', action='store_true', default=False,
# 					help='hard Gumbel softmax')
# parser.add_argument('--temp', type=float, default=1.0, metavar='S',
# 					help='tau(temperature) (default: 1.0)')
parser.add_argument("--n_classes", type=int, default=10)
parser.add_argument("--anneal", type=None, default=True)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

# temp_min = 0.5 # would this be a problem?
# ANNEAL_RATE = 0.00003

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())

dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

data_loader = torch.utils.data.DataLoader(dataset,
	batch_size=args.batch_size, shuffle=True, **kwargs)

vae = VAE(
		encoder_layer_sizes=args.encoder_layer_sizes,
		latent_size=args.latent_size,
		decoder_layer_sizes=args.decoder_layer_sizes,
		n_classes=args.n_classes
		).to(device)

def train(model):
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

	# mu_c = torch.zeros(1,args.latent_size,args.n_classes) 
	# logvar_c = torch.ones(1,args.latent_size,args.n_classes)

	for epoch in range(1,args.epochs + 1):
		model.train()

		if args.anneal:
			epoch_lr = adjust_learning_rate(1e-3, optimizer, epoch)

		total_loss = 0
		total_psnr = 0
	
		total_c = []
		total_labels = []
   
		#temp = args.temp

		for batch_idx, (data, label) in enumerate(data_loader):
			data = data.to(device)
			optimizer.zero_grad()
			data = data.view(-1, 784)

			recon_x, means, logvars, z = model(data)
			loss = model.loss_function(recon_x, data, z, means, logvars)
			#psnr
			psnr = PSNR(recon_x,data,1.0)
			loss.backward()

			total_loss += loss.item() * len(data)
			total_psnr += psnr.item()
			optimizer.step()

			if batch_idx % args.log_interval == 0:
				print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tPSNR:{:.6f}'.format(
					epoch, batch_idx * len(data), len(data_loader.dataset),
					100. * batch_idx / len(data_loader),
					loss.item(), psnr.item()))

			qc = model.get_gamma(z).detach().numpy()
			total_labels = np.concatenate((total_labels, label.detach().numpy()))
			total_c.append(np.argmax(qc,axis=1))

		total_c = np.concatenate(total_c)
		
		print('====> Epoch: {} Average loss: {:.4f} clustering acc:{}'.format(
			epoch, 
			total_loss / len(data_loader.dataset),
			cluster_acc(total_c,total_labels)))
	
	# with torch.no_grad():
	# 	sample = model.inference(n=64).cpu()
	# 	save_image(sample.view(64, 1, 28, 28),
	# 		'./sample_' + 'vae'+ str(epoch) + '.png')


if __name__ == "__main__":
	vae.initialize_gmm(data_loader)
	train(vae)


	# plt.plot(range(args.epochs),ave_KLD, labels="second loss term")
	# plt.plot(range(args.epochs),ave_KLD_2, labels="third loss term")
	# plt.legend()
	# plt.savefig('./loss_terms'+'.png')
	# plt.clf()



