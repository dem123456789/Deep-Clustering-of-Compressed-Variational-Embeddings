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
parser.add_argument("--encoder_layer_sizes", type=list, default=[784, 256])
parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 784])
parser.add_argument("--latent_size", type=int, default=20)
parser.add_argument('--hard', action='store_true', default=False,
					help='hard Gumbel softmax')
parser.add_argument('--temp', type=float, default=1.0, metavar='S',
					help='tau(temperature) (default: 1.0)')
parser.add_argument("--n_classes", type=int, default=10)


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

temp_min = 0.5 # would this be a problem?
ANNEAL_RATE = 0.00003

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

#optimizer = optim.SGD(model.parameters(),lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, mu_c, logvar_c, qy):

	batch_size = x.size()[0]
	latent_size = mu.size()[1]
	n_classes = qy.size()[1]

	BCE = F.binary_cross_entropy(recon_x, x) #BCE = (recon_x-x).abs().mean() # L-1 loss
	 
	# see Appendix B from VAE paper: 
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# https://arxiv.org/abs/1312.6114
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

	#MLP for z, which is gaussian distribution
	KLD = torch.zeros(batch_size)
	
	for i in range(n_classes):

		KLD_c = -0.5 * torch.sum(1 + (logvar-logvar_c[:,:,i])- (mu-mu_c[:,:,i]).pow(2) / logvar_c[:,:,i].exp() - logvar.exp()/logvar_c[:,:,i].exp(),dim=1)
		KLD = KLD + qy[:,i] * KLD_c

	#KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
 
	# Normalise by same number of elements as in reconstruction
	KLD = torch.sum(KLD)
	KLD /= batch_size *784
	
	#KLD for q(c|z) and p(c)
	log_ratio = torch.log(qy*n_classes+1e-20)
	KLD_2 = torch.sum(qy * log_ratio)
     
	# Normalise by same number of elements as in reconstruction
	KLD_2 /= batch_size *784

	return BCE + KLD + KLD_2 # get the mean here

def update_qzc(mu, logvar, z, qy):

	batch_size = z.size()[0]
	n_classes = qy.size()[1]

	for i in range(n_classes):
		p_k = torch.sum(qy[:,i], dim=0) # D1
		qy_t = qy[:,i].view(-1,batch_size) #D 1x128
		mu[:,:,i] = (torch.mm(qy_t,z) / p_k)*0.1 + mu[:,:,i]*0.9 #D 1x20
		logvar[:,:,i] = (torch.mm(qy_t,(z-mu[:,:,i]).pow(2)) /p_k)*0.1+logvar[:,:,i]*0.9 #1x20
	
	return mu, logvar

def train(model):
	optimizer = optim.Adam(model.parameters(), lr=1e-3)
	model.train()

	mu_c = torch.zeros(1,args.latent_size,args.n_classes) 
	logvar_c = torch.ones(1,args.latent_size,args.n_classes)

	# ave_KLD = []
	# ave_KLD_2 = []

	for epoch in range(1,args.epochs + 1):
		total_loss = 0
		# total_KLD = np.zeros(1)
		# total_KLD_2 = np.zeros(1)
		total_psnr = 0
	
		total_c = []
		total_labels = []
   
		temp = args.temp

		for batch_idx, (data, label) in enumerate(data_loader):
			data = data.to(device)
			optimizer.zero_grad()
			data = data.view(-1, 784)

			recon_x, means, logvars, z, qy, c = model(data,temp,args.hard)
			loss = loss_function(recon_x, data, means, logvars, mu_c, logvar_c, qy)
			
			mu_c, logvar_c = update_qzc(mu_c,logvar_c,z,qy)
			#print(qy[1:10])
			mu_c = mu_c.detach()
			logvar_c = logvar_c.detach()

			c = c.detach().numpy()
			# KLD = KLD.detach().numpy()
			# KLD_2 = KLD_2.detach().numpy()
			
			#psnr
			psnr = PSNR(recon_x,data,1.0)
			loss.backward(retain_graph=True)

			total_loss += loss.item() * len(data)
			# total_KLD += KLD * len(data)
			# total_KLD_2 += KLD_2 * len(data)
			total_psnr += psnr.item()
			optimizer.step()

			if batch_idx % args.log_interval == 0:
				print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tPSNR:{:.6f}'.format(
					epoch, batch_idx * len(data), len(data_loader.dataset),
					100. * batch_idx / len(data_loader),
					loss.item(), psnr.item()))

		total_c = np.concatenate((total_c, c))
		total_labels = np.concatenate((total_labels, label.detach().numpy()))
		# ave_KLD = np.append(ave_KLD,total_KLD / len(data_loader.dataset))
		# ave_KLD_2 = np.concatenate((ave_KLD_2,total_KLD_2 / len(data_loader.dataset)))

		print('====> Epoch: {} Average loss: {:.4f} clustering acc:{}'.format(
			epoch, 
			total_loss / len(data_loader.dataset),
			cluster_acc(total_c,total_labels)))
    
	with torch.no_grad():
		sample = model.inference(n=64).cpu()
		save_image(sample.view(64, 1, 28, 28),
			'./sample_' + 'vae'+ str(epoch) + '.png')
	
	return total_c, total_labels


def cluster_acc(labels_pred,labels):
	labels = labels.astype(np.int)
	labels_pred = labels_pred.astype(np.int)
	
	from sklearn.utils.linear_assignment_ import linear_assignment
	assert labels.size == labels_pred.size
	D = max(labels_pred.max(), labels.max())+1
	w = np.zeros((D,D), dtype=np.int64)

	for i in range(labels_pred.size):
		w[labels_pred[i], labels[i]] += 1
	ind = linear_assignment(w.max() - w)
  
	return sum([w[i,j] for i,j in ind])*1.0/labels_pred.size

if __name__ == "__main__":
	total_c, total_labels = train(vae)
	print(total_c[1:100],total_labels[1:100])

	# plt.plot(range(args.epochs),ave_KLD, labels="second loss term")
	# plt.plot(range(args.epochs),ave_KLD_2, labels="third loss term")
	# plt.legend()
	# plt.savefig('./loss_terms'+'.png')
	# plt.clf()



