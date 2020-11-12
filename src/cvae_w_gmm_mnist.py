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
from ctVAE_NN_gmm import VAE_gumbel
#from VAE_gumbel import VAE_gumbel

from sklearn import preprocessing, metrics
from sklearn.cluster import KMeans
from sklearn import preprocessing, model_selection, mixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description='VAE MNIST Example')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
					help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
					help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
					help='how many batches to wait before logging training status')
parser.add_argument('--hard', action='store_true', default=False,
					help='hard Gumbel softmax')
parser.add_argument('--temp', type=float, default=1.0, metavar='S',
					help='tau(temperature) (default: 1.0)')
parser.add_argument("--encoder_layer_sizes", type=list, default=[784, 256])
parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 784])
parser.add_argument("--latent_size", type=int, default=10)
parser.add_argument("--cat_dim", type=int, default=2)
parser.add_argument("--anneal", type=None, default=True)


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())

X = torch.cat((train_dataset.train_data.view(-1,784),test_dataset.test_data.view(-1,784)))
X = X.numpy()

labels = torch.cat((train_dataset.train_labels,test_dataset.test_labels))
labels = labels.numpy()

dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
data_loader = torch.utils.data.DataLoader(dataset,
	batch_size=args.batch_size, shuffle=True, **kwargs)

latent_dim = args.latent_size
cat_dim = args.cat_dim  # one-of-K vector

temp_min = 0.5
ANNEAL_RATE = 0.00003

model = VAE_gumbel(temp=args.temp, input_dim=784, latent_dim=latent_dim, n_centroids=10, binary=True,
		cat_dim=cat_dim, encodeLayer=[500,500,2000], decodeLayer=[2000,500,500]).to(device)
#change binary=True, why so?

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

def adjust_learning_rate(init_lr, optimizer, epoch):
	lr = max(init_lr * (0.9 ** (epoch//10)), 0.0002)
	for param_group in optimizer.param_groups:
		param_group["lr"] = lr
	return lr
	

def train(epoch):
	model.train()
	
	if args.anneal:
	   epoch_lr = adjust_learning_rate(1e-3, optimizer, epoch)
	
	total_loss = 0
	total_psnr = 0
	
	# total_mu = np.empty((0,args.latent_size))
	# total_logvar = np.empty((0,args.latent_size))
	#total_z = np.empty((0,args.latent_size))
	total_labels = []
	pred_labels = []

	temp = args.temp

	for batch_idx, (data, label) in enumerate(data_loader):
		data = data.to(device)
		optimizer.zero_grad()
		data = data.view(-1, 784)
		recon_batch, qy, z = model(data, temp, args.hard)
		print(z.size())

		loss = model.loss_function(recon_batch, data, z, qy)
		#psnr
		psnr = PSNR(recon_batch,data,1.0)

		loss.backward()
		total_loss += loss.item()* len(data)
		optimizer.step()

		if batch_idx % 100 == 1:
			temp = np.maximum(temp * np.exp(-ANNEAL_RATE * batch_idx), temp_min)

		if batch_idx % args.log_interval == 0:
			print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tPSNR:{:.6f}'.format(
				epoch, batch_idx * len(data), len(data_loader.dataset),
				100. * batch_idx / len(data_loader),
				loss.item(),psnr.item()))
		
		gamma = model.get_gamma(z).data.cpu().numpy()

		#total_z = np.row_stack((total_z,z.detach().numpy()))
		total_labels = np.concatenate((total_labels, label.detach().numpy()))
		pred_labels.append(np.argmax(gamma,axis=1))
	
	pred_labels = np.concatenate(pred_labels)
	print('====> Epoch: {} lr: {} Average loss: {:.4f}'.format(
		  epoch, epoch_lr, total_loss / len(data_loader.dataset)))
	return total_labels, pred_labels


def z_clustering_visual(z,total_labels,pred_labels):
	total_labels = total_labels.astype(np.int)

	n_sne = 7000
	time_start = time.time()
	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
	tsne_results = tsne.fit_transform(z)
	print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

	plt.scatter(tsne_results[:,0],tsne_results[:,1],c=total_labels)
	plt.xlabel('latent variable z_1')
	plt.xlabel('latent variable z_2')
	plt.savefig('./z_plot'+'.png')
	plt.clf()

	plt.scatter(tsne_results[:,0],tsne_results[:,1],c=pred_labels)
	plt.xlabel('latent variable z_1')
	plt.xlabel('latent variable z_2')
	plt.savefig('./z_gmm_plot'+'.png')
	plt.clf()


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
	model.initialize_gmm(data_loader, args.temp, args.hard)
	for epoch in range(1,args.epochs + 1):
		total_labels, pred_labels = train(epoch)
		acc = cluster_acc(pred_labels,total_labels)
		print('Epoch:{} clustering acc: [{}]'.format(epoch,acc))
		
		# with torch.no_grad():
		# 	sample = torch.randn(64, 20).to(device)
		# 	sample = model.decode(sample).cpu()
		# 	save_image(sample.view(64, 1, 28, 28),
		# 			   './sample_' + str(epoch) + '.png')
