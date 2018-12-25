from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from VAE_basic import VAE
from VAE_vade import VaDE
from VAE_gumbel import VAE_gumbel
from bmm_implement import BMM

from vae_gmm_mnist import *
from vae_w_gmm_mnist import *
from ctvae_bmm_mnist import *
from utils import *

from sklearn.cluster import KMeans
from sklearn import mixture

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
					help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
					help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
					help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
					help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
					help='how many batches to wait before logging training status')

parser.add_argument("--latent_size", type=int, default=20)

parser.add_argument("--RF_n_iter", type=int, default=30)
parser.add_argument("--SNE_n_iter", type=int, default=300)

parser.add_argument('--temp', type=float, default=1.0, metavar='S',
					help='tau(temperature) (default: 1.0)')
parser.add_argument('--hard', action='store_true', default=False,
					help='hard Gumbel softmax')
parser.add_argument("--categorical_dim", type=int, default=2)


parser.add_argument("--anneal", type=None, default=True)

if __name__ == '__main__':

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

	gmm_x = cluster_sample(X,labels)
	acc_gmm_x = cluster_acc(gmm_x,labels)

	dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
	data_loader = torch.utils.data.DataLoader(dataset,
		batch_size=args.batch_size, shuffle=True, **kwargs)

	cvae = VAE_gumbel(temp=args.temp,
		latent_dim=args.latent_size,
		categorical_dim=args.categorical_dim).to(device)

	acc_bmm = train_model_cvae(cvae,
		data_loader,
		args.epochs,
		args.latent_size,
		args.categorical_dim,
		device,
		args.SNE_n_iter,
		args.anneal,
		args.log_interval,
		args.temp,
		args.hard)

	vade = VaDE(input_dim=784, 
		z_dim=args.latent_size, 
		n_centroids=10, 
		binary=True,
		encodeLayer=[512,256], 
		decodeLayer=[256,512]).to(device)

	acc_vade = train_model_vade(vade, 
		data_loader, 
		args.epochs, 
		args.latent_size,
		device,
		args.SNE_n_iter,
		args.anneal,
		args.log_interval)

	vae = VAE(
		encoder_layer_sizes=[784,512,256],
		latent_size=args.latent_size,
		decoder_layer_sizes=[256,512,784]).to(device)

	acc_gmm = train_model(vae, 
		data_loader, 
		args.epochs, 
		args.latent_size,
		device,
		args.SNE_n_iter,
		args.log_interval,
		args.anneal)

	# plt.plot(range(args.RF_n_iter),km_score,label="kmeans")
	# plt.plot(range(args.RF_n_iter),gmm_score,label="gmm")
	# plt.plot(range(args.RF_n_iter),vade_score,label="vade")

	# plt.xlabel('iteration')
	# plt.ylabel('oob error')
	# plt.legend()
	# plt.savefig('./rf_iteration'+'.png')
	# plt.clf()

	plt.plot(ange(args.epochs),acc_vade,label="vae_with_gmm")
	plt.plot(range(args.epochs),acc_gmm,label="vae+gmm")
	plt.plot(range(args.epochs),acc_bmm,label="cvae+bmm")
	plt.hlines(acc_gmm_x, xmin=0, xmax=(args.epochs-1), linestyles='solid', label='gmm')

	# plt.hlines(acc_vade_rf, xmin=0, xmax=args.epochs, linestyles='solid', label='vae_with_gmm+rf')
	# plt.hlines(acc_km_rf, xmin=0, xmax=args.epochs, linestyles='solid', label='vae+kmeans+rf')
	# plt.hlines(acc_gmm_rf, xmin=0, xmax=args.epochs, linestyles='solid', label='vae+gmm+rf')

	plt.legend()
	plt.savefig('./vae_with_gmm_acc'+'.png')
	plt.clf()  

