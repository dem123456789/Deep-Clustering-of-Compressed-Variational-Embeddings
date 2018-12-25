import torch
import torch.nn as nn
import numpy as np


def PSNR(output,target,max=1.0):
	MAX = torch.tensor(max).to(target.device)
	criterion = nn.MSELoss().to(target.device)
	MSE = criterion(output,target)
	psnr = 20*torch.log10(MAX)-10*torch.log10(MSE)
	return psnr

def update_qzc(mu, logvar, z, qy):

	batch_size = z.size()[0]
	n_classes = qy.size()[1]

	for i in range(n_classes):
		p_k = torch.sum(qy[:,i], dim=0) # D1
		qy_t = qy[:,i].view(-1,batch_size) #D 1x128
		mu[:,:,i] = (torch.mm(qy_t,z) / p_k)*0.1 + mu[:,:,i]*0.9 #D 1x20
		logvar[:,:,i] = (torch.mm(qy_t,(z-mu[:,:,i]).pow(2)) /p_k)*0.1+logvar[:,:,i]*0.9 #1x20
	
	return mu, logvar

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

def adjust_learning_rate(init_lr, optimizer, epoch):
	lr = max(init_lr * (0.9 ** (epoch//10)), 0.0002)
	for param_group in optimizer.param_groups:
		param_group["lr"] = lr
	return lr