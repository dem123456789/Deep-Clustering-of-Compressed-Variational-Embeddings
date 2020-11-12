import torch
import torch.nn as nn
import numpy as np
import time 

from numpy.core.umath_tests import inner1d
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from bmm_implement import BMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def PSNR(output,target,max=1.0):
	MAX = torch.tensor(max).to(target.device)
	criterion = nn.MSELoss().to(target.device)
	MSE = criterion(output,target)
	psnr = 20*torch.log10(MAX)-10*torch.log10(MSE)
	return psnr

def adjust_learning_rate(init_lr, optimizer, epoch):
	lr = max(init_lr * (0.9 ** (epoch//10)), 0.0002)
	for param_group in optimizer.param_groups:
		param_group["lr"] = lr
	return lr

def cluster_sample(z,total_labels,z_type=False):
	total_labels = total_labels.astype(np.int)

	#km_pred = KMeans(n_clusters=10).fit_predict(z)

	if z_type:
		bmm = BMM(n_comp=10,n_iter=300).fit(z)
		bmm_means = bmm.qh
		mm_pred = bmm.predict(z)
	else:
		gmm =  GaussianMixture(n_components=10,covariance_type='diag').fit(z)
		mm_pred = gmm.predict(z)
	
	return mm_pred

def rf_Iteration(z, pred_label, RF_n_iter):
	pred_label = pred_label.astype(np.int)
	pred_score = []

	for i in range(RF_n_iter):
		rf_pred = RandomForestClassifier(n_estimators=100, random_state=0, max_features=None, oob_score=True)
		z_train, z_test, label_train, label_test = train_test_split(z, pred_label, test_size=0.67)
		rf_pred.fit(z_train, label_train)

		pred_score.append(rf_pred.oob_score_)
		print('Iteration:{} \t oob:[{}]\t acc:[{}]'.format(
		  i, rf_pred.oob_score_, rf_pred.score(z_test, label_test)))
		
		updated_pred = rf_pred.predict(z)

	return pred_score, updated_pred

def visualization(z, pred_labels, total_labels, SNE_n_iter, epoch, em):

	time_start = time.time()
	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=SNE_n_iter)
	tsne_results = tsne.fit_transform(z)
	print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

	plt.scatter(tsne_results[1:1000,0],tsne_results[1:1000,1],c=total_labels[1:1000])
	plt.xlabel('latent variable z_1')
	plt.xlabel('latent variable z_2')
	plt.savefig('./z_plot'+str(em)+str(epoch)+'.png')
	plt.clf()

	plt.scatter(tsne_results[1:1000,0],tsne_results[1:1000,1],c=pred_labels[1:1000])
	plt.xlabel('latent variable z_1')
	plt.xlabel('latent variable z_2')
	plt.savefig('./z_classifier_plot'+str(em)+str(epoch)+'.png')
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

