import torch
import config
config.init()
import time
import torch.backends.cudnn as cudnn
import models
import os
import datetime
import argparse
import itertools
from torch import nn
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from data import *
from utils import *
from metrics import *

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Config')
for k in config.PARAM:
	exec('parser.add_argument(\'--{0}\',default=config.PARAM[\'{0}\'], help=\'\')'.format(k))
args = vars(parser.parse_args())
for k in config.PARAM:
	if(config.PARAM[k]!=args[k]):
		exec('config.PARAM[\'{0}\'] = {1}'.format(k,args[k]))
for k in config.PARAM:
	exec('{0} = config.PARAM[\'{0}\']'.format(k))

seeds = list(range(init_seed,init_seed+num_Experiments))
model_names = ['vade_bmm']
code_sizes = [10,12,16]
metrics_names= ['bpp','psnr','cluster_acc']

default_linestyles = {'vade':'-','vade_bmm':'-','cvae':'-'}
default_labels = {'vade':'VAE+GMM','vade_bmm':'Concrete VAE+bmm','cvae':'ConvVAE+GMM'}

def main():
	results = mergeResult()
	show(results)
	return

def mergeResult():
	results = {m: {k: torch.zeros(len(code_sizes), num_Experiments) for k in metrics_names} for m in model_names}
	for i in range(len(model_names)):       
		for j in range(len(code_sizes)):
			for l in range(num_Experiments):
				resume_model_TAG = '{}_{}_{}_{}'.format(code_sizes[j],seeds[l],model_data_name,model_names[i]) if(resume_TAG=='') else '{}_{}_{}_{}'.format(code_sizes[j],seeds[l],model_data_name,model_names[i],resume_TAG)
				model_TAG = resume_model_TAG if(special_TAG=='') else '{}_{}'.format(resume_model_TAG,special_TAG)
				result = load('./output/result/{}.pkl'.format(model_TAG))
				results[model_names[i]]['psnr'][j,l] = result.panel['psnr'].avg
				results[model_names[i]]['cluster_acc'][j,l] = result.panel['cluster_acc'].val
	return results

def show(results):
	colors = ['red', 'black', 'blue', 'brown', 'green', 'cyan']
	for i in range(len(metrics_names)):
		if(metrics_names[i] =='bpp'):
			continue
		plt.figure(i)
		for j in range(len(model_names)):
			x = results[model_names[j]]['bpp'].mean(dim=1).numpy()
			y = results[model_names[j]][metrics_names[i]].mean(dim=0).numpy()
			linestyle = '-' if model_names[j] not in default_linestyles else default_linestyles[model_names[j]]
			label = model_names[j] if model_names[j] not in default_labels else default_labels[model_names[j]]
			plt.plot(x,y,color=colors[j],linestyle=linestyle,label=label)
		plt.xlabel('bpp')
		plt.ylabel(metrics_names[i])
		plt.grid()
		plt.legend()
		plt.savefig('./output/plots/{}.pkl'.format(metrics_names[i]))
	return

if __name__ == "__main__":
	main() 