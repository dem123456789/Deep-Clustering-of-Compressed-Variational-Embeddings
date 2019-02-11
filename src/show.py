import numpy as np
import cv2
import config
from PIL import Image
from matplotlib import pyplot as plt
from utils import *

config.init()
for k in config.PARAM:
    exec('{0} = config.PARAM[\'{0}\']'.format(k))
seeds = list(range(init_seed,init_seed+num_Experiments))

model_names = ['testnet_2','Joint','Joint_tod_lstm']
metric_names = config.PARAM['test_metric_names']
#default_linestyles = {'Joint_dist_2_2':'-','Joint_dist_base_2_2':'--','Joint_dist_4_4':'-','Joint_dist_base_4_4':'--','Joint_dist_8_8':'-','Joint_dist_base_8_8':'--'}
#default_labels = {'Joint_dist_2_2':'Distributed (2,2)','Joint_dist_base_2_2':'Independent (2,2)','Joint_dist_4_4':'Distributed (4,4)','Joint_dist_base_4_4':'Independent (4,4)','Joint_dist_8_8':'Distributed (8,8)','Joint_dist_base_8_8':'Independent (8,8)'}
default_linestyles = {'Joint':'-','testnet_2':'-','Joint_tod_lstm':'-'}
default_labels = {'Joint':'ConvLSTM','testnet_2':'ResConvLSTM','Joint_tod_lstm':'Toderici (baseline)'}
def main():
    results = mergeResult()
    show(results)
    return

def mergeResult():
    results = {m:{k:torch.zeros(num_Experiments,num_iter) for k in metric_names}for m in model_names}
    for i in range(len(model_names)):       
         for j in range(num_Experiments):
            resume_model_TAG = '{}_{}_{}'.format(seeds[j],model_data_name,model_names[i]) if(resume_TAG=='') else '{}_{}_{}_{}'.format(s,model_data_name,model_names[i],resume_TAG)
            model_TAG = resume_model_TAG if(special_TAG=='') else '{}_{}'.format(resume_model_TAG,special_TAG)
            result = load('./output/result/{}.pkl'.format(model_TAG))
            for n in range(num_iter):
                for k in metric_names:
                    if(k in result['result'][n].panel):
                        results[model_names[i]][k][j,n] = result['result'][n].panel[k].avg    
    return results
    
def show(results):
    colors = ['red', 'black', 'blue', 'brown', 'green', 'cyan']
    for i in range(len(metric_names)):
        if(metric_names[i] =='bpp'):
            continue
        plt.figure(i)
        for j in range(len(model_names)):
            x = results[model_names[j]]['bpp'].mean(dim=0).numpy()
            y = results[model_names[j]][metric_names[i]].mean(dim=0).numpy()
            linestyle = '-' if model_names[j] not in default_linestyles else default_linestyles[model_names[j]]
            label = model_names[j] if model_names[j] not in default_labels else default_labels[model_names[j]]
            plt.plot(x,y,color=colors[j],linestyle=linestyle,label=label)
        plt.xlabel('bpp')
        plt.ylabel(metric_names[i])
        plt.grid()
        plt.legend()
        plt.show()
    return


if __name__ == "__main__":
   main()