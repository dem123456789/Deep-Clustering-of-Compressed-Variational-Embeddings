import numpy as np
import cv2
import config
from PIL import Image
from matplotlib import pyplot as plt
from utils import *

config.init()
model_data_name = config.PARAM['model_data_name']
model_dir = config.PARAM['model_dir']
model_name = config.PARAM['model_name']
resume_TAG = config.PARAM['resume_TAG']
special_TAG = config.PARAM['special_TAG']
seed = 0

def show(TAG):
    colors = ['red', 'black', 'blue', 'brown', 'green']
    default_label = None
    for i in range(len(TAG)):
        result = load('./output/result/{}.pkl'.format(TAG[i]))
        plt.figure(0)
        label = (TAG[i][2:]+'_'+str(result['epoch']) if('epoch' in result) else TAG[i][2:]) if default_label is None else default_label[i]
        plt.plot(result['bpp'],result['psnr'],color=colors[i],label=label)
        print('Label: {}, PSNR: {}, AUC: {}'.format(label,result['psnr'][-1],AUC(result['bpp'],result['psnr'])))
    plt.xlabel('bpp')
    plt.ylabel('psnr(db)')
    plt.grid()
    plt.legend()
    plt.show()
    label = None
    for i in range(len(TAG)):
        result = load('./output/result/{}.pkl'.format(TAG[i]))
        if('acc' in result):
            plt.figure(1)
            label = (TAG[i][2:]+'_'+str(result['epoch']) if('epoch' in result) else TAG[i][2:]) if default_label is None else default_label[i]
            plt.plot(result['bpp'],result['acc'],color=colors[i],label=label)
            print('Label: {}, ACC: {}, AUC: {}'.format(label,result['acc'][-1],AUC(result['bpp'],result['psnr'])))
    plt.xlabel('bpp')
    plt.ylabel('acc')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    head = str(seed)+'_'+model_data_name
    TAG = [head+'_Joint',head+'_Joint_8_iter']
    show(TAG)
