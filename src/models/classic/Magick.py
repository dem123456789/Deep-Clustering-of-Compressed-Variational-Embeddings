import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config
import os
import io
import time
from PIL import Image
from utils import *

class Magick(object):
    def __init__(self):
        super(Magick, self).__init__()
        self.supported_format = ['jpg','jp2','bpg','webp','png']
        
    def encode(self,input,protocol):
        format = protocol['format']
        if format not in self.supported_format:
            print('not supported format')
            exit()
        filename = protocol['filename']
        save_img(input,'./output/tmp/{filename}.png'.format(filename=filename))
        if(format in ['jpg','jp2','webp','png']):
            head = 'convert ./output/tmp/{filename}.png'.format(filename=filename)
            tail = './output/tmp/{filename}_converted.{format}'.format(filename=filename,format=format)
            quality = protocol['quality']
            sampling_factor = protocol['sampling_factor']
            option = ''
            if(quality is not None):
                option += '-quality {quality} '.format(quality=quality)
            if(sampling_factor is not None):
                option += '-sampling-factor {sampling_factor} '.format(sampling_factor=sampling_factor)
            command = '{head} {option}{tail}'.format(head=head, option=option, tail = tail)
        elif(format == 'bpg'):
            head = 'bpgenc ./output/tmp/{filename}.png'.format(filename=filename)
            tail = '-o ./output/tmp/{filename}_converted.bpg'.format(filename=filename)
            quality = protocol['quality']
            sampling_factor = protocol['sampling_factor']
            option = ''
            if(quality is not None):
                quality = 50-quality//2
                option += '-q {quality} '.format(quality=quality)
            if(sampling_factor is not None):
                sampling_factor = ''.join(sampling_factor.split(':'))
                option += '-f {sampling_factor} '.format(sampling_factor=sampling_factor)
            command = '{head} {option}{tail}'.format(head=head, option=option, tail = tail)
        try:
            os.system(command)
        except:
            time.sleep(0.1)
            os.system(command)
        code = open('./output/tmp/{filename}_converted.{format}'.format(filename=filename,format=format), 'rb').read()
        return code

    def decode(self,protocol):
        format = protocol['format']
        if format not in self.supported_format:
            print('not supported format')
            exit()
        filename = protocol['filename']       
        if(format in ['jpg','jp2','webp','png']):
            command = 'convert ./output/tmp/{filename}_converted.{format} ./output/tmp/{filename}.png'.format(filename=filename,format=format)
        elif(format == 'bpg'):
            command = 'bpgdec -o ./output/tmp/{filename}.png ./output/tmp/{filename}_converted.bpg'.format(filename=filename)
        try:
            os.system(command)
        except:
            time.sleep(0.1)
            os.system(command)
        code = open('./output/tmp/{filename}.png'.format(filename=filename), 'rb').read()
        bytesio = io.BytesIO(code)
        output = Image.open(bytesio)
        output = transforms.ToTensor()(output)
        output = output.unsqueeze(0)
        return output


