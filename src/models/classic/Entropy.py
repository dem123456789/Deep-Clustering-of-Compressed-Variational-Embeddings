import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config
import torchvision.transforms as transforms

config.init()
device = config.PARAM['device']
code_channel = 32

class Entropy(object):
    def __init__(self):
        super(Entropy, self).__init__()
        
    def encode(self,code,protocol):
        code = (torch.stack(code,0)+ 1)//2
        code = code.cpu().numpy().astype(np.int8)
        code = np.packbits(code)
        return code

    def decode(self,code,protocol):
        num_iter = protocol['num_iter']
        depth = protocol['depth']
        jump_rate = protocol['jump_rate']
        input_size = protocol['input_size']

        code = np.unpackbits(code)
        code = np.reshape(code,(num_iter,input_size[0],input_size[1],code_channel,input_size[3]//(jump_rate**depth),input_size[4]//(jump_rate**depth))).astype(np.float32) * 2 - 1
        code = torch.from_numpy(code).to(device)
        code = list(torch.split(code,1,0))
        for i in range(num_iter):
            code[i] = torch.squeeze(code[i],0)
        return code