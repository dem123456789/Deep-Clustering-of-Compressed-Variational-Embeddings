import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config
import torchvision.transforms as transforms

config.init()
device = config.PARAM['device']

class Entropy(object):
    def __init__(self):
        super(Entropy, self).__init__()
        
    def encode(self,code,protocol):
        code = (code + 1)//2
        code = code.cpu().numpy().astype(np.int8)
        code = np.packbits(code)
        return code

    def decode(self,code,protocol):
        code = np.unpackbits(code)
        code = torch.from_numpy(code.astype(np.float32)).to(device)
        code = (code * 2) - 1
        return code