import torch
import torch.nn as nn

def PSNR(output,target,max=1.0):
    MAX = torch.tensor(max).to(target.device)
    criterion = nn.MSELoss().to(target.device)
    MSE = criterion(output,target)
    psnr = 20*torch.log10(MAX)-10*torch.log10(MSE)
    return psnr
