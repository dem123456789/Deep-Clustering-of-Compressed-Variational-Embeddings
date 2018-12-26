import torch
import numpy as np
from utils import *
from data import *

def collate(input):
    for k in input:
        input[k] = torch.stack(input[k],0)
    return input

if __name__ == '__main__':
    batch_size = 2
    train_dataset, test_dataset = fetch_dataset('MOSI')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=0, collate_fn=input_collate)
    print(len(train_dataset))    
    for i, input in enumerate(train_loader):
        input = collate(input)
        input = dict_to_device(input,device)
        print(input)
        exit()   
