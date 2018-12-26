import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class Conv2dLSTMCell(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, hidden_kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.hidden_kernel_size = hidden_kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.conv_ih = nn.Conv2d(self.input_channels,4*self.output_size,self.kernel_size,self.stride,self.padding,self.dilation,self.bias)
        self.conv_hh = nn.Conv2d(self.input_channels,4*self.output_size,self.hidden_kernel_size,self.stride,self.padding,self.dilation,self.bias)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.conv_ih.reset_parameters()
        self.conv_hh.reset_parameters()
        return
        
    def forward(self, input, hidden):
        hx, cx = hidden
        gates = self.conv_ih(input) + self.conv_hh(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        return hy, cy

class Conv2dLSTM(nn.Module):
    def __init__(self, input_size, output_size, num_layers, kernel_size=3, hidden_kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(ConvLSTM, self).__init__()
        self.convlstm = self.make_convlstm(input_size, output_size, num_layers, kernel_size, hidden_kernel_size, stride, padding, dilation, bias)
   
    def make_convlstm(self, input_size, output_size, num_layers, kernel_size, hidden_kernel_size, stride, padding, dilation, bias):
        convlstm = nn.ModuleList([])
        for i in range(num_layers):
            convlstm.append(ConvLSTMCell(input_size,output_size,kernel_size,hidden_kernel_size,stride,padding,dilation,bias))
            input_size = output_size
        return convlstm
        
    def forward(self, input, hidden):
        output = [] 
        for i in range(input.size(1)):
            hidden = self.convlstmcell(input[:,i],hidden)
            output.append(hidden[0])
        output = torch.stack(output,1)
        return output,hidden
