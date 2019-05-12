import config
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict 
from modules.shuffle import PixelUnShuffle,PixelShuffle
from modules.organic import oConv1d,oConv2d,oConv3d
from utils import _ntuple

device = config.PARAM['device']

def Normalization(normalization,output_size):
    if(normalization=='none'):
        return nn.Sequential()
    elif(normalization=='bn'):
        return nn.BatchNorm2d(output_size)
    elif(normalization=='in'):
        return nn.InstanceNorm2d(output_size)
    else:
        raise ValueError('Normalization mode not supported')
    return
    
def Activation(activation):
    if(activation=='none'):
        return nn.Sequential()
    elif(activation=='tanh'):
        return nn.Tanh()
    elif(activation=='relu'):
        return nn.ReLU(inplace=True)
    elif(activation=='prelu'):
        return nn.PReLU()
    elif(activation=='elu'):
        return nn.ELU(inplace=True)
    elif(activation=='selu'):
        return nn.SELU(inplace=True)
    elif(activation=='celu'):
        return nn.CELU(inplace=True)
    elif(activation=='sigmoid'):
        return nn.Sigmoid()
    elif(activation=='softmax'):
        return nn.SoftMax()
    else:
        raise ValueError('Activation mode not supported')
    return

class BasicCell(nn.Module):
    def __init__(self, cell_info):
        super(BasicCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layer'])])
        for i in range(cell_info['num_layer']):
            if(cell_info['mode']=='downsample'):
                cell_in_info = {'cell':'Conv2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size'],
                            'kernel_size':3,'stride':2,'padding':1,'dilation':1,'group':1,'bias':cell_info['bias'],
                            'normalization':cell_info['normalization'],'activation':cell_info['activation']}
            elif(cell_info['mode']=='pass'):
                cell_in_info = {'cell':'Conv2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size'],
                            'kernel_size':3,'stride':1,'padding':1,'dilation':1,'group':1,'bias':cell_info['bias'],
                            'normalization':cell_info['normalization'],'activation':cell_info['activation']}
            elif(cell_info['mode']=='upsample'):
                cell_in_info = {'cell':'ConvTranspose2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size'],
                            'kernel_size':2,'stride':2,'padding':0,'output_padding':0,'dilation':1,'group':1,'bias':cell_info['bias'],
                            'normalization':cell_info['normalization'],'activation':cell_info['activation']}
            elif(cell_info['mode']=='fc'):
                cell_in_info = {'cell':'Conv2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size'],
                            'kernel_size':1,'stride':1,'padding':0,'dilation':1,'group':1,'bias':cell_info['bias'],
                            'normalization':cell_info['normalization'],'activation':cell_info['activation']}
            elif(cell_info['mode']=='fc_down'):
                cell_in_info = {'cell':'Conv2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size'],
                            'kernel_size':1,'stride':2,'padding':0,'dilation':1,'group':1,'bias':cell_info['bias'],
                            'normalization':cell_info['normalization'],'activation':cell_info['activation']}
            else:
                raise ValueError('model mode not supported')
            cell[i]['in'] = Cell(cell_in_info)
            if(i==0):
                cell_info = {**cell_info,'cell':'Conv2d','input_size':cell_info['output_size'],'kernel_size':3,'stride':1,'padding':1}
        return cell
        
    def forward(self, input):
        x = input
        for i in range(self.cell_info['num_layer']):
            x = self.cell[i]['in'](x)
        return x

class ResBasicCell(nn.Module):
    def __init__(self, cell_info):
        super(ResBasicCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layer'])])
        for i in range(cell_info['num_layer']):
            if(cell_info['mode'] == 'downsample' and i == 0):
                stride = 2
            elif(cell_info['mode'] == 'pass' or i > 0):
                stride = 1
            else:
                raise ValueError('model mode not supported')
            if(cell_info['input_size'] != cell_info['output_size'] or stride != 1):
                cell_shortcut_info = {'cell':'Conv2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size'],
                            'kernel_size':1,'stride':stride,'padding':0,'dilation':1,'group':1,'bias':False,
                            'normalization':cell_info['normalization'],'activation':'none'}
            else:
                cell_shortcut_info = {'cell':'none'}

            cell_in_info = {'cell':'Conv2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size'],
                            'kernel_size':3,'stride':stride,'padding':1,'dilation':1,'group':1,'bias':False,
                            'normalization':cell_info['normalization'],'activation':'none'}
            cell_out_info = {'cell':'Conv2d','input_size':cell_info['output_size'],'output_size':cell_info['output_size'],
                            'kernel_size':3,'stride':1,'padding':1,'dilation':1,'group':1,'bias':False,
                            'normalization':cell_info['normalization'],'activation':'none'}
            cell[i]['shortcut'] = Cell(cell_shortcut_info)
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['out'] = Cell(cell_out_info)
            cell[i]['activation'] = Activation(cell_info['activation'])
            if(i==0):
                cell_info['input_size'] = cell_info['output_size']
        return cell
        
    def forward(self, input):
        x = input
        for i in range(len(self.cell)):
            shortcut = self.cell[i]['shortcut'](x)
            x = self.cell[i]['in'](x)
            x = self.cell[i]['out'](x)
            x = self.cell[i]['activation'](x+shortcut)
        return x
        
class LSTMCell(nn.Module):
    def __init__(self, cell_info):
        super(LSTMCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        self.hidden = None
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        _tuple = _ntuple(2)
        cell_info['activation'] = _tuple(cell_info['activation'])
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layer'])])
        for i in range(cell_info['num_layer']):
            cell_in_info = {**cell_info['in'][i],'output_size':4*cell_info['in'][i]['output_size']}
            cell_hidden_info = {**cell_info['hidden'][i],'output_size':4*cell_info['hidden'][i]['output_size']}
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['hidden'] = Cell(cell_hidden_info)
            cell[i]['activation'] = nn.ModuleList([Activation(cell_info['activation'][0]),Activation(cell_info['activation'][1])])
        return cell
        
    def init_hidden(self, hidden_size):
        hidden = [[torch.zeros(hidden_size,device=device)],[torch.zeros(hidden_size,device=device)]]
        return hidden
    
    def free_hidden(self):
        self.hidden = None
        return
        
    def forward(self, input, hidden=None):
        x = input
        x = x.unsqueeze(1) if(input.dim()==4) else x
        hx,cx = [None for _ in range(len(self.cell))],[None for _ in range(len(self.cell))]
        for i in range(len(self.cell)):
            y = [None for _ in range(x.size(1))]
            for j in range(x.size(1)):
                gates = self.cell[i]['in'](x[:,j])
                if(hidden is None):
                    if(self.hidden is None):
                        self.hidden = self.init_hidden((gates.size(0),self.cell_info['hidden'][i]['output_size'],*gates.size()[2:]))
                    else:
                        if(i==len(self.hidden[0])):
                            tmp_hidden = self.init_hidden((gates.size(0),self.cell_info['hidden'][i]['output_size'],*gates.size()[2:]))
                            self.hidden[0].extend(tmp_hidden[0])
                            self.hidden[1].extend(tmp_hidden[1])
                        else:
                            pass
                if(j==0):
                    hx[i],cx[i] = self.hidden[0][i],self.hidden[1][i]
                gates += self.cell[i]['hidden'](hx[i])
                ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
                ingate = torch.sigmoid(ingate)
                forgetgate = torch.sigmoid(forgetgate)
                cellgate = self.cell[i]['activation'][0](cellgate)
                outgate = torch.sigmoid(outgate)
                cx[i] = (forgetgate * cx[i]) + (ingate * cellgate)
                hx[i] = outgate * self.cell[i]['activation'][1](cx[i])                
                y[j] = hx[i]
            x = torch.stack(y,dim=1)
        self.hidden = [hx,cx]
        x = x.squeeze(1) if(input.dim()==4) else x
        return x

class ResLSTMCell(nn.Module):
    def __init__(self, cell_info):
        super(ResLSTMCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        self.hidden = None
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        _tuple = _ntuple(2)
        cell_info['activation'] = _tuple(cell_info['activation'])
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layer'])])
        for i in range(cell_info['num_layer']):
            if(i==0):
                cell_shortcut_info = cell_info['shortcut'][i]
                cell[i]['shortcut'] = Cell(cell_shortcut_info)
            cell_in_info = {**cell_info['in'][i],'output_size':4*cell_info['in'][i]['output_size']}
            cell_hidden_info = {**cell_info['hidden'][i],'output_size':4*cell_info['hidden'][i]['output_size']}
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['hidden'] = Cell(cell_hidden_info)            
            cell[i]['activation'] = nn.ModuleList([Activation(cell_info['activation'][0]),Activation(cell_info['activation'][1])])         
        return cell
        
    def init_hidden(self, hidden_size):
        hidden = [[torch.zeros(hidden_size,device=device)],[torch.zeros(hidden_size,device=device)]]
        return hidden
    
    def free_hidden(self):
        self.hidden = None
        return

    def forward(self, input, hidden=None):
        x = input
        x = x.unsqueeze(1) if(input.dim()==4) else x
        hx,cx = [None for _ in range(len(self.cell))],[None for _ in range(len(self.cell))]
        shortcut = [None for _ in range(x.size(1))]
        for i in range(len(self.cell)):
            y = [None for _ in range(x.size(1))]
            for j in range(x.size(1)):
                if(i==0):
                    shortcut[j] = self.cell[i]['shortcut'](x[:,j])
                gates = self.cell[i]['in'](x[:,j])
                if(hidden is None):
                    if(self.hidden is None):
                        self.hidden = self.init_hidden((gates.size(0),self.cell_info['hidden'][i]['output_size'],*gates.size()[2:]))
                    else:
                        if(i==len(self.hidden[0])):
                            tmp_hidden = self.init_hidden((gates.size(0),self.cell_info['hidden'][i]['output_size'],*gates.size()[2:]))
                            self.hidden[0].extend(tmp_hidden[0])
                            self.hidden[1].extend(tmp_hidden[1])
                        else:
                            pass
                else:
                    self.hidden = hidden
                if(j==0):
                    hx[i],cx[i] = self.hidden[0][i],self.hidden[1][i]
                gates += self.cell[i]['hidden'](hx[i])
                ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
                ingate = torch.sigmoid(ingate)
                forgetgate = torch.sigmoid(forgetgate)
                cellgate = self.cell[i]['activation'][0](cellgate)
                outgate = torch.sigmoid(outgate)
                cx[i] = (forgetgate * cx[i]) + (ingate * cellgate)  
                hx[i] = outgate * self.cell[i]['activation'][1](cx[i]) if(i<len(self.cell)-1) else outgate*(shortcut[j] + self.cell[i]['activation'][1](cx[i]))
                y[j] = hx[i]
            x = torch.stack(y,dim=1)
        self.hidden = [hx,cx]
        x = x.squeeze(1) if(input.dim()==4) else x
        return x

class PixelShuffleCell(nn.Module):
    def __init__(self, cell_info):
        super(PixelShuffleCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        if(cell_info['mode'] == 'down'):        
            cell = PixelUnShuffle(cell_info['scale_factor'])
        elif(cell_info['mode'] == 'up'):        
            cell = PixelShuffle(cell_info['scale_factor'])
        else:
            raise ValueError('model mode not supported')
        return cell
        
    def forward(self, input):
        x = self.cell(input)
        return x

class PoolCell(nn.Module):
    def __init__(self, cell_info):
        super(PoolCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        if(cell_info['mode'] == 'avg'):        
            cell = nn.AvgPool2d(kernel_size=cell_info['kernel_size'],stride=cell_info['stride'],padding=cell_info['padding'],
            ceil_mode=cell_info['ceil_mode'],count_include_pad=cell_info['count_include_pad'])
        elif(cell_info['mode'] == 'max'):        
            cell = nn.MaxPool2d(kernel_size=cell_info['kernel_size'],stride=cell_info['stride'],padding=cell_info['padding'],
            dilation=cell_info['dilation'],return_indices=cell_info['return_indices'],ceil_mode=cell_info['ceil_mode'])
        elif(cell_info['mode'] == 'adapt_avg'):        
            cell = nn.AdaptiveAvgPool2d(cell_info['output_size'])
        elif(cell_info['mode'] == 'adapt_max'):        
            cell = nn.AdaptiveMaxPool2d(cell_info['output_size'])
        else:
            raise ValueError('model mode not supported')
        return cell
        
    def forward(self, input):
        x = self.cell(input)
        return x

class BottleNeckCell(nn.Module):
    def __init__(self, cell_info):
        super(BottleNeckCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layer'])])
        for i in range(cell_info['num_layer']):
            if(cell_info['mode'] == 'downsample' and i == 0):
                stride = 2
            elif(cell_info['mode'] == 'pass' or i > 0):
                stride = 1
            else:
                raise ValueError('model mode not supported')
            if(cell_info['input_size'] != cell_info['output_size'] or stride != 1):
                cell_shortcut_info = {'cell':'Conv2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size'],
                            'kernel_size':1,'stride':stride,'padding':0,'dilation':1,'group':1,'bias':False,
                            'normalization':cell_info['normalization'],'activation':'none'}
            else:
                cell_shortcut_info = {'cell':'none'}
 
            cell_reduce_info = {'cell':'Conv2d','input_size':cell_info['input_size'],'output_size':cell_info['neck'][i]['input_size'],
                            'kernel_size':1,'stride':1,'padding':0,'dilation':1,'group':1,'bias':False,
                            'normalization':cell_info['normalization'],'activation':cell_info['activation']}
            cell_neck_info = cell_info['neck'][i]
            cell_expand_info = {'cell':'Conv2d','input_size':cell_info['neck'][i]['output_size'],'output_size':cell_info['output_size'],
                            'kernel_size':1,'stride':1,'padding':0,'dilation':1,'group':1,'bias':False,
                            'normalization':cell_info['normalization'],'activation':'none'}
            cell[i]['shortcut'] = Cell(cell_shortcut_info)
            cell[i]['reduce'] = Cell(cell_reduce_info)
            cell[i]['neck'] = Cell(cell_neck_info)
            cell[i]['expand'] = Cell(cell_expand_info)
            cell[i]['activation'] = Activation(cell_info['activation'])
            if(i==0):
                cell_info['input_size'] = cell_info['output_size']
        return cell
        
    def forward(self, input):
        x = input
        for i in range(len(self.cell)):
            shortcut = self.cell[i]['shortcut'](x)
            x = self.cell[i]['reduce'](x)
            x = self.cell[i]['neck'](x)
            x = self.cell[i]['expand'](x)
            x = self.cell[i]['activation'](x+shortcut)
        return x

class CartesianCell(nn.Module):
    def __init__(self, cell_info):
        super(CartesianCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleDict({})
        if(cell_info['mode']=='downsample'):
            cell_main_info = {'cell':'oConv2d','input_size':cell_info['input_size'],'output_size':int(cell_info['output_size']//np.prod(cell_info['cardinality'])),
                        'kernel_size':3,'stride':2,'padding':1,'dilation':1,'bias':cell_info['bias']}
        elif(cell_info['mode']=='pass'):
            cell_main_info = {'cell':'oConv2d','input_size':cell_info['input_size'],'output_size':int(cell_info['output_size']//np.prod(cell_info['cardinality'])),
                        'kernel_size':3,'stride':1,'padding':1,'dilation':1,'bias':cell_info['bias']}
        elif(cell_info['mode']=='fc'):
            cell_main_info = {'cell':'oConv2d','input_size':cell_info['input_size'],'output_size':int(cell_info['output_size']//np.prod(cell_info['cardinality'])),
                        'kernel_size':1,'stride':1,'padding':0,'dilation':1,'bias':cell_info['bias']}
        elif(cell_info['mode']=='fc_down'):
            cell_main_info = {'cell':'oConv2d','input_size':cell_info['input_size'],'output_size':int(cell_info['output_size']//np.prod(cell_info['cardinality'])),
                        'kernel_size':1,'stride':2,'padding':0,'dilation':1,'bias':cell_info['bias']}
        cell['main'] = Cell(cell_main_info)
        cell['normalization'] = Normalization(cell_info['normalization'],cell_info['output_size'])
        cell['activation'] = Activation(cell_info['activation'])
        return cell
    
    def run(self, input, coordinates, cardinality, sharing_rate):
        x = input
        if(len(cardinality)!=0):
            sharing_x = x[:,:int(len(coordinates[0])*sharing_rate[0])]
            non_sharing_x = x[:,int(len(coordinates[0])*sharing_rate[0]):].chunk(cardinality[0],dim=1)
            shared_coordinates = coordinates[0][:int(len(coordinates[0])*sharing_rate[0])]
            non_shared_coordinates = coordinates[0][int(len(coordinates[0])*sharing_rate[0]):].chunk(cardinality[0])
            sharing_y = []
            non_sharing_y = []
            for i in range(cardinality[0]):
                next_x = torch.cat((sharing_x,non_sharing_x[i]),dim=1)
                next_coordinates = [torch.cat((shared_coordinates,non_shared_coordinates[i])),coordinates[1]]
                next_cardinality = cardinality[1:]
                next_sharing_rate = sharing_rate[1:]         
                y_i = self.run(next_x,next_coordinates,next_cardinality,next_sharing_rate)
                sharing_y_i = y_i[:,:int(y_i.size(1)*sharing_rate[0])]
                non_sharing_y_i = y_i[:,int(y_i.size(1)*sharing_rate[0]):]
                sharing_y.append(sharing_y_i)
                non_sharing_y.append(non_sharing_y_i)
            sharing_y = torch.cat(sharing_y,dim=1)
            non_sharing_y = torch.cat(non_sharing_y,dim=1)
            y = torch.cat((sharing_y,non_sharing_y),dim=1)
            return y
        else:
            y = self.cell['main'](x,coordinates)
        return y
                       
    def forward(self, input):
        x = input
        coordinates = [torch.arange(self.cell_info['input_size'],device=device),torch.arange(int(self.cell_info['output_size']//np.prod(self.cell_info['cardinality'])),device=device)]
        x = self.run(x,coordinates,self.cell_info['cardinality'],self.cell_info['sharing_rate'])
        x = self.cell['normalization'](x)
        x = self.cell['activation'](x)
        return x

class ShuffleCell(nn.Module):
    def __init__(self, cell_info):
        super(ShuffleCell, self).__init__()
        self.cell_info = cell_info
        
    def forward(self, input):
        input_size = [*input.size()[:self.cell_info['dim']],*self.cell_info['input_size'],*input.size()[(self.cell_info['dim']+1):]]
        permutation = [i for i in range(len(input.size()[:self.cell_info['dim']]))] + \
                    [self.cell_info['permutation'][i]+self.cell_info['dim'] for i in range(len(self.cell_info['permutation']))] + \
                    [i+self.cell_info['dim']+len(self.cell_info['input_size']) for i in range(len(input.size()[(self.cell_info['dim']+1):]))]
        output_size = [*input.size()[:self.cell_info['dim']],-1,*input.size()[(self.cell_info['dim']+1):]]
        x = input.reshape(input_size)
        x = x.permute(permutation)
        x = x.reshape(output_size)
        return x

class DenseCell(nn.Module):
    def __init__(self, cell_info):
        super(DenseCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = nn.ModuleList([nn.ModuleDict({}) for _ in range(cell_info['num_layer'])])
        for i in range(cell_info['num_layer']):
            cell_in_info = {'input_size':cell_info['input_size'],'output_size':cell_info['bottleneck']*cell_info['growth_rate'],'num_layer':1,'cell':'BasicCell','mode':'fc',
            'normalization':'none','activation':'none'}
            cell_out_info = {'input_size':cell_info['bottleneck']*cell_info['growth_rate'],'output_size':cell_info['growth_rate'],'num_layer':1,'cell':'BasicCell','mode':'pass',
            'normalization':'none','activation':'none'}
            cell[i]['in'] = Cell(cell_in_info)
            cell[i]['out'] = Cell(cell_out_info)
            cell[i]['in_activation'] = Activation(cell_info['activation'])
            cell[i]['out_activation'] = Activation(cell_info['activation'])
            cell[i]['in_normalization'] = Normalization(cell_info['normalization'],cell_info['input_size'])
            cell[i]['out_normalization'] = Normalization(cell_info['normalization'],cell_info['bottleneck']*cell_info['growth_rate'])
            cell_info['input_size'] = cell_info['input_size'] + cell_info['growth_rate']
        return cell
        
    def forward(self, input):
        x = input
        for i in range(len(self.cell)):
            shortcut = x
            x = self.cell[i]['in'](self.cell[i]['in_activation'](self.cell[i]['in_normalization'](x)))
            x = self.cell[i]['out'](self.cell[i]['out_activation'](self.cell[i]['out_normalization'](x)))
            x = torch.cat([shortcut,x], dim=1)
        return x

class DownTransitionCell(nn.Module):
    def __init__(self, cell_info):
        super(DownTransitionCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = []
        if(cell_info['mode'] == 'avg'):
            cell.append(Cell({'cell':'PoolCell','mode':'avg','kernel_size':2}))
        elif(cell_info['mode'] == 'cnn'):
            cell.append(Cell({'cell':'Conv2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size'],
                        'kernel_size':2,'stride':2,'padding':0,'dilation':1,'group':1,'bias':False,
                        'normalization':cell_info['normalization'],'activation':cell_info['activation']}))
        elif(cell_info['mode'] == 'cnn_avg'):
            cell.append(Cell({'cell':'Conv2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size'],
                        'kernel_size':1,'stride':1,'padding':0,'dilation':1,'group':1,'bias':False,
                        'normalization':cell_info['normalization'],'activation':cell_info['activation']}))
            cell.append(Cell({'cell':'PoolCell','mode':'avg','kernel_size':2}))
        elif(cell_info['mode'] == 'dense_cnn_avg'):
            cell.append(Cell({'cell':'Normalization','input_size':cell_info['input_size'],'mode':'bn'}))
            cell.append(Cell({'cell':'Activation','mode':'relu'}))
            cell.append(Cell({'cell':'Conv2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size'],
                        'kernel_size':1,'stride':1,'padding':0,'dilation':1,'group':1,'bias':False,
                        'normalization':'none','activation':'none'}))
            cell.append(Cell({'cell':'PoolCell','mode':'avg','kernel_size':2}))
        elif(cell_info['mode'] == 'pixelshuffle'):
            cell.append(Cell({'cell':'PixelShuffleCell','mode':'down','scale_factor':2}))
        else:
            raise ValueError('model mode not supported')
        cell = nn.Sequential(*cell)
        return cell
        
    def forward(self, input):
        x = input
        x = self.cell(x)
        return x

class UpTransitionCell(nn.Module):
    def __init__(self, cell_info):
        super(UpTransitionCell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()

    def make_cell(self):
        cell_info = copy.deepcopy(self.cell_info)
        cell = []
        if(cell_info['mode'] == 'cnn'):
            cell.append(Cell({'cell':'ConvTranspose2d','input_size':cell_info['input_size'],'output_size':cell_info['output_size'],
                        'kernel_size':2,'stride':2,'padding':0,'output_padding':0,'dilation':1,'group':1,'bias':cell_info['bias'],
                        'normalization':cell_info['normalization'],'activation':cell_info['activation']}))
        elif(cell_info['mode'] == 'pixelshuffle'):
            cell.append(Cell({'cell':'PixelShuffleCell','mode':'up','scale_factor':2}))
        else:
            raise ValueError('model mode not supported')
        cell = nn.Sequential(*cell)
        return cell
        
    def forward(self, input):
        x = input
        x = self.cell(x)
        return x
        
class Cell(nn.Module):
    def __init__(self, cell_info):
        super(Cell, self).__init__()
        self.cell_info = cell_info
        self.cell = self.make_cell()
        
    def make_cell(self):
        if(self.cell_info['cell'] == 'none'):
            cell = nn.Sequential()
        elif(self.cell_info['cell'] == 'Normalization'):
            cell = Normalization(self.cell_info['mode'],self.cell_info['input_size'])
        elif(self.cell_info['cell'] == 'Activation'):
            cell = Activation(self.cell_info['mode'])
        elif(self.cell_info['cell'] == 'Conv2d'):
            default_cell_info = {'kernel_size':3,'stride':1,'padding':1,'dilation':1,'group':1,'bias':False,'normalization':'none','activation':'relu'}
            self.cell_info = {**default_cell_info,**self.cell_info}
            module = nn.Conv2d(self.cell_info['input_size'],self.cell_info['output_size'],self.cell_info['kernel_size'],\
                self.cell_info['stride'],self.cell_info['padding'],self.cell_info['dilation'],self.cell_info['group'],self.cell_info['bias'])
            normalization = Normalization(self.cell_info['normalization'],self.cell_info['output_size'])
            activation = Activation(self.cell_info['activation'])
            cell = nn.Sequential(OrderedDict([
                                  ('module', module),
                                  ('normalization', normalization),
                                  ('activation', activation),
                                ]))
        elif(self.cell_info['cell'] == 'ConvTranspose2d'):
            default_cell_info = {'kernel_size':3,'stride':1,'padding':1,'output_padding':0,'dilation':1,'group':1,'bias':False,'normalization':'none','activation':'relu'}
            self.cell_info = {**default_cell_info,**self.cell_info}
            module = nn.ConvTranspose2d(self.cell_info['input_size'],self.cell_info['output_size'],self.cell_info['kernel_size'],\
                self.cell_info['stride'],self.cell_info['padding'],self.cell_info['output_padding'],self.cell_info['group'],self.cell_info['bias'],self.cell_info['dilation'])
            normalization = Normalization(self.cell_info['normalization'],self.cell_info['output_size'])
            activation = Activation(self.cell_info['activation'])
            cell = nn.Sequential(OrderedDict([
                                  ('module', module),
                                  ('normalization', normalization),
                                  ('activation', activation),
                                ]))
        elif(self.cell_info['cell'] == 'oConv2d'):
            default_cell_info = {'kernel_size':3,'stride':1,'padding':1,'dilation':1,'group':1,'bias':False}
            self.cell_info = {**default_cell_info,**self.cell_info}
            cell = oConv2d(self.cell_info['input_size'],self.cell_info['output_size'],self.cell_info['kernel_size'],\
                self.cell_info['stride'],self.cell_info['padding'],self.cell_info['dilation'],self.cell_info['bias'])
        elif(self.cell_info['cell'] == 'BasicCell'):
            default_cell_info = {'mode':'pass','normalization':'none','activation':'relu','bias':False}
            self.cell_info = {**default_cell_info,**self.cell_info}
            cell = BasicCell(self.cell_info)
        elif(self.cell_info['cell'] == 'ResBasicCell'):
            cell = ResBasicCell(self.cell_info)
        elif(self.cell_info['cell'] == 'LSTMCell'):
            default_cell_info = {'activation':'tanh'}
            self.cell_info = {**default_cell_info,**self.cell_info}
            cell = LSTMCell(self.cell_info)
        elif(self.cell_info['cell'] == 'ResLSTMCell'):
            default_cell_info = {'activation':'tanh'}
            self.cell_info = {**default_cell_info,**self.cell_info}
            cell = ResLSTMCell(self.cell_info)
        elif(self.cell_info['cell'] == 'PixelShuffleCell'):
            cell = PixelShuffleCell(self.cell_info)
        elif(self.cell_info['cell'] == 'PoolCell'):
            default_cell_info = {'kernel_size':2,'stride':None,'padding':0,'dilation':1,'return_indices':False,'ceil_mode':False,'count_include_pad':True}
            self.cell_info = {**default_cell_info,**self.cell_info}
            cell = PoolCell(self.cell_info)
        elif(self.cell_info['cell'] == 'BottleNeckCell'):
            cell = BottleNeckCell(self.cell_info)
        elif(self.cell_info['cell'] == 'CartesianCell'):
            default_cell_info = {'bias':False}
            self.cell_info = {**default_cell_info,**self.cell_info}            
            cell = CartesianCell(self.cell_info)
        elif(self.cell_info['cell'] == 'ShuffleCell'):
            cell = ShuffleCell(self.cell_info)
        elif(self.cell_info['cell'] == 'DenseCell'):
            cell = DenseCell(self.cell_info)
        elif(self.cell_info['cell'] == 'DownTransitionCell'):
            cell = DownTransitionCell(self.cell_info)
        elif(self.cell_info['cell'] == 'UpTransitionCell'):
            cell = UpTransitionCell(self.cell_info)
        else:
            raise ValueError('model mode not supported')
        return cell
        
    def forward(self, *input):
        x = self.cell(*input)
        return x
