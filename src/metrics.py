import torch
import torch.nn as nn
import config
import numbers
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import *
from sklearn.utils.linear_assignment_ import linear_assignment
from utils import dict_to_device
config.init()

def PSNR(output,target,max=1.0):
    MAX = torch.tensor(max).to(target.device)
    criterion = nn.MSELoss().to(target.device)
    MSE = criterion(output,target)
    psnr = (20*torch.log10(MAX)-10*torch.log10(MSE)).item()
    return psnr
    
def BPP(code,img):
    if(isinstance(code,np.ndarray)):
        nbytes = code.nbytes
    elif(isinstance(code,list)):
        nbytes = 0 
        for i in range(len(code)):
            nbytes += code[i].nbytes
    else:
        raise ValueError('Code data type not supported')
    num_pixel = img.numel()/img.size(1)
    bpp = 8*nbytes/num_pixel
    return bpp

def flatten_output(output):
    if(output['child'] is None):
        return F.log_softmax(output['this'],1)
    else:
        flat_output = []
        for i in range(len(output['child'])):
            output_i = F.log_softmax(output['this'],1)[:,[i]]
            flat_output.append(output_i+flatten_output(output['child'][i]))
        flat_output = torch.cat(flat_output,1)
    return flat_output
        
def ACC(output,target,topk=1):  
    with torch.no_grad():
        batch_size = target.size(0)
        pred_k = output.topk(topk, 1, True, True)[1]
        correct_k = pred_k.eq(target.view(-1,1).expand_as(pred_k)).float().sum()
        acc = (correct_k*(100.0 / batch_size)).item()
    return acc
    
def cluster_ACC(output,target,topk=1):
    with torch.no_grad():
        batch_size = target.size(0)
        pred_k = output.topk(topk, 1, True, True)[1]
        D = max(pred_k.max(), target.max()) + 1
        w = torch.zeros(D,D)    
        for i in range(batch_size):
            w[pred_k[i], target[i]] += 1
        ind = linear_assignment(w.max() - w)
        correct_k = sum([w[i,j] for i,j in ind])
        cluster_acc = (correct_k*(100.0 / batch_size)).item()
    return cluster_acc
    
def F1(output,target,topk=1):  
    with torch.no_grad():
        pred_k = output.topk(topk, 1, True, True)[1]
        correct_k = pred_k.eq(target.view(-1,1).expand_as(pred_k)).sum(dim=1).byte()
        pred = pred_k[:,0]
        pred[correct_k,] = target[correct_k,]
        f1 = f1_score(target.numpy(),pred.numpy(),average='macro')
    return f1

def Precision(output,target,topk=1):  
    with torch.no_grad():
        pred_k = output.topk(topk, 1, True, True)[1]
        correct_k = pred_k.eq(target.view(-1,1).expand_as(pred_k)).sum(dim=1).byte()
        pred = pred_k[:,0]
        pred[correct_k,] = target[correct_k,]
        precision = precision_score(target.numpy(),pred.numpy(),average='macro')
    return precision

def Recall(output,target,topk=1):  
    with torch.no_grad():
        pred_k = output.topk(topk, 1, True, True)[1]
        correct_k = pred_k.eq(target.view(-1,1).expand_as(pred_k)).sum(dim=1).byte()
        pred = pred_k[:,0]
        pred[correct_k,] = target[correct_k,]
        recall = recall_score(target.numpy(),pred.numpy(),average='macro')
    return recall
    
class Meter_Panel(object):

    def __init__(self,meter_names):
        self.meter_names = meter_names
        self.panel = {k: Meter() for k in meter_names}
        self.metric = Metric(meter_names)

    def reset(self):
        for k in self.panel:
            self.panel[k].reset()
        self.metric.reset()
        return
        
    def update(self, new, n=1):
        if(isinstance(new, Meter_Panel)):
            for i in range(len(new.meter_names)):
                if(new.meter_names[i] in self.panel):
                    self.panel[new.meter_names[i]].update(new.panel[new.meter_names[i]])
                else:
                    self.panel[new.meter_names[i]] = new.panel[new.meter_names[i]]
                    self.meter_names += [new.meter_names[i]]
        elif(isinstance(new, dict)):
            for k in new:
                if(k not in self.panel):
                    self.panel[k] = Meter()
                    self.meter_names += [k]
                if(isinstance(n,int)):
                    self.panel[k].update(new[k],n)
                else:
                    self.panel[k].update(new[k],n[k])
        else:
            raise ValueError('Not supported data type for updating meter panel')
        return
        
    def eval(self, input, output, protocol):
        tuning_param = protocol['tuning_param']
        metric_names = protocol['metric_names']
        evaluation = self.metric.eval(input,output,protocol)
        return evaluation
        
    def summary(self,names):
        fmt_str = ''
        if('loss' in names and 'loss' in self.panel):
            fmt_str += '\tLoss: {:.4f}'.format(self.panel['loss'].avg)
        if('bpp' in names and 'bpp' in self.panel):
            fmt_str += '\tBPP: {:.4f}'.format(self.panel['bpp'].avg)
        if('psnr' in names and 'psnr' in self.panel):
            fmt_str += '\tPSNR: {:.4f}'.format(self.panel['psnr'].avg)
        if('acc' in names and 'acc' in self.panel):
            fmt_str += '\tACC: {:.4f}'.format(self.panel['acc'].avg)
        if('cluster_acc' in names and 'cluster_acc' in self.panel):
            fmt_str += '\tACC: {:.4f}'.format(self.panel['cluster_acc'].val)
        if('batch_time' in names and 'batch_time' in self.panel):
            fmt_str += '\tBatch Time: {:.4f}'.format(self.panel['batch_time'].avg)
        return fmt_str
                    
                
class Meter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history_val = []
        self.history_avg = [0]
        return
        
    def update(self, new, n=1):
        if(isinstance(new,Meter)):
            self.val = new.val
            self.avg = new.avg
            self.sum = new.sum
            self.count = new.count
            self.history_val.extend(new.history_val)
            self.history_avg.extend(new.history_avg)
        elif(isinstance(new,numbers.Number)):
            self.val = new
            self.sum += new * n
            self.count += n
            self.avg = self.sum / self.count
            self.history_val.append(self.val)
            self.history_avg[-1] = self.avg
        else:
            self.val = new
            self.count += n
            self.history_val.append(self.val)
        return
        
        
class Metric(object):
    
    batch_metric_names = ['psnr','bpp','acc']
    full_metric_names = ['cluster_acc','f1','precsion','recall','prc','roc','roc_auc']
    
    def __init__(self, metric_names):
        self.reset(metric_names)
        
    def reset(self, metric_names):
        self.metric_names = metric_names
        self.if_save = not set(self.metric_names).isdisjoint(self.full_metric_names)
        self.score = None
        self.label = None
        return
        
    def eval(self, input, output, protocol):
        tuning_param = protocol['tuning_param']
        metric_names = protocol['metric_names']
        evaluation = {}
        evaluation['loss'] = output['loss'].item()
        if(tuning_param['compression'] > 0):
            if('psnr' in metric_names):
                evaluation['psnr'] = PSNR(output['compression']['img'],input['img'])
            if('bpp' in metric_names):
                evaluation['bpp'] = BPP(output['compression']['code'],input['img'])
        if(tuning_param['classification'] > 0):
            topk=protocol['topk']
            if(self.if_save):
                self.score = torch.cat((self.score,output['classification'].cpu()),0) if self.score is not None else output['classification'].cpu()
                self.label = torch.cat((self.label,input['label'].cpu()),0) if self.label is not None else input['label'].cpu()
            if('acc' in metric_names):
                evaluation['acc'] = ACC(output['classification'],input['label'],topk=topk)
            if('cluster_acc' in metric_names):
                evaluation['cluster_acc'] = cluster_ACC(self.score,self.label,topk=topk)
            if('f1' in metric_names):
                evaluation['f1'] = F1(self.score,self.label,topk=topk)
            if('precision' in metric_names):
                evaluation['precision'] = Precision(self.score.self.label,topk=topk)
            if('recall' in metric_names):
                evaluation['recall'] = Recall(self.score,self.label,topk=topk)
            if('prc' in metric_names):
                evaluation['prc'] = PRC(self.score,self.label)
            if('roc' in metric_names):
                evaluation['roc'] = ROC(self.score,self.label)
            if('roc_auc' in metric_names):
                evaluation['roc_auc'] = ROC_AUC(self.score,self.label)
        return evaluation
   