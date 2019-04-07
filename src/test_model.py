import torch
import config
config.init()
import time
import torch.backends.cudnn as cudnn
import models
import os
import datetime
import argparse
import itertools
from torch import nn
from data import *
from utils import *
from metrics import *

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Config')
for k in config.PARAM:
    exec('parser.add_argument(\'--{0}\',default=config.PARAM[\'{0}\'], help=\'\')'.format(k))
args = vars(parser.parse_args())
for k in config.PARAM:
    if(config.PARAM[k]!=args[k]):
        exec('config.PARAM[\'{0}\'] = {1}'.format(k,args[k]))
for k in config.PARAM:
    exec('{0} = config.PARAM[\'{0}\']'.format(k))
    
def main():
    seeds = list(range(init_seed,init_seed+num_Experiments))
    for i in range(num_Experiments):
        resume_model_TAG = '{}_{}_{}'.format(seeds[i],model_data_name,model_name) if(resume_TAG=='') else '{}_{}_{}_{}'.format(seeds[i],model_data_name,model_name,resume_TAG)
        model_TAG = resume_model_TAG if(special_TAG=='') else '{}_{}'.format(resume_model_TAG,special_TAG)
        print('Experiment: {}'.format(model_TAG))
        result = runExperiment(model_TAG)
        save(result,'./output/result/{}.pkl'.format(model_TAG))  
    return

def runExperiment(model_TAG):
    model_TAG_list = model_TAG.split('_')
    seed = int(model_TAG_list[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    randomGen = np.random.RandomState(seed)
    
    print(config.PARAM)
    _,test_dataset = fetch_dataset(data_name=test_data_name)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*world_size, pin_memory=True, num_workers=num_workers*world_size, collate_fn = input_collate)
    best = load('./output/model/{}_best.pkl'.format(resume_model_TAG))
    last_epoch = best['epoch']
    print('Test from {}'.format(last_epoch))
    model = eval('models.{}.{}(classes_size=test_dataset.classes_size).to(device)'.format(model_dir,model_name)) 
    model.load_state_dict(best['model_dict'])
    test_protocol = init_test_protocol(test_dataset)
    result = test(test_loader,model,last_epoch,test_protocol)
    print_result(model_TAG,last_epoch,result) 
    return result
            
def test(validation_loader,model,epoch,protocol):
    meter_panel = Meter_Panel(metric_names)
    with torch.no_grad():
        model.train(False)
        end = time.time()
        for i, input in enumerate(validation_loader):
            input = collate(input)
            input = dict_to_device(input,device)
            protocol = update_test_protocol(input,protocol)  
            output = model(input,protocol)
            output['loss'] = torch.mean(output['loss']) if(world_size > 1) else output['loss']
            evaluation = meter_panel.eval(input,output,protocol)
            batch_time = time.time() - end
            meter_panel.update(evaluation,batch_size)
            meter_panel.update({'batch_time':batch_time})
            end = time.time()
    return meter_panel
     
def init_test_protocol(dataset):
    protocol = {}
    protocol['tuning_param'] = config.PARAM['tuning_param'].copy()
    protocol['metric_names'] = config.PARAM['test_metric_names'].copy()
    protocol['loss_mode'] = config.PARAM['loss_mode']                                                        
    return protocol
    
def collate(input):
    for k in input:
        input[k] = torch.stack(input[k],0)
    return input

def update_test_protocol(input,protocol):
    if(input['img'].size(1)==1):
        protocol['img_mode'] = 'L'
    elif(input['img'].size(1)==3):
        protocol['img_mode'] = 'RGB'
    else:
        raise ValueError('Wrong number of channel')
    return protocol
        
def print_result(model_TAG,epoch,result):
    print('Test Epoch({}): {}({}_{}){}'.format(model_TAG,epoch,result.summary(['loss']+config.PARAM['test_metric_names'])))
    return
    
if __name__ == "__main__":
    main()    