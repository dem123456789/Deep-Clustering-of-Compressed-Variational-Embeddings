import torch
import config
import time
import torch.backends.cudnn as cudnn
import models
from data import *
from utils import *

cudnn.benchmark = True
config.init()
for k in config.PARAM:
    exec('{0} = config.PARAM[\'{0}\']'.format(k))
init_seed = 0
seeds = list(range(init_seed,init_seed+num_Experiments))

def main():
    for i in range(num_Experiments):
        print('Experiment: {}'.format(seeds[i]))
        runExperiment(seeds[i])
    return
        
def runExperiment(seed):
    print(config.PARAM)
    resume_model_TAG = '{}_{}_{}'.format(seed,model_data_name,model_name) if(resume_TAG=='') else '{}_{}_{}_{}'.format(seed,model_data_name,model_name,resume_TAG)
    model_TAG = resume_model_TAG if(special_TAG=='') else '{}_{}'.format(resume_model_TAG,special_TAG)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    randomGen = np.random.RandomState(seed)
    
    _,test_dataset = fetch_dataset(data_name=test_data_name)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*world_size, pin_memory=True, num_workers=num_workers*world_size, collate_fn = input_collate)
    best = load('./output/model/{}_best.pkl'.format(resume_model_TAG))
    last_epoch = best['epoch']
    print('Test from {}'.format(last_epoch))
    bpp = np.zeros(num_iter)
    loss = np.zeros(num_iter)
    psnr = np.zeros(num_iter)
    acc = np.zeros(num_iter)
    for i in range(1,num_iter+1):
        model = eval('models.{}.{}().to(device)'.format(model_dir,model_name))
        model.load_state_dict(best['model_dict'])
        test_result = test(test_loader,model,last_epoch,i,model_TAG)
        print_result(test_result)
        bpp[i-1] = test_result['bpp'].avg
        loss[i-1] = test_result['loss'].avg
        psnr[i-1] = test_result['psnr'].avg
        acc[i-1] = test_result['acc'].avg
    save({'epoch':last_epoch,'bpp':bpp,'loss':loss,'psnr':psnr,'acc':acc},'./output/result/{}.pkl'.format(model_TAG))  
    return

    
def test(validation_loader,model,epoch,iter,model_TAG):
    entropy_codec = models.classic.Entropy()
    Meter_names = ['batch_time','data_time','loss','bpp','psnr','acc']
    Meter_Panel =  {k: Meter() for k in Meter_names}
    model.eval()
    with torch.no_grad():
        end = time.time()
        output = {}
        for i, input in enumerate(validation_loader):
            input = input_to_device(input,device)
            protocol = set_protocol(input,iter)
            Meter_Panel['data_time'].update(time.time() - end)
            code = model.codec.encode(input,protocol)
            entropy_code = entropy_codec.encode(code,protocol)
            decoded_code = entropy_codec.decode(entropy_code,protocol)
            output['classification'] = model.classifier.classify(decoded_code,protocol) if tuning_param['classification'] > 0 else 0
            output['compression'] = model.codec.decode(decoded_code,protocol)
            loss = model.loss_fn(output,input,protocol).item()           
            bpp = BPP(entropy_code.nbytes,input['img'])
            psnr = PSNR(output['compression'],input['img'],1.0).item() if tuning_param['compression'] > 0 else 0
            acc = ACC(output['classification'],input['label'],topk=topk)[0] if tuning_param['classification'] > 0 else 0
            Meter_Panel['bpp'].update(bpp, input['img'].size(0))
            Meter_Panel['loss'].update(loss, input['img'].size(0))
            Meter_Panel['psnr'].update(psnr, input['img'].size(0))
            Meter_Panel['acc'].update(acc, input['img'].size(0))
            Meter_Panel['batch_time'].update(time.time() - end)
            end = time.time()
        save_img(input['img'],'./output/img/image.png')
        save_img(output['compression'],'./output/img/image_{}_{}_{}.png'.format(model_TAG,epoch,iter))
    return Meter_Panel

def set_protocol(input,iter):
    protocol = {}
    protocol['tuning_param'] = config.PARAM['tuning_param']
    protocol['num_iter'] = iter
    protocol['depth'] = config.PARAM['max_depth']
    protocol['img_shape'] = (input['img'].size(2),input['img'].size(3))
    protocol['patch_shape'] = config.PARAM['patch_shape']
    protocol['step'] = config.PARAM['step']
    protocol['jump_rate'] = config.PARAM['jump_rate']
    if(input['img'].size(1)==1):
        protocol['mode'] = 'L'
    elif(input['img'].size(1)==3):
        protocol['mode'] = 'RGB'
    else:
        raise ValueError('Wrong number of channel')
    return protocol

def print_result(test_result):
    print('Test: BPP: {bpp.avg:.4f}\tLoss: {loss.avg:.4f}\tPSNR: {psnr.avg:.4f}\tACC: {acc.avg:.4f}'
        .format(loss=test_result['loss'],bpp=test_result['bpp'],psnr=test_result['psnr'],acc=test_result['acc']))
    return
    
if __name__ == "__main__":
    main()