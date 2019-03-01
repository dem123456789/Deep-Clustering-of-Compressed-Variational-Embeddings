import torch
import config
import time
import torch.backends.cudnn as cudnn
import models
import torch.optim as optim
import os
import datetime
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from data import *
from utils import *
from metrics import *

cudnn.benchmark = True
temp_min = 0.5
ANNEAL_RATE = 0.00003

config.init()
for k in config.PARAM:
	exec('{0} = config.PARAM[\'{0}\']'.format(k))
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
	
	train_dataset,_ = fetch_dataset(data_name=train_data_name)
	validated_num_epochs = max_num_epochs
	valid_data_size = len(train_dataset) if(data_size==0) else data_size
	train_loader,test_loader = split_dataset(train_dataset,train_dataset,valid_data_size,batch_size=batch_size,radomGen=randomGen)
	print('Training data size {}, Number of Batches {}, Test data size {}'.format(valid_data_size,len(train_loader),len(train_dataset)))      
	last_epoch = 0
	model = eval('models.{}.{}(train_dataset.classes_size).to(device)'.format(model_dir,model_name))
	param_protocol = init_param_protocol(train_dataset,randomGen)
	init_param(train_loader,model,param_protocol)
	optimizer = make_optimizer(optimizer_name,model)
	scheduler = make_scheduler(scheduler_name,optimizer)
	if(resume_mode == 2):
		_,model,_,_ = resume(model,optimizer,scheduler,resume_model_TAG)       
	elif(resume_mode == 1):
		last_epoch,model,optimizer,scheduler = resume(model,optimizer,scheduler,resume_model_TAG)
	if(world_size > 1):
		model = torch.nn.DataParallel(model,device_ids=list(range(world_size)))
	best_pivot = 255
	best_pivot_name = 'loss'
	train_meter_panel = Meter_Panel(config.PARAM['train_metric_names'])
	test_meter_panel = Meter_Panel(config.PARAM['test_metric_names'])
	for epoch in range(last_epoch, validated_num_epochs+1):
		train_protocol = init_train_protocol(train_dataset)
		test_protocol = init_test_protocol(train_dataset)
		cur_train_meter_panel = train(train_loader,model,optimizer,epoch,train_protocol)
		cur_test_meter_panel = test(test_loader,model,epoch,test_protocol,model_TAG)
		print_result(epoch,cur_train_meter_panel,cur_test_meter_panel)
		# scheduler.step(cur_test_meter_panel.panel['loss'].avg)
		scheduler.step(epoch)
		train_meter_panel.update(cur_train_meter_panel)
		test_meter_panel.update(cur_test_meter_panel)
		if(save_mode>=0):
			model_state_dict = model.module.state_dict() if(world_size > 1) else model.state_dict()
			save_result = {'config':config.PARAM,'epoch':epoch+1,'model_dict':model_state_dict,'optimizer_dict':optimizer.state_dict(),
				'scheduler_dict': scheduler.state_dict(),'train_meter_panel':train_meter_panel,'test_meter_panel':test_meter_panel}
			save(save_result,'./output/model/{}_checkpoint.pkl'.format(model_TAG))
			if(best_pivot > test_meter_panel.panel[best_pivot_name].avg):
				best_pivot = test_meter_panel.panel[best_pivot_name].avg
				save(save_result,'./output/model/{}_best.pkl'.format(model_TAG))
	return
	
def train(train_loader,model,optimizer,epoch,protocol):
	meter_panel = Meter_Panel(protocol['metric_names'])
	model.train(True)
	end = time.time()
	for i, input in enumerate(train_loader):
		input = collate(input)
		input = dict_to_device(input,device)
		protocol = update_protocol(input,i,len(train_loader),protocol)
		output = model(input,protocol)
		output['loss'] = torch.mean(output['loss']) if(world_size > 1) else output['loss']                                                                                          
		optimizer.zero_grad()
		output['loss'].backward()
		optimizer.step()
		if (config.PARAM['tuning_param']=='Joint_cvae_bmm' and i % 100) == 1:
			model.temp = np.maximum(model.temp * np.exp(-ANNEAL_RATE * i), temp_min)
		evaluation = meter_panel.eval(input,output,protocol)
		batch_time = time.time() - end
		meter_panel.update(evaluation,input['img'].size(0))
		meter_panel.update({'batch_time':batch_time})
		end = time.time()
		if(i % (len(train_loader)//5) == 0):
			estimated_finish_time = str(datetime.timedelta(seconds=(len(train_loader)-i-1)*batch_time))
			print('Train Epoch: {}[({:.0f}%)]{}, Estimated Finish Time: {}'.format(
				epoch, 100. * i / len(train_loader), meter_panel.summary(['loss','psnr','cluster_acc','batch_time']), estimated_finish_time))
	return meter_panel

def test(validation_loader,model,epoch,protocol,model_TAG):
	meter_panel = Meter_Panel(protocol['metric_names'])
	with torch.no_grad():
		model.train(False)
		end = time.time()
		for i, input in enumerate(validation_loader):
			input = collate(input)
			input = dict_to_device(input,device)
			protocol = update_protocol(input,i,len(validation_loader),protocol)
			output = model(input,protocol)
			output['loss'] = torch.mean(output['loss']) if(world_size > 1) else output['loss']
			if (config.PARAM['tuning_param']=='Joint_cvae_bmm' and i % 100) == 1:
				model.temp = np.maximum(model.temp * np.exp(-ANNEAL_RATE * i), temp_min)
			evaluation = meter_panel.eval(input,output,protocol)
			batch_time = time.time() - end
			meter_panel.update(evaluation,input['img'].size(0))
			meter_panel.update({'batch_time':batch_time})
			end = time.time()
		if(tuning_param['compression'] > 0):
			save_img(input['img'],'./output/img/image.png')
			save_img(output['compression']['img'],'./output/img/image_{}_{}.png'.format(model_TAG,epoch))
	return meter_panel

def make_optimizer(optimizer_name,model):
	if(optimizer_name=='Adam'):
		optimizer = optim.Adam(model.parameters(),lr=1e-3)
	elif(optimizer_name=='SGD'):
		optimizer = optim.SGD(model.parameters(),lr=1e-1, momentum=0.9)
	else:
		raise ValueError('Optimizer name not supported')
	return optimizer
	
def make_scheduler(scheduler_name,optimizer):
	if(scheduler_name=='MultiStepLR'):
		scheduler = MultiStepLR(optimizer,milestones=[10,20,50,100],gamma=factor)
	elif(scheduler_name=='ReduceLROnPlateau'):
		scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=factor,verbose=True,threshold=1,threshold_mode='abs')
	elif(scheduler_name=='Vade'):
		scheduler = adjust_learning_rate(optimizer, min_lr=0.0002)
	else:
		raise ValueError('Scheduler_name name not supported')
	return scheduler

def init_param_protocol(dataset,randomGen):
	protocol = {}
	protocol['run_mode'] = 'test'
	protocol['tuning_param'] = config.PARAM['tuning_param'].copy()
	protocol['metric_names'] = config.PARAM['test_metric_names'].copy()
	protocol['topk'] = config.PARAM['topk']
	protocol['loss_mode'] = config.PARAM['loss_mode']
	if(config.PARAM['balance']):
		protocol['classes_counts'] = dataset.classes_counts.expand(world_size,-1).to(device)
	protocol['init_param_mode'] = 'gmm'
	protocol['classes_size'] = dataset.classes_size
	protocol['randomGen'] = randomGen
	protocol['tuning_param']['classification'] = 0
	return protocol
		
def init_param(train_loader,model,protocol):
	with torch.no_grad():
		model.train(False)
		for i, input in enumerate(train_loader):
			input = collate(input)
			input = dict_to_device(input,device)
			protocol = update_protocol(input,i,len(train_loader),protocol)
			output = model(input,protocol)
			z = output['compression']['code'].view(input['img'].size(0),-1)
			Z = torch.cat((Z,z),0) if i > 0 else z
		if(protocol['init_param_mode'] == 'random'):
			C = torch.rand(Z.size(0), protocol['classes_size'],device=device)
			nk = C.sum(dim=0,keepdim=True) + 10*np.finfo(np.float32).eps
			model.param['mu'].copy_(Z.t().matmul(C)/nk)
			model.param['var'].copy_((Z**2).t().matmul(C)/nk - 2*model.param['mu']*Z.t().matmul(C)/nk + model.param['mu']**2)
		elif(protocol['init_param_mode'] == 'kmeans'):
			from sklearn.cluster import KMeans
			C = Z.new_zeros(Z.size(0), protocol['classes_size'])
			km = KMeans(n_clusters=protocol['classes_size'], n_init=1, random_state=protocol['randomGen']).fit(Z.cpu().numpy())
			C[torch.arange(C.size(0)), torch.tensor(km.labels_).long()] = 1
			nk = C.sum(dim=0,keepdim=True) + 10*np.finfo(np.float32).eps
			model.param['mu'].copy_(Z.t().matmul(C)/nk)
			model.param['var'].copy_((Z**2).t().matmul(C)/nk - 2*model.param['mu']*Z.t().matmul(C)/nk + model.param['mu']**2)
		elif(protocol['init_param_mode'] == 'gmm'):
			from sklearn.mixture import GaussianMixture
			gm = GaussianMixture(n_components=protocol['classes_size'], covariance_type='diag', random_state=protocol['randomGen']).fit(Z.cpu().numpy())
			model.param['mu'].copy_(torch.tensor(gm.means_.T).float().to(device))
			model.param['logvar'].copy_(torch.log(torch.tensor(gm.covariances_.T).float().to(device)))
		elif(protocol['init_param_mode'] == 'bmm'):
			from bmm_implement import BMM
			Z = torch.argmax(Z.view(input['img'].size(0),32,2),dim=2)
			bmm = BMM(n_comp=10,n_iter=300).fit(Z.cpu().numpy())
			bmmq = torch.tensor(bmm.q).float().to(device)
			# torch.log(bmmq/(1-bmmq))
			# qt = torch.t(torch.tensor(bmm.q).float().to(device))
			model.param['mean'].copy_(torch.log(bmmq/(1-bmmq))) #p(z=1|c=k)
		else:
			raise ValueError('Initialization method not supported')
	return
		
def init_train_protocol(dataset):
	protocol = {}
	protocol['run_mode'] = 'train'
	protocol['tuning_param'] = config.PARAM['tuning_param'].copy()
	protocol['metric_names'] = config.PARAM['train_metric_names'].copy()
	protocol['topk'] = config.PARAM['topk']
	protocol['loss_mode'] = config.PARAM['loss_mode']
	if(config.PARAM['balance']):
		protocol['classes_counts'] = dataset.classes_counts.expand(world_size,-1).to(device)
	return protocol 

def init_test_protocol(dataset):
	protocol = {}
	protocol['run_mode'] = 'test'
	protocol['tuning_param'] = config.PARAM['tuning_param'].copy()
	protocol['metric_names'] = config.PARAM['test_metric_names'].copy()
	protocol['topk'] = config.PARAM['topk']
	protocol['loss_mode'] = config.PARAM['loss_mode']
	if(config.PARAM['balance']):
		protocol['classes_counts'] = dataset.classes_counts.expand(world_size,-1).to(device)
	return protocol
	
def collate(input):
	for k in input:
		input[k] = torch.stack(input[k],0)
	return input

def update_protocol(input,i,num_batch,protocol):
	protocol['num_iter'] = config.PARAM['num_iter']
	protocol['depth'] = config.PARAM['max_depth']
	protocol['jump_rate'] = config.PARAM['jump_rate']
	protocol['patch_shape'] = config.PARAM['patch_shape']
	protocol['img_shape'] = (input['img'].size(2),input['img'].size(3))
	protocol['step'] = config.PARAM['step']
	if(protocol['run_mode'] == 'train'):
		if('cluster_acc' in protocol['metric_names'] and i % (num_batch//5) != 0):
			protocol['metric_names'].remove('cluster_acc')
		elif('cluster_acc' not in protocol['metric_names'] and i % (num_batch//5) == 0):
			protocol['metric_names'].append('cluster_acc')
	elif(protocol['run_mode'] == 'test'):
		if(i == 0):
			protocol['metric_names'].remove('cluster_acc') 
		elif(i == num_batch-1):
			protocol['metric_names'].append('cluster_acc')
	if(input['img'].size(1)==1):
		protocol['mode'] = 'L'
	elif(input['img'].size(1)==3):
		protocol['mode'] = 'RGB'
	else:
		raise ValueError('Wrong number of channel')
	return protocol 

def print_result(epoch,train_meter_panel,test_meter_panel):
	estimated_finish_time = str(datetime.timedelta(seconds=(max_num_epochs - epoch - 1)*train_meter_panel.panel['batch_time'].sum))
	print('Test Epoch: {}{}{}, Estimated Finish Time: {}'.format(epoch,test_meter_panel.summary(['loss','psnr','cluster_acc']),train_meter_panel.summary(['batch_time']),estimated_finish_time))
	return

def resume(model,optimizer,scheduler,resume_model_TAG):
	if(os.path.exists('./output/model/{}_checkpoint.pkl'.format(resume_model_TAG))):
		checkpoint = load('./output/model/{}_checkpoint.pkl'.format(resume_model_TAG))
		last_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['model_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_dict'])
		print('Resume from {}'.format(last_epoch))
	else:
		last_epoch = 0
		print('Not found existing model, and start from epoch {}'.format(last_epoch))
	return last_epoch,model,optimizer,scheduler
	
	
if __name__ == "__main__":
	main()   
