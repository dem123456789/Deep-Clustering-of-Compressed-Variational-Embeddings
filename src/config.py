import torch

def init():
    global PARAM
    PARAM = {
        'model_data_name': 'MNIST',
        'train_data_name': 'MNIST',
        'test_data_name': 'MNIST',
        'model_dir': 'mnist',
        'model_name': 'VariationalAutoEncoder',
        'resume_TAG': '',
        'special_TAG': '',
        'lr': 1e-1,
        'milestones': [150,250],
        'gamma': 0.1,
        'branch': False,
        'balance': False,
        'batch_size': 128,
        'num_workers': 0,
        'data_size': 0,
        'device': 'cuda:0',
        'device_ids': [0,1],
        'max_num_epochs': 350,
        'save_mode': 0,
        'world_size': 1,
        'metric_names': ['psnr','acc'],
        'topk': 1,
        'init_seed': 0,
        'num_Experiments': 1,
        'tuning_param': {'compression': 1, 'classification': 0},
        'num_iter': 16,
        'max_depth': 3,
        'patch_shape': (32,32),
        'step': [1.0,1.0],
        'jump_rate': 2,
        'num_nodes': 2,
        'resume_mode': 0
    }