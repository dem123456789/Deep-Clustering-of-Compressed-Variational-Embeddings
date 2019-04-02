import torch

def init():
    global PARAM
    PARAM = {
        'model_data_name': 'MNIST',
        'train_data_name': 'MNIST',
        'test_data_name': 'MNIST',
        'model_dir': 'mnist',
        'model_name': 'Joint_cvae_bmm',
        'resume_TAG': '',
        'special_TAG': '',
        'optimizer_name': 'Adam',
        'scheduler_name': 'Vade',
        'factor': 0.5,
        'normalize': False,
        'branch': False,
        'balance': False,
        'batch_size': [128,200],
        'num_workers': 0,
        'data_size': 0,
        'device': 'cpu',
        'max_num_epochs': 100,
        'save_mode': 0,
        'world_size': 1,
        'train_metric_names': ['psnr','cluster_acc'],
        'test_metric_names': ['psnr','cluster_acc'],
        'topk': 1,
        'init_seed': 0,
        'num_Experiments': 1,
        'tuning_param': {'compression': 1, 'classification': 1},
        'loss_mode': {'compression':'bce','classification':'ce'},
        'code_size': 8,
        'cell_name': 'BasicCell',
        'num_iter': 16,
        'max_depth': 3,
        'patch_shape': (32,32),
        'step': [1.0,1.0],
        'jump_rate': 2,
        'train_num_node': 8,
        'test_num_node': 8,
        'resume_mode': 0,
        'printed': False,
        'temp': 1.0
    }
