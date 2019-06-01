#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python train_model.py --model_name \'vade\' --init_seed 0 --special_TAG \'28\' --init_param_mode \'gmm\' &
CUDA_VISIBLE_DEVICES="1" python train_model.py --model_name \'vade\' --init_seed 0 --special_TAG \'28\' --init_param_mode \'bmm\' &
CUDA_VISIBLE_DEVICES="2" python train_model.py --model_name \'cvade\' --init_seed 0 --special_TAG \'28\' --init_param_mode \'gmm\' &
CUDA_VISIBLE_DEVICES="3" python train_model.py --model_name \'cvade\' --init_seed 0 --special_TAG \'28\' --init_param_mode \'bmm\' &
CUDA_VISIBLE_DEVICES="0" python train_model.py --model_name \'vadebmm\' --init_seed 0 --special_TAG \'28\' --init_param_mode \'gmm\' &
CUDA_VISIBLE_DEVICES="1" python train_model.py --model_name \'vadebmm\' --init_seed 0 --special_TAG \'28\' --init_param_mode \'bmm\' &
CUDA_VISIBLE_DEVICES="2" python train_model.py --model_name \'cvadebmm\' --init_seed 0 --special_TAG \'28\' --init_param_mode \'gmm\' &
CUDA_VISIBLE_DEVICES="3" python train_model.py --model_name \'cvadebmm\' --init_seed 0 --special_TAG \'28\' --init_param_mode \'bmm\' &
