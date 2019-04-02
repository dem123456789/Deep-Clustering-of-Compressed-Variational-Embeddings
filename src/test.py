#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import numpy as np
#import math
#import config
#from modules import Quantize
#from functions import pixel_unshuffle
#from data import extract_patches_2d, reconstruct_from_patches_2d
#from utils import RGB_to_L, L_to_RGB,dict_to_device
#
#
# def collate(input):
	# for k in input:
		# input[k] = torch.stack(input[k],0)
	# return input

# if __name__ == '__main__':
	# batch_size = 2
	# train_dataset, test_dataset = fetch_dataset('MOSI')
	# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=0, collate_fn=input_collate)
	# print(len(train_dataset))    
	# for i, input in enumerate(train_loader):
		# input = collate(input)
		# input = dict_to_device(input,device)
		# print(input)
		# exit()   

		
# onfig.init()
# device = config.PARAM['device']
# #code_size = 1
# z_dim=10

# classes_size = 10
# mu_p = torch.rand(10, 10)
# var_p = torch.rand(10, 10)
# theta_p = torch.rand(10)

# def loss_fn_base(input, output, protocol):
	# N = output['compression']['code'].size()[0]
	# D = output['compression']['code'].size()[1]
	# K = classes_size

	# Z = output['compression']['code'].unsqueeze(2).expand(N, D, K) # NxDxK
	# z_mean_t = output['compression']['param']['mu'].unsqueeze(2).expand(N, D, K)
	# z_log_var_t = output['compression']['param']['logvar'].unsqueeze(2).expand(N, D, K)

	# u_tensor3 = mu_p.unsqueeze(0).expand(N,D,K) # NxDxK
	# lambda_tensor3 = var_p.unsqueeze(0).expand(N,D,K)
	# theta_tensor2 = theta_p.unsqueeze(0).expand(N, K) # NxK

	# p_c_z = torch.exp(torch.log(theta_tensor2) - torch.sum(0.5*torch.log(2*math.pi*lambda_tensor3)+\
		# (Z-u_tensor3)**2/(2*lambda_tensor3), dim=1)) + 1e-10 # NxK
	# gamma = p_c_z / torch.sum(p_c_z, dim=1, keepdim=True) # NxK

	# BCE = -torch.sum(input['img'].view(-1,1024)*torch.log(torch.clamp(output['compression']['img'].view(-1,1024), min=1e-10))+
		# (1-input['img'].view(-1,1024))*torch.log(torch.clamp(1-output['compression']['img'].view(-1,1024), min=1e-10)), 1)
	# logpzc = torch.sum(0.5*gamma*torch.sum(math.log(2*math.pi)+torch.log(lambda_tensor3)+
		# torch.exp(z_log_var_t)/lambda_tensor3 + (z_mean_t-u_tensor3)**2/lambda_tensor3, dim=1), dim=1)
	
	# qentropy = -0.5*torch.sum(1+output['compression']['param']['logvar']+math.log(2*math.pi), 1)

	# logpc = -torch.sum(torch.log(theta_tensor2)*gamma, 1)
	# logqcx = torch.sum(torch.log(gamma)*gamma, 1)

	# loss = torch.mean(BCE+logpzc+qentropy+logpc+logqcx)
	# # if not config.PARAM['printed']:
	# # 	print(loss)
	# # 	config.PARAM['printed'] = True
	
	# return loss

# def loss_fn(input, output, protocol):
	# # theta_1 = self.theta_p
	# # mu_2 = self.mu_p
	# # var_2 = self.var_p

	# #self.mu_p DxK
	# #self.var_p DxK
	# #self.theta_p 1xK
	# #z-self.mu_p NxDxK
	
	# z = output['compression']['code'].view(output['compression']['code'].size(0),-1,1) #NxDx1
	
	# q_mu = output['compression']['param']['mu'].view(input['img'].size(0),-1,1)
	# q_logvar = output['compression']['param']['logvar'].view(input['img'].size(0),-1,1) #NxDx1
	# q_c_z = torch.exp(torch.log(theta_p) - torch.sum(0.5*torch.log(2*math.pi*var_p) +\
		# (z-mu_p)**2/(2*var_p),dim=1)) + 1e-10
	# #10*np.finfo(np.float32).eps #Nx1
	
	# q_c_z = q_c_z/q_c_z.sum(dim=1,keepdim=True) #Nxk
	
	# BCE = F.binary_cross_entropy(output['compression']['img'],input['img'],reduction='none').view(input['img'].size(0),-1).sum(dim=1)

	# logpzc = torch.sum(0.5*q_c_z*torch.sum(math.log(2*math.pi)+torch.log(var_p)+\
		# torch.exp(q_logvar)/var_p + (q_mu-mu_p)**2/var_p, dim=1), dim=1)

	# qentropy = -0.5*torch.sum(1+output['compression']['param']['logvar']+math.log(2*math.pi), 1)
	# # qentropy = (-0.5*torch.sum(1+q_logvar+math.log(2*math.pi), 1)).squeeze(1)

	# logpc = -torch.sum(torch.log(theta_p)*q_c_z, 1)
	# logqcx = torch.sum(torch.log(q_c_z)*q_c_z, 1)

	# loss = torch.mean(BCE+logpzc+qentropy+logpc+logqcx)
	# # if not config.PARAM['printed']:
	# # 	print(loss)

	# # loss = loss.sum()/input['img'].numel()#input['img'].numel() = 102400
	# return loss

#def main():
#    a = torch.randn(4, 4, 3)
#    print(a)
#    print(torch.sum(a, dim=2))
#    print(torch.sum(a, dim=-1))
#
#if __name__ == "__main__":
#    main()

# import torch
# import torch.nn as nn
# from torch.nn import Parameter
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision
# from torchvision import datasets, transforms
# from torch.autograd import Variable
# from torchvision.utils import save_image

# import numpy as np
# import math
# from sklearn.mixture import GaussianMixture
# from sklearn.utils.linear_assignment_ import linear_assignment



# def cluster_acc(Y_pred, Y):
#     assert Y_pred.size == Y.size
#     D = max(Y_pred.max(), Y.max())+1
#     w = np.zeros((D,D), dtype=np.int64)
#     for i in range(Y_pred.size):
#         w[Y_pred[i], Y[i]] += 1
#         ind = linear_assignment(w.max() - w)
#     return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

# def cluster_ACC(output,target,topk=1):
#     with torch.no_grad():
#         batch_size = target.size(0)
#         pred_k = torch.argmax(output,dim=1)
# #        pred_k = output.topk(topk, 1, True, True)[1]
#         D = max(pred_k.max(), target.max()) + 1
#         w = torch.zeros(D,D)
#         for i in range(batch_size):
#             w[pred_k[i], target[i]] += 1
#         ind = linear_assignment(w.max() - w)
#         correct_k = sum([w[i,j] for i,j in ind])
#         cluster_acc = (correct_k*(100.0 / batch_size)).item()
#     return cluster_acc

# def main():
#     A = torch.rand(128,10)
#     B = torch.LongTensor(128).random_(0, 9)
#     A_pred = torch.argmax(A,dim=1)

#     acc1 = cluster_acc(A_pred.numpy(),B.numpy())
#     print(acc1)
#     acc2 = cluster_ACC(A,B)
#     print(acc2)
#     print(torch.all(torch.eq(acc1,acc2)))

# if __name__ == "__main__":
#     main()
