import config

config.init()
import torch
import torch.nn as nn
import models
from data import fetch_dataset, make_data_loader
from utils import collate, to_device, process_control_name, process_dataset


def idx2onehot(idx):
    idx = idx.view(idx.size(0), 1)
    onehot = idx.new_zeros(idx.size(0), config.PARAM['classes_size']).float()
    onehot.scatter_(1, idx, 1)
    return onehot


def init_param(train_loader, model):
    with torch.no_grad():
        model.train(False)
        Z = []
        C = []
        for i, input in enumerate(train_loader):
            input = collate(input)
            input = to_device(input, config.PARAM['device'])
            output = model(input)
            Z.append(output['z'])
            C.append(input['label'])
        Z = torch.cat(Z, dim=0)
        C = idx2onehot(torch.cat(C, dim=0))
        if config.PARAM['mode'] == 'classification':
            nk = C.sum(dim=0, keepdim=True) + 1e-10
            mu = Z.t().matmul(C) / nk
            logvar = torch.log((Z ** 2).t().matmul(C) / nk - 2 * mu * Z.t().matmul(C) / nk + mu ** 2)
        else:
            if config.PARAM['init_param_mode'] == 'random':
                C = torch.distributions.categorical.Categorical(
                    Z.new_ones(config.PARAM['classes_size']) / config.PARAM['classes_size'])
                C = idx2onehot(C.sample((Z.size(0),)))
                nk = C.sum(dim=0, keepdim=True) + 1e-10
                mu = Z.t().matmul(C) / nk
                logvar = torch.log((Z ** 2).t().matmul(C) / nk - 2 * mu * Z.t().matmul(C) / nk + mu ** 2)
            elif config.PARAM['init_param_mode'] == 'kmeans':
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=config.PARAM['classes_size'], n_init=1).fit(Z.cpu().numpy())
                C = idx2onehot(torch.tensor(km.labels_, dtype=torch.long))
                nk = C.sum(dim=0, keepdim=True) + 1e-10
                mu = Z.t().matmul(C) / nk
                logvar = torch.log((Z ** 2).t().matmul(C) / nk - 2 * mu * Z.t().matmul(C) / nk + mu ** 2)
            elif config.PARAM['init_param_mode'] == 'gmm':
                from sklearn.mixture import GaussianMixture
                gm = GaussianMixture(n_components=config.PARAM['classes_size'], covariance_type='diag').fit(
                    Z.cpu().numpy())
                C = idx2onehot(
                    torch.tensor(gm.predict(Z.cpu().numpy()), dtype=torch.long, device=config.PARAM['device']))
                mu = torch.tensor(gm.means_.T, dtype=torch.float, device=config.PARAM['device'])
                logvar = torch.log(torch.tensor(gm.covariances_.T, dtype=torch.float, device=config.PARAM['device']))
            else:
                raise ValueError('Not valid init param')
        model.param['mu'].copy_(mu)
        model.param['logvar'].copy_(logvar)
        return C

if __name__ == "__main__":
    process_control_name()
    dataset = fetch_dataset(config.PARAM['data_name'], config.PARAM['subset'])
    process_dataset(dataset['train'])
    data_loader = make_data_loader(dataset)
    model = eval('models.{}().to(config.PARAM["device"])'.format(config.PARAM['model_name']))
    init_param(data_loader['train'], model)