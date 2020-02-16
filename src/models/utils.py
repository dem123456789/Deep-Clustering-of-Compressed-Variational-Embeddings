import config
import torch
import torch.nn as nn
from modules import make_cell
from utils import collate, to_device


def make_model(model):
    if isinstance(model, dict):
        if 'cell' in model:
            return make_cell(model)
        elif 'nn' in model:
            return eval(model['nn'])
        else:
            cell = nn.ModuleDict({})
            for k in model:
                cell[k] = make_model(model[k])
            return cell
    elif isinstance(model, list):
        cell = nn.ModuleList([])
        for i in range(len(model)):
            cell.append(make_model(model[i]))
        return cell
    elif isinstance(model, tuple):
        container = []
        for i in range(len(model)):
            container.append(make_model(model[i]))
        cell = nn.Sequential(*container)
        return cell
    else:
        raise ValueError('Not valid model format')
    return


def normalize(input):
    broadcast_size = [1] * input.dim()
    broadcast_size[1] = input.size(1)
    m = config.PARAM['stats'].mean.view(broadcast_size).to(input.device)
    s = config.PARAM['stats'].std.view(broadcast_size).to(input.device)
    input = input.sub(m).div(s)
    return input


def denormalize(input):
    broadcast_size = [1] * input.dim()
    broadcast_size[1] = input.size(1)
    m = config.PARAM['stats'].mean.view(broadcast_size).to(input.device)
    s = config.PARAM['stats'].std.view(broadcast_size).to(input.device)
    input = input.mul(s).add(m)
    return input


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
            var = (Z ** 2).t().matmul(C) / nk - 2 * mu * Z.t().matmul(C) / nk + mu ** 2
        else:
            if config.PARAM['init_param_mode'] == 'none':
                C = torch.distributions.categorical.Categorical(
                    Z.new_ones(config.PARAM['classes_size']) / config.PARAM['classes_size'])
                C = idx2onehot(C.sample((Z.size(0),)))
                mu = model.param['mu'].data
                var = model.param['var'].data
            elif config.PARAM['init_param_mode'] == 'random':
                C = torch.distributions.categorical.Categorical(
                    Z.new_ones(config.PARAM['classes_size']) / config.PARAM['classes_size'])
                C = idx2onehot(C.sample((Z.size(0),)))
                nk = C.sum(dim=0, keepdim=True) + 1e-10
                mu = Z.t().matmul(C) / nk
                var = (Z ** 2).t().matmul(C) / nk - 2 * mu * Z.t().matmul(C) / nk + mu ** 2
            elif config.PARAM['init_param_mode'] == 'kmeans':
                from sklearn.cluster import KMeans
                km = KMeans(n_clusters=config.PARAM['classes_size'], n_init=1).fit(Z.cpu().numpy())
                C = idx2onehot(torch.tensor(km.labels_, dtype=torch.long))
                nk = C.sum(dim=0, keepdim=True) + 1e-10
                mu = Z.t().matmul(C) / nk
                var = (Z ** 2).t().matmul(C) / nk - 2 * mu * Z.t().matmul(C) / nk + mu ** 2
            elif config.PARAM['init_param_mode'] == 'gmm':
                from sklearn.mixture import GaussianMixture
                gm = GaussianMixture(n_components=config.PARAM['classes_size'], covariance_type='diag').fit(
                    Z.cpu().numpy())
                C = idx2onehot(
                    torch.tensor(gm.predict(Z.cpu().numpy()), dtype=torch.long, device=config.PARAM['device']))
                mu = torch.tensor(gm.means_.T, dtype=torch.float, device=config.PARAM['device'])
                var = torch.tensor(gm.covariances_.T, dtype=torch.float, device=config.PARAM['device'])
            else:
                raise ValueError('Not valid init param')
        model.param['mu'].copy_(mu)
        model.param['var'].copy_(var)
        return C