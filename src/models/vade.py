import config
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import collate, to_device
from .utils import make_model


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def loss(input, output, param):
    CE = F.binary_cross_entropy(output['img'], input['img'], reduction='sum')
    q_c_z = output['label']
    q_mu = output['mu'].view(input['img'].size(0), -1, 1)
    q_logvar = output['logvar'].view(input['img'].size(0), -1, 1)
    KLD = torch.sum(q_c_z * 0.5 * torch.sum(
        (q_mu - param['mu']) ** 2 / torch.exp(param['logvar']) + torch.exp(q_logvar) / torch.exp(param['logvar']) - 1 +
        param['logvar'] - q_logvar, dim=1), dim=1)
    KLD = KLD + torch.sum(q_c_z * (torch.log(q_c_z) - F.log_softmax(param['logits'], dim=-1)),
                          dim=1)
    KLD = KLD.sum()
    return CE + KLD


def idx2onehot(idx):
    if config.PARAM['subset'] == 'label' or config.PARAM['subset'] == 'identity':
        idx = idx.view(idx.size(0), 1)
        onehot = idx.new_zeros(idx.size(0), config.PARAM['classes_size']).float()
        onehot.scatter_(1, idx, 1)
    else:
        onehot = idx.float()
    return onehot


class VADE(nn.Module):
    def __init__(self):
        super(VADE, self).__init__()
        self.model = make_model(config.PARAM['model'])
        self.param = nn.ParameterDict({
            'mu': nn.Parameter(torch.zeros(config.PARAM['latent_size'], config.PARAM['classes_size'])),
            'logvar': nn.Parameter(torch.ones(config.PARAM['latent_size'], config.PARAM['classes_size'])),
            'logits': nn.Parameter(torch.ones(config.PARAM['classes_size']) / config.PARAM['classes_size'])
        })

    def generate(self, N):
        x = torch.randn([N, config.PARAM['latent_size']], device=config.PARAM['device'])
        x = self.model['decoder_latent'](x)
        x = self.model['decoder'](x)
        generated = x.view(x.size(0), *config.PARAM['img_shape'])
        return generated

    def init_param(self, train_loader):
        with torch.no_grad():
            self.train(False)
            Z = []
            for i, input in enumerate(train_loader):
                input = collate(input)
                input = to_device(input, config.PARAM['device'])
                output = self(input)
                Z.append(output['z'])
            Z = torch.cat(Z, dim=0)
            if config.PARAM['init_param_mode'] == 'random':
                C = torch.rand(Z.size(0), config.PARAM['classes_size'], device=config.PARAM['device'])
                nk = C.sum(dim=0, keepdim=True) + 1e-10
                mu = Z.t().matmul(C) / nk
                logvar = torch.log((Z ** 2).t().matmul(C) / nk - 2 * mu * Z.t().matmul(C) / nk + mu ** 2)
                self.classifier.param['mu'].copy_(mu)
                self.classifier.param['logvar'].copy_(logvar)
            elif config.PARAM['init_param_mode'] == 'kmeans':
                from sklearn.cluster import KMeans
                C = Z.new_zeros(Z.size(0), config.PARAM['classes_size'])
                km = KMeans(n_clusters=config.PARAM['classes_size'], n_init=1,
                            random_state=config.PARAM['randomGen']).fit(Z.cpu().numpy())
                C[torch.arange(C.size(0)), torch.tensor(km.labels_).long()] = 1
                nk = C.sum(dim=0, keepdim=True) + 1e-10
                mu = Z.t().matmul(C) / nk
                logvar = torch.log(
                    (Z ** 2).t().matmul(C) / nk - 2 * self.classifier.param['mu'] * Z.t().matmul(C) / nk + mu ** 2)
                self.param['mu'].copy_(mu)
                self.param['logvar'].copy_(logvar)
            elif config.PARAM['init_param_mode'] == 'gmm':
                from sklearn.mixture import GaussianMixture
                gm = GaussianMixture(n_components=config.PARAM['classes_size'], covariance_type='diag',
                                     random_state=config.PARAM['randomGen']).fit(Z.cpu().numpy())
                mu = torch.tensor(gm.means_.T, device=config.PARAM['device']).float()
                logvar = torch.log(torch.tensor(gm.covariances_.T, device=config.PARAM['device']).float())
                self.param['mu'].copy_(mu)
                self.param['logvar'].copy_(logvar)
            else:
                raise ValueError('Not valid init param')
        return

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        x = input['img']
        x = x.view(x.size(0), -1)
        x = self.model['encoder'](x)
        output['mu'] = self.model['encoder_latent_mu'](x)
        output['logvar'] = self.model['encoder_latent_logvar'](x)
        if self.training:
            output['z'] = reparameterize(output['mu'], output['logvar'])
        else:
            output['z'] = output['mu']
        q_c_z = torch.exp(F.log_softmax(self.param['logits'], dim=-1) - torch.sum(
            0.5 * torch.log(2 * math.pi * self.param['var']) + (output['z'].unsqueeze(-1) - self.param['mu']) ** 2 / (
                    2 * self.param['var']), dim=1)) + 1e-10
        output['label'] = q_c_z / torch.sum(q_c_z, dim=1, keepdim=True)
        x = self.model['decoder_latent'](output['z'])
        decoded = self.model['decoder'](x)
        output['img'] = decoded.view(decoded.size(0), *config.PARAM['img_shape'])
        output['loss'] = loss(input, output, self.param)
        return output


def vade():
    normalization = 'none'
    activation = 'relu'
    img_shape = config.PARAM['img_shape']
    latent_size = config.PARAM['latent_size']
    hidden_size = config.PARAM['hidden_size']
    num_layers = config.PARAM['num_layers']
    config.PARAM['model'] = {}
    # Encoder
    config.PARAM['model']['encoder'] = []
    config.PARAM['model']['encoder'].append(
        {'cell': 'LinearCell', 'input_size': np.prod(img_shape).item(), 'output_size': hidden_size,
         'bias': True, 'normalization': normalization, 'activation': activation})
    for i in range(num_layers - 2):
        config.PARAM['model']['encoder'].append(
            {'cell': 'LinearCell', 'input_size': hidden_size // (2 ** i), 'output_size': hidden_size // (2 ** (i + 1)),
             'bias': True, 'normalization': normalization, 'activation': activation})
    config.PARAM['model']['encoder'] = tuple(config.PARAM['model']['encoder'])
    # latent
    config.PARAM['model']['encoder_latent_mu'] = {
        'cell': 'LinearCell', 'input_size': hidden_size // (2 ** (num_layers - 2)), 'output_size': latent_size,
        'bias': True, 'normalization': 'none', 'activation': 'none'}
    config.PARAM['model']['encoder_latent_logvar'] = {
        'cell': 'LinearCell', 'input_size': hidden_size // (2 ** (num_layers - 2)), 'output_size': latent_size,
        'bias': True, 'normalization': 'none', 'activation': 'none'}
    config.PARAM['model']['decoder_latent'] = {
        'cell': 'LinearCell', 'input_size': latent_size, 'output_size': hidden_size // (2 ** (num_layers - 2)),
        'bias': True, 'normalization': normalization, 'activation': activation}
    # Decoder
    config.PARAM['model']['decoder'] = []
    for i in range(num_layers - 2):
        config.PARAM['model']['decoder'].append(
            {'cell': 'LinearCell', 'input_size': hidden_size // (2 ** (num_layers - 2 - i)),
             'output_size': hidden_size // (2 ** (num_layers - 2 - i - 1)),
             'bias': True, 'normalization': normalization, 'activation': activation})
    config.PARAM['model']['decoder'].append(
        {'cell': 'LinearCell', 'input_size': hidden_size, 'output_size': np.prod(img_shape).item(),
         'bias': True, 'normalization': 'none', 'activation': 'sigmoid'})
    config.PARAM['model']['decoder'] = tuple(config.PARAM['model']['decoder'])
    model = VADE()
    return model