import config
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import make_model, idx2onehot


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
    return CE


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
            0.5 * (math.log(2 * math.pi) + self.param['logvar']) + (
                        output['z'].unsqueeze(-1) - self.param['mu']) ** 2 / (2 * torch.exp(self.param['logvar'])),
            dim=1)) + 1e-10
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