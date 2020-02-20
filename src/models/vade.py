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
    q_c_z = output['label'] if config.PARAM['mode'] == 'clustering' else idx2onehot(input['label'])
    q_mu = output['mu'].unsqueeze(1)
    q_logvar = output['logvar'].unsqueeze(1)
    KLD = torch.sum(q_c_z * 0.5 * torch.sum(
        (q_mu - param['mu']) ** 2 / param['var'] + torch.exp(q_logvar) / param['var'] - 1 + torch.log(
            param['var']) - q_logvar, dim=-1), dim=-1)
    KLD = KLD + torch.sum(q_c_z * (torch.log(q_c_z + 1e-10) - F.log_softmax(param['logits'], dim=-1)), dim=-1)
    KLD = KLD.sum()
    return CE + KLD


class VADE(nn.Module):
    def __init__(self):
        super(VADE, self).__init__()
        self.model = make_model(config.PARAM['model'])
        self.param = nn.ParameterDict({
            'mu': nn.Parameter(torch.randn(config.PARAM['classes_size'], config.PARAM['latent_size'])),
            'var': nn.Parameter(torch.ones(config.PARAM['classes_size'], config.PARAM['latent_size'])),
            'logits': nn.Parameter(torch.ones(config.PARAM['classes_size']) / config.PARAM['classes_size'])
        })

    def generate(self, C):
        x = torch.randn([C.size(0), config.PARAM['latent_size']], device=config.PARAM['device'])
        onehot = idx2onehot(C)
        mu, var = onehot.matmul(self.param['mu']), onehot.matmul(self.param['var'])
        x = mu + x * torch.sqrt(var)
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
            0.5 * torch.log(2 * math.pi * self.param['var']) + (output['z'].unsqueeze(1) - self.param['mu']) ** 2 / (
                    2 * self.param['var']), dim=-1)) + 1e-10
        output['label'] = q_c_z / torch.sum(q_c_z, dim=-1, keepdim=True)
        x = self.model['decoder_latent'](output['z'])
        decoded = self.model['decoder'](x)
        output['img'] = decoded.view(decoded.size(0), *config.PARAM['img_shape'])
        output['loss'] = loss(input, output, self.param)
        return output


class MCVADE(nn.Module):
    def __init__(self):
        super(MCVADE, self).__init__()
        self.model = make_model(config.PARAM['model'])
        self.param = nn.ParameterDict({
            'mu': nn.Parameter(torch.randn(config.PARAM['classes_size'], config.PARAM['latent_size'])),
            'var': nn.Parameter(torch.ones(config.PARAM['classes_size'], config.PARAM['latent_size'])),
            'logits': nn.Parameter(torch.ones(config.PARAM['classes_size']) / config.PARAM['classes_size'])
        })

    def generate(self, C):
        x = torch.randn([C.size(0), config.PARAM['latent_size']], device=config.PARAM['device'])
        config.PARAM['attr'] = idx2onehot(C)
        mu, var = config.PARAM['attr'].matmul(self.param['mu']), config.PARAM['attr'].matmul(self.param['var'])
        x = mu + x * torch.sqrt(var)
        x = self.model['decoder_latent_mc'](x)
        x = self.model['decoder_latent'](x)
        x = self.model['decoder'](x)
        generated = x.view(x.size(0), *config.PARAM['img_shape'])
        return generated

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=config.PARAM['device'], dtype=torch.float32)}
        x = input['img']
        if self.training:
            config.PARAM['attr'] = idx2onehot(input['label'])
            x = x.view(x.size(0), -1)
        else:
            config.PARAM['attr'] = None
            x = x.view(x.size(0), 1, -1)
            x = x.expand(x.size(0), config.PARAM['classes_size'], x.size(-1))
        x = self.model['encoder'](x)
        x = self.model['encoder_latent_mc'](x)
        mu = self.model['encoder_latent_mu'](x)
        logvar = self.model['encoder_latent_logvar'](x)
        if self.training:
            z = reparameterize(mu, logvar)
        else:
            z = mu
        if self.training:
            q_c_z = torch.exp(F.log_softmax(self.param['logits'], dim=-1) - torch.sum(
                0.5 * torch.log(2 * math.pi * self.param['var']) + (z.unsqueeze(1) - self.param['mu']) ** 2 / (
                        2 * self.param['var']), dim=-1)) + 1e-10
            output['label'] = q_c_z / torch.sum(q_c_z, dim=-1, keepdim=True)
        else:
            q_c_z = torch.exp(F.log_softmax(self.param['logits'], dim=-1) - torch.sum(
                0.5 * torch.log(2 * math.pi * self.param['var']) + (z.unsqueeze(2) - self.param['mu']) ** 2 / (
                        2 * self.param['var']), dim=-1)) + 1e-10
            q_c_z = q_c_z / torch.sum(q_c_z, dim=-1, keepdim=True)
            output['label'] = torch.diagonal(q_c_z, dim1=1, dim2=2)
        if self.training:
            config.PARAM['attr'] = idx2onehot(input['label'])
            output['mu'] = mu
            output['logvar'] = logvar
            output['z'] = z
        else:
            label = output['label'].topk(1, 1, True, True)[1].squeeze()
            config.PARAM['attr'] = idx2onehot(label)
            output['mu'] = mu[torch.arange(z.size(0)), label]
            output['logvar'] = logvar[torch.arange(z.size(0)), label]
            output['z'] = z[torch.arange(z.size(0)), label]
        x = self.model['decoder_latent_mc'](output['z'])
        x = self.model['decoder_latent'](x)
        decoded = self.model['decoder'](x)
        output['img'] = decoded.view(decoded.size(0), *config.PARAM['img_shape'])
        output['loss'] = loss(input, output, self.param)
        config.PARAM['attr'] = None
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


def mcvade():
    normalization = 'none'
    activation = 'relu'
    img_shape = config.PARAM['img_shape']
    latent_size = config.PARAM['latent_size']
    hidden_size = config.PARAM['hidden_size']
    num_layers = config.PARAM['num_layers']
    sharing_rate = config.PARAM['sharing_rate']
    num_mode = config.PARAM['classes_size']
    config.PARAM['model'] = {}
    # Encoder
    config.PARAM['model']['encoder'] = []
    config.PARAM['model']['encoder'].append(
        {'cell': 'LinearCell', 'input_size': np.prod(img_shape).item(), 'output_size': hidden_size,
         'bias': True, 'num_mode': num_mode, 'normalization': normalization, 'activation': activation})
    for i in range(num_layers - 2):
        config.PARAM['model']['encoder'].append(
            {'cell': 'MultimodalController', 'input_size': hidden_size // (2 ** i), 'num_mode': num_mode,
             'sharing_rate': sharing_rate})
        config.PARAM['model']['encoder'].append(
            {'cell': 'LinearCell', 'input_size': hidden_size // (2 ** i), 'output_size': hidden_size // (2 ** (i + 1)),
             'bias': True, 'num_mode': num_mode, 'normalization': normalization,
             'activation': activation})
    config.PARAM['model']['encoder'] = tuple(config.PARAM['model']['encoder'])
    # latent
    config.PARAM['model']['encoder_latent_mc'] = {
        'cell': 'MultimodalController', 'input_size': hidden_size // (2 ** (num_layers - 2)), 'num_mode': num_mode,
        'sharing_rate': sharing_rate}
    config.PARAM['model']['encoder_latent_mu'] = {
        'cell': 'LinearCell', 'input_size': hidden_size // (2 ** (num_layers - 2)), 'output_size': latent_size,
        'bias': True, 'num_mode': num_mode, 'normalization': 'none', 'activation': 'none'}
    config.PARAM['model']['encoder_latent_logvar'] = {
        'cell': 'LinearCell', 'input_size': hidden_size // (2 ** (num_layers - 2)), 'output_size': latent_size,
        'bias': True, 'num_mode': num_mode, 'normalization': 'none', 'activation': 'none'}
    config.PARAM['model']['decoder_latent_mc'] = {
        'cell': 'MultimodalController', 'input_size': latent_size, 'num_mode': num_mode, 'sharing_rate': sharing_rate}
    config.PARAM['model']['decoder_latent'] = {
        'cell': 'LinearCell', 'input_size': latent_size, 'output_size': hidden_size // (2 ** (num_layers - 2)),
        'bias': True, 'num_mode': num_mode, 'normalization': normalization, 'activation': activation}
    # Decoder
    config.PARAM['model']['decoder'] = []
    for i in range(num_layers - 2):
        config.PARAM['model']['decoder'].append(
            {'cell': 'MultimodalController', 'input_size': hidden_size // (2 ** (num_layers - 2 - i)),
             'num_mode': num_mode, 'sharing_rate': sharing_rate})
        config.PARAM['model']['decoder'].append(
            {'cell': 'LinearCell', 'input_size': hidden_size // (2 ** (num_layers - 2 - i)),
             'output_size': hidden_size // (2 ** (num_layers - 2 - i - 1)),
             'bias': True, 'num_mode': num_mode, 'normalization': normalization, 'activation': activation})
    config.PARAM['model']['decoder'].append(
        {'cell': 'MultimodalController', 'input_size': hidden_size, 'num_mode': num_mode,
         'sharing_rate': sharing_rate})
    config.PARAM['model']['decoder'].append(
        {'cell': 'LinearCell', 'input_size': hidden_size, 'output_size': np.prod(img_shape).item(),
         'bias': True, 'num_mode': num_mode, 'normalization': 'none', 'activation': 'sigmoid'})
    config.PARAM['model']['decoder'] = tuple(config.PARAM['model']['decoder'])
    model = MCVADE()
    return model