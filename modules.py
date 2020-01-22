import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random

from functools import reduce

### Layers

def _xavier_override(tensor, fan_in, fan_out, gain=1.0):
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    nn.init._no_grad_uniform_(tensor, -a, a)

def createLinear(n_in, n_out):
    """ Create an nn.Linear instance with its weights initialized by Xavier
    """
    linear = nn.Linear(n_in, n_out)
    nn.init.xavier_uniform_(linear.weight)
    _xavier_override(linear.bias, linear.bias.shape[0], linear.bias.shape[0])
    return linear

def createLinearLayers(n_input, n_hidden, n_layers, last_layer_dim, activation=nn.Softplus):
    layers = [createLinear(n_input, n_hidden)]
    for _ in range(n_layers-2):
        layers.append(activation())
        layers.append(createLinear(n_hidden, n_hidden))
    layers.append(createLinear(n_hidden, last_layer_dim))
    return layers

def weights_init(m, sd):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, sd)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, sd)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        _xavier_override(m.bias, m.bias.shape[0], m.bias.shape[0])

class ReShape(nn.Module):
    def __init__(self, shape):
        super(ReShape, self).__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(x.size(0), *self.shape)

### Modules

class BasicNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.layers_pt = nn.ModuleList(self.layers)
        print(f'Created {type(self).__name__} with {self.number_of_params()} params')
        device = torch.device(device)
        self.to(device)
        self.device = device

    def number_of_params(self):
        return reduce(lambda accu, param: accu + np.prod(param.size()), self.parameters(), 0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def random(self, num_samples):
        with torch.no_grad():
            return self.random_with_grad(num_samples)

    def random_with_grad(self, num_samples):
        gen_input = torch.zeros(num_samples, self.n_input, device=self.device)
        gen_input.normal_(0, 1)
        return self(gen_input)

class MlpModule(BasicNet):
    def __init__(self, n_input, n_hidden, n_layers, last_layer_dim, activation, residual, device):
        self.n_input = n_input
        self.n_layers = n_layers
        self.residual = residual
        self.layers = createLinearLayers(n_input, n_hidden, n_layers, last_layer_dim, activation=activation)
        super().__init__(device)
    def forward(self, x):
        y = super().forward(x.view(len(x), -1))
        if self.residual:
            return y + x
        else:
            return y

class DenseNetModule(BasicNet):
    def __init__(self, n_input, n_hidden, n_layers, last_layer_dim, activation, end_activation, device):
        self.n_input = n_input
        self.n_layers = n_layers

        dense_layers = []
        for i in range(n_layers):
            dense_layers.append(createLinear(n_input, n_hidden))
            dense_layers.append(activation())
            n_input += n_hidden
        
        addl_layers = [createLinear(n_input, last_layer_dim)]
        if not end_activation is None:
            addl_layers.append(end_activation())

        self.dense_layers = dense_layers
        self.addl_layers = addl_layers
        self.layers = [*dense_layers, *addl_layers]

        super().__init__(device)
    def forward(self, x):
        x = x.view(len(x), -1)
        for i in range(self.n_layers):
            y = self.dense_layers[i*2](x)
            y = self.dense_layers[i*2+1](y)
            x = torch.cat([y, x], dim=1)
        for layer in self.addl_layers:
            x = layer(x)
        return x

class Generator(BasicNet):
    def __init__(self, n_input, n_feat, output_channels, batch_norm, device):
        self.n_input = n_input
        output_channels = 3
        activation = nn.ReLU

        if batch_norm:
            layers = [
                nn.Linear(n_input, n_feat*4*4*4),
                activation(),

                nn.Linear(n_feat*4*4*4, n_feat*2*7*7),
                ReShape([n_feat*2, 7, 7]),
                nn.BatchNorm2d(n_feat*2),
                activation(),
                
                nn.ConvTranspose2d(n_feat*2, n_feat*2, 4, 2, 1),
                nn.BatchNorm2d(n_feat*2),
                activation(),

                nn.ConvTranspose2d(n_feat*2, output_channels, 4, 2, 1),
                nn.Sigmoid()
            ]
        else:
            layers = [
                nn.Linear(n_input, n_feat*4*4*4),
                activation(),

                nn.Linear(n_feat*4*4*4, n_feat*2*7*7),
                ReShape([n_feat*2, 7, 7]),
                
                nn.ConvTranspose2d(n_feat*2, n_feat*2, 4, 2, 1),
                activation(),

                nn.ConvTranspose2d(n_feat*2, output_channels, 4, 2, 1),
                nn.Sigmoid()
            ]

        for layer in layers:
            weights_init(layer, 0.02)
        self.layers = layers
        super().__init__(device)

    def random_with_grad(self, num_samples):
        gen_input = torch.randn(num_samples, self.n_input, device=self.device)
        return self(gen_input)

class Discriminator(BasicNet):
    def __init__(self, n_feat, input_channels, activation, end_activation, batch_norm, device):
        if batch_norm:
            layers = [
                nn.Conv2d(input_channels, n_feat, 4, 2, 1),
                activation(),

                nn.Conv2d(n_feat, n_feat*2, 4, 2, 1),
                activation(),
                nn.BatchNorm2d(n_feat*2),

                nn.Conv2d(n_feat*2, n_feat*4, 4, 2, 2),
                activation(),
                nn.BatchNorm2d(n_feat*4),

                nn.Conv2d(n_feat*4, 1, 4, 1, 0)
            ]
        else:
            layers = [
                nn.Conv2d(input_channels, n_feat, 4, 2, 1),
                activation(),

                nn.Conv2d(n_feat, n_feat*2, 4, 2, 1),
                activation(),

                nn.Conv2d(n_feat*2, n_feat*4, 4, 2, 2),
                activation(),

                nn.Conv2d(n_feat*4, 1, 4, 1, 0)
            ]
        if not end_activation is None:
            layers.append(end_activation())

        for layer in layers:
            weights_init(layer, 0.07)
        self.layers = layers
        super().__init__(device)

class NCSNDde(BasicNet):
    def __init__(self, n_feat, activation, device):
        layers = [
            nn.Conv2d(4, n_feat, 4, 2, 1),
            activation(),

            nn.Conv2d(n_feat, n_feat*2, 4, 2, 1),
            activation(),

            nn.Conv2d(n_feat*2, n_feat*4, 4, 2, 2),
            activation(),

            nn.Conv2d(n_feat*4, 10, 4, 1, 0)
        ]
        
        for layer in layers:
            weights_init(layer, 0.07)
        self.layers = layers
        super().__init__(device)
    
    def forward(self, x, sigma, sigma_idx):
        c = torch.ones(len(x), 1, 28, 28, dtype=x.dtype).to(self.device) * sigma
        x = torch.cat((x, c), dim=1)
        for layer in self.layers:
            x = layer(x)
        return x[:, sigma_idx]
