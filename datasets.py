import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import math
import random
import matplotlib.pyplot as pp

import torchvision.datasets as ds
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose

def _sample_2d_data(dataset, n_samples):

    z = torch.randn(n_samples, 2)

    if dataset == '8gaussians':
        scale = 4
        sq2 = 1/math.sqrt(2)
        centers = [(1,0), (-1,0), (0,1), (0,-1), (sq2,sq2), (-sq2,sq2), (sq2,-sq2), (-sq2,-sq2)]
        centers = torch.tensor([(scale * x, scale * y) for x,y in centers])
        return sq2 * (0.5 * z + centers[torch.randint(len(centers), size=(n_samples,))])

    elif dataset == '2spirals':
        n = torch.sqrt(torch.rand(n_samples // 2)) * 540 * (2 * math.pi) / 360
        d1x = - torch.cos(n) * n + torch.rand(n_samples // 2) * 0.5
        d1y =   torch.sin(n) * n + torch.rand(n_samples // 2) * 0.5
        x = torch.cat([torch.stack([ d1x,  d1y], dim=1),
                       torch.stack([-d1x, -d1y], dim=1)], dim=0) / 3
        return x + 0.1*z

    elif dataset == 'checkerboard':
        x1 = torch.rand(n_samples) * 4 - 2
        x2_ = torch.rand(n_samples) - torch.randint(0, 2, (n_samples,), dtype=torch.float) * 2
        x2 = x2_ + x1.floor() % 2
        return torch.stack([x1, x2], dim=1) * 2

    elif dataset == 'rings':
        n_samples4 = n_samples3 = n_samples2 = n_samples // 4
        n_samples1 = n_samples - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, set endpoint=False in np; here shifted by one
        linspace4 = torch.linspace(0, 2 * math.pi, n_samples4 + 1)[:-1]
        linspace3 = torch.linspace(0, 2 * math.pi, n_samples3 + 1)[:-1]
        linspace2 = torch.linspace(0, 2 * math.pi, n_samples2 + 1)[:-1]
        linspace1 = torch.linspace(0, 2 * math.pi, n_samples1 + 1)[:-1]

        circ4_x = torch.cos(linspace4)
        circ4_y = torch.sin(linspace4)
        circ3_x = torch.cos(linspace4) * 0.75
        circ3_y = torch.sin(linspace3) * 0.75
        circ2_x = torch.cos(linspace2) * 0.5
        circ2_y = torch.sin(linspace2) * 0.5
        circ1_x = torch.cos(linspace1) * 0.25
        circ1_y = torch.sin(linspace1) * 0.25

        x = torch.stack([torch.cat([circ4_x, circ3_x, circ2_x, circ1_x]),
                         torch.cat([circ4_y, circ3_y, circ2_y, circ1_y])], dim=1) * 3.0

        # random sample
        x = x[torch.randint(0, n_samples, size=(n_samples,))]

        # Add noise
        return x + torch.normal(mean=torch.zeros_like(x), std=0.08*torch.ones_like(x))

    else:
        raise RuntimeError('Invalid `dataset` to sample from.')

def _loadAllData(dataset, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    sample = dataset[0][0]
    tensor = torch.zeros(len(dataset), *sample.shape, dtype=sample.dtype)
    idx = 0
    for _, batch_data in enumerate(loader):
        tensor[idx:idx+batch_size] = batch_data[0]
        idx += len(batch_data[0])
    return tensor

class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, name, data_size, transform=None, dynamic=True):
        self.name = name
        self.data_size = data_size
        self.transform = transform if not transform is None else lambda x:x
        self.dynamic = dynamic
        if not dynamic:
            self.data = _sample_2d_data(name, data_size)
            self.data = self.transform(self.data)
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, i):
        if self.dynamic:
            return tuple([self.transform(_sample_2d_data(self.name, 2)[random.randint(0,1)])])
        else:
            return tuple([self.data[i]])

class StackedMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_size, train, transform, download):
        self.mnist = _loadAllData(ds.MNIST(root, train=train, transform=transform, download=download), 60000)
        self.data_size = data_size
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, i):
        indices = np.random.randint(0, len(self.mnist), size=[3])
        return (self.mnist[indices].squeeze(), 0)

def _createLoader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)

def _getTransform(transform):
    if transform is None:
        return ToTensor()
    else:
        return Compose([
            ToTensor(),
            transform
        ])

def toyLoader(name, length, batch_size, transform, shuffle, dynamic):
    dataset = ToyDataset(name, length, transform=transform, dynamic=dynamic)
    return _createLoader(dataset, batch_size, shuffle)

def eightGaussiansLoader(length, batch_size, transform=None, shuffle=True, dynamic=True):
    return toyLoader('8gaussians', length, batch_size, transform, shuffle, dynamic)

def twoSpiralsLoader(length, batch_size, transform=None, shuffle=True, dynamic=True):
    return toyLoader('2spirals', length, batch_size, transform, shuffle, dynamic)

def checkerboardLoader(length, batch_size, transform=None, shuffle=True, dynamic=True):
    return toyLoader('checkerboard', length, batch_size, transform, shuffle, dynamic)

def ringsLoader(length, batch_size, transform=None, shuffle=True, dynamic=True):
    return toyLoader('rings', length, batch_size, transform, shuffle, dynamic)

def mnistLoader(root, download, batch_size, train=True, transform=None, shuffle=True):
    dataset = ds.MNIST(root, train=train, transform=_getTransform(transform), download=download)
    return _createLoader(dataset, batch_size, shuffle)

def fashionLoader(root, download, batch_size, train=True, transform=None, shuffle=True):
    dataset = ds.FashionMNIST(root, train=train, transform=_getTransform(transform), download=download)
    return _createLoader(dataset, batch_size, shuffle)

def stackedMnistLoader(root, download, batch_size, train=True, transform=None, shuffle=True, data_size=None):
    if data_size is None:
        data_size = batch_size * 100
    dataset = StackedMNISTDataset(root, data_size, train, _getTransform(transform), download)
    return _createLoader(dataset, batch_size, shuffle)

def uciLoader(root, name, batch_size):
    data = np.load(f"{root}/{name}.npz")
    train = np.concatenate([data["train"], data["val"]], axis=0)
    train_set = torch.utils.data.TensorDataset(torch.from_numpy(train))
    test = data["test"]
    test_set = torch.utils.data.TensorDataset(torch.from_numpy(test))
    return _createLoader(train_set, batch_size, True), _createLoader(test_set, batch_size, False)
