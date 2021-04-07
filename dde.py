import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import matplotlib.pyplot as pp
import os

from datasets import toyLoader, eightGaussiansLoader, twoSpiralsLoader, checkerboardLoader, ringsLoader, mnistLoader, fashionLoader, stackedMnistLoader
from modules import MlpModule, DenseNetModule, Generator, Discriminator, NCSNDde

args = None
model = None
loader = None
output_shape = None
is_toy_dataset = True

class SinActivation(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sin(x)

def activation_from_name(name):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU
    elif name == 'softplus':
        return nn.Softplus
    elif name == 'sigmoid':
        return nn.Sigmoid
    elif name == 'tanh':
        return nn.Tanh
    elif name == 'sin':
        return SinActivation
    elif name.startswith('lrelu:'):
        return lambda: nn.LeakyReLU(float(name[6:]))
    else:
        raise ValueError(f'Unknown activation {name}')

class DdeModel:
    def __init__(self, gen, dde_real, dde_gen):
        self.nets = [gen, dde_real, dde_gen]
    def save(self, path):
        gen, dde_real, dde_gen = self.nets
        cp = {}
        cp['gen'] = { 'gen': gen.state_dict() }
        cp['dde_real'] = { 'gen': dde_real.state_dict() }
        cp['dde_gen'] = { 'gen': dde_gen.state_dict() }
        torch.save({ 'model': cp }, path)
    def load(self, path):
        gen, dde_real, dde_gen = self.nets
        cp = torch.load(path)['model']
        gen.load_state_dict(cp['gen']['gen'])
        dde_real.load_state_dict(cp['dde_real']['gen'])
        dde_gen.load_state_dict(cp['dde_gen']['gen'])

class NetModel:
    def __init__(self, net):
        self.net = net
    def save(self, path):
        cp = {}
        cp['net'] = self.net.state_dict()
        torch.save({ 'model': cp }, path)
    def load(self, path):
        cp = torch.load(path)['model']
        self.net.load_state_dict(cp['net'])

### Visualization

def createFigure(size=[500,500]):
    fig = pp.figure(figsize=[s/100 for s in size],frameon=False)
    ax = pp.Axes(fig, [0,0,1,1])
    ax.set_axis_off()
    fig.add_axes(ax)
    return fig, ax

def plotDensity(ax, x, y, z, title='density', norm=None):
    # density_reduce_extreme from BNAF
    z = np.clip(z, None, z.mean() + 3 * z.std())
    limit = 4
    ax.imshow(z, extent=(-limit, limit, -limit, limit), norm=norm)
    ax.set_facecolor(pp.cm.jet(0.))
    ax.set_title(title)

def getDdeDensity(dde, size, limit):
    x = torch.linspace(0, 1, size) * limit*2 - limit
    xx, yy = torch.meshgrid((x, x))
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=1).to(dde.device)
    probs = dde(zz)
    probs = probs - probs.min()
    probs = torch.exp(probs).view(size,size).cpu().detach().numpy()
    return xx, yy, probs

def plotDdeDensity(ax, dde, size=200, limit=4, cmap=None, norm=None):   
    xx, yy, probs = getDdeDensity(dde, size, limit)
    plotDensity(ax, xx, yy, probs, norm=norm)

def plotToySamples(ax, data, limit=4, cmap=pp.cm.jet):
    upper = limit
    lower = -limit
    ax.hist2d(data[:,0], data[:,1], range=[[lower, upper],[lower, upper]], bins=int(math.sqrt(len(data))),cmap=cmap)

def visualize(path):
    if path is None:
        path = ''
    else:
        path += '_'
    path = args.vis_path + path

    print('Visualizing the model')
    if is_toy_dataset:
        fig, ax = createFigure()
        plotDdeDensity(ax, model.nets[1])
        fig.savefig(path + 'dde_real_density.png')
        pp.close(fig)

        if args.train_generator:
            fig, ax = createFigure()
            plotDdeDensity(ax, model.nets[2])
            fig.savefig(path + 'dde_gen_density.png')
            pp.close(fig)

            fig, ax = createFigure()
            samples = model.nets[0].random(args.n_toy_samples).cpu().numpy()
            plotToySamples(ax, samples)
            if not args.save_toy_samples_to is None:
                np.savez_compressed(args.save_toy_samples_to, data=samples)
            fig.savefig(path + 'dde_gen_samples.png')
            pp.close(fig)
    else:
        imgs = model.nets[0].random(10).cpu().numpy()
        fig = pp.figure(figsize=(5, 2))
        ax = fig.subplots(2, 5)
        for y in range(2):
            for x in range(5):
                img = imgs[y*5+x].reshape(output_shape)
                if len(img.shape) == 3:
                    img = np.transpose(img, [1, 2, 0])
                if img.shape[2] == 1:
                    img = np.squeeze(img)
                ax[y,x].imshow(img)
                ax[y,x].set_axis_off()
        fig.savefig(path + 'dde_gen_samples.png')
        pp.close(fig)

### Train

def _get_noise(shape, sigma, device):
    noise = torch.zeros(*shape, device=device)
    noise.normal_(0, sigma)
    return noise

def _get_noisy(x, sigma):
    return x + _get_noise(x.shape, sigma, x.device)

def _get_dde_output(sigma, dde, x):
    x.requires_grad_(True)
    prob = dde(x)
    grad = torch.autograd.grad(prob, x, grad_outputs=torch.ones(prob.shape, device=prob.device), create_graph=True)[0]
    return x + grad*sigma*sigma, prob

def train():
    print('Training the model')
    print(f'Data size is {len(loader.dataset)}')

    gen, dde_real, dde_gen = model.nets
    gen_optm = torch.optim.Adam(gen.parameters(), lr=args.lr)
    dde_real_optm = torch.optim.Adam(dde_real.parameters(), lr=args.lr)
    dde_gen_optm = torch.optim.Adam(dde_gen.parameters(), lr=args.lr)
    optms = [gen_optm, dde_real_optm, dde_gen_optm]

    if args.lr_step > 0:
        schedulers = [torch.optim.lr_scheduler.StepLR(gen_optm, args.lr_step, args.lr_step_gamma), 
                      torch.optim.lr_scheduler.StepLR(dde_real_optm, args.lr_step, args.lr_step_gamma),
                      torch.optim.lr_scheduler.StepLR(dde_gen_optm, args.lr_step, args.lr_step_gamma)]

    sigma = args.sigma
    batch_id = 0
    if args.visualize_every > 0:
        visualize('batch_0')
        
    for epoch in range(1, 1+args.epochs):
        stats = {}
        stats['kl_loss'] = 0
        stats['loss_dde_real'] = 0
        stats['loss_dde_gen'] = 0
        if args.reduce_sigma_every > 0 and (epoch % args.reduce_sigma_every == 0):
            sigma *= args.reduce_sigma_gamma
            print(f'Reduced sigma to {sigma}')
        
        for batch_i, batch_data in enumerate(loader):
            batch_id += 1

            net_input = None
            kl_loss = 0
            loss_dde_gen = 0
            loss_dde_real = 0

            for optm in optms:
                optm.zero_grad()
            
            net_gt = batch_data[0]

            # Train dde_real
            if (args.stop_training_real_dde_after < 0) or (epoch <= args.stop_training_real_dde_after):
                net_gt = net_gt.to(gen.device)
                net_input = _get_noisy(net_gt, sigma)
                dde_real_output_real, dde_real_logP_real = _get_dde_output(sigma, dde_real, net_input)
                loss_dde_real = F.mse_loss(dde_real_output_real, net_gt, reduction='sum') / 2
                loss_dde_real.backward()
                dde_real_optm.step()
                stats['loss_dde_real'] += float(loss_dde_real)

            # Train dde_gen and gen
            if args.train_generator:
                # dde_gen
                gen_output = gen.random_with_grad(len(net_gt))
                dde_gen_input = _get_noisy(gen_output, sigma)
                dde_gen_output_gen, dde_gen_logP_gen = _get_dde_output(sigma, dde_gen, dde_gen_input)
                loss_dde_gen = F.mse_loss(dde_gen_output_gen, gen_output, reduction='sum') / 2
                loss_dde_gen.backward()
                dde_gen_optm.step()
                stats['loss_dde_gen'] += float(loss_dde_gen)

                # gen
                if batch_id % args.train_gen_every == 0:
                    for optm in optms:
                        optm.zero_grad()
                    
                    gen_output = gen.random_with_grad(len(net_gt))
                    dde_gen_input = _get_noisy(gen_output, sigma)

                    dde_gen_output_gen, dde_gen_logP_gen = _get_dde_output(sigma, dde_gen, dde_gen_input)
                    dde_real_output_gen, dde_real_logP_gen = _get_dde_output(sigma, dde_real, dde_gen_input)
                    kl_loss = torch.sum(dde_gen_logP_gen - dde_real_logP_gen)
                    kl_loss.backward()

                    gen_optm.step()
                    stats['kl_loss'] += float(kl_loss)
            if args.visualize_every > 0 and batch_id % args.visualize_every == 0:
                visualize(f'batch_{batch_id}')
        # -- end batch --

        for key in stats:
            value = stats[key]
            if isinstance(value, float):
                value = f'{value:e}'
            print(f'Epoch {epoch} {key} {value}')
    # -- end epoch --
    if not args.save_to is None:
        model.save(args.save_to)

def train_ncsn():
    print('Train NCSN DDE')
    net = model.net
    optm = torch.optim.Adam(net.parameters(), lr=args.lr)

    sigmas = [np.power(0.6, i) for i in range(0, 10)]

    def get_ncsn_dde_output(x, sigma, sigma_idx):
        x.requires_grad_(True)
        prob = net(x, sigma, sigma_idx)
        grad = torch.autograd.grad(prob, x, grad_outputs=torch.ones(prob.shape, device=prob.device), create_graph=True)[0]
        return x + grad*sigma*sigma, prob

    def generate_samples(n):
        T = 100
        eps = 2e-5
        x = np.random.uniform(size=[n, 3, 28, 28]).astype(np.float32)
        for i in range(len(sigmas)):
            sig = sigmas[i]
            alpha = eps * sig * sig / (sigmas[-1] * sigmas[-1])
            for t in range(T):
                z = np.random.normal(size=[n, 3, 28, 28]).astype(np.float32)
                d_x, _ = get_ncsn_dde_output(torch.from_numpy(x).to(net.device), sig, i)
                d_x = d_x.detach().cpu().numpy()
                grad = (d_x - x) / (sig * sig)
                x += 0.5 * alpha * grad + np.sqrt(alpha) * z
        return x

    def visualize_ncsn(path):
        if path is None:
            path = ''
        else:
            path += '_'
        path = args.vis_path + path

        print('Visualizing the model')
        imgs = generate_samples(10)
        fig = pp.figure(figsize=(5, 2))
        ax = fig.subplots(2, 5)
        for y in range(2):
            for x in range(5):
                img = imgs[y*5+x].reshape(output_shape)
                if len(img.shape) == 3:
                    img = np.transpose(img, [1, 2, 0])
                if img.shape[2] == 1:
                    img = np.squeeze(img)
                ax[y,x].imshow(img)
                ax[y,x].set_axis_off()
        fig.savefig(path + 'ncsn_dde_gen_samples.png')
        pp.close(fig)

    batch_id = 0
    if args.visualize_every > 0:
        visualize_ncsn('batch_0')

    for epoch in range(1, 1+args.epochs):
        stats = {}
        stats['loss_dde_real'] = 0
        
        for batch_i, batch_data in enumerate(loader):
            batch_id += 1
            
            sigma_idx = batch_id % 10
            sigma = sigmas[sigma_idx]

            optm.zero_grad()
            
            net_gt = batch_data[0].to(net.device)
            net_input = _get_noisy(net_gt, sigma)
            dde_real_output_real, _ = get_ncsn_dde_output(net_input, sigma, sigma_idx)
            loss_dde_real = F.mse_loss(dde_real_output_real, net_gt, reduction='sum') / 2
            loss_dde_real.backward()
            optm.step()
            stats['loss_dde_real'] += float(loss_dde_real)

            if args.visualize_every > 0 and batch_id % args.visualize_every == 0:
                visualize_ncsn(f'batch_{batch_id}')
        # -- end batch --

        for key in stats:
            value = stats[key]
            if isinstance(value, float):
                value = f'{value:e}'
            print(f'Epoch {epoch} {key} {value}')
    # -- end epoch --
    if not args.save_to is None:
        model.save(args.save_to)

def run(argv):
    global args
    global model
    global loader
    global output_shape
    global is_toy_dataset

    parser = argparse.ArgumentParser()
    # training
    parser.add_argument('--train', action='store_true', default=False, help='Train a model')
    parser.add_argument('--train_generator', type=bool, default=True, help='Include generator training')
    parser.add_argument('--sigma', type=float, default=0.2, help='Noise standard deviation')
    parser.add_argument('--save_to', type=str, default='model.bin', help='Where to save the trained model')
    parser.add_argument('--batch_size', type=int, default=2048, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr_step', type=int, default=0, help='Learning rate step size')
    parser.add_argument('--lr_step_gamma', type=float, default=0.5, help='Learning rate step gamma')
    parser.add_argument('--reduce_sigma_every', type=int, default=0, help='Decrease sigma value every N epochs')
    parser.add_argument('--reduce_sigma_gamma', type=float, default=0.8, help='When decreasing sigma, sigma *= sigma_gamma')
    parser.add_argument('--stop_training_real_dde_after', type=int, default=-1, help='Only train real_dde for this number of epochs')
    parser.add_argument('--train_gen_every', type=int, default=10, help='Step generator every N iterations')
    parser.add_argument('--visualize_every', type=int, default=50, help='Visualize model every N batches')

    # loading
    parser.add_argument('--load_from', type=str, default=None, help='Where to load a trained model')
    # dataset
    parser.add_argument('--dataset', type=str, default='ts', help='Dataset (ts, eg, cb, mnist, fashion, stacked-mnist, ncsn)')
    parser.add_argument('--dataset_dir', type=str, default='/tmp/', help='Dataset location for MNIST and Fashion. Defaults to /tmp/')
    parser.add_argument('--download', action='store_true', default=False, help='Download the dataset if it does not exist')
    parser.add_argument('--datasize', type=int, default=51200, help='Datasize (pseudo, for epoch computation) for 2D datasets')
    parser.add_argument('--dynamic', action='store_true', default=True, help='Whether 2D datasets are dynamically generated')
    # model parameters
    #parser.add_argument('--type', type=str, default='mlp', help='Network architecture (mlp, densenet, or dcgan)')
    parser.add_argument('--dense_sigmoid', action='store_true', default=False, help='Add sigmoid to DenseNet generator')
    parser.add_argument('--gen_activation', type=str, default='softplus', help='Generator activation function')
    parser.add_argument('--dde_activation', type=str, default='softplus', help='Discriminator activation function')
    parser.add_argument('--n_input', type=int, default=2, help='Number of input to generator')
    parser.add_argument('--n_hidden', type=int, default=32, help='Number of hidden units in a layer')
    parser.add_argument('--n_gen_hidden', type=int, default=64, help='Number of hidden units in DenseNet generator layer')
    parser.add_argument('--n_layers', type=int, default=25, help='Number of layers')
    parser.add_argument('--n_feat_gen', type=int, default=64, help='Number of features in DCGAN generator')
    parser.add_argument('--n_feat_dsc', type=int, default=45, help='Number of features in DCGAN discriminator')
    parser.add_argument('--gen_bn', type=bool, default=True, help='Use BN in DCGAN generator')
    parser.add_argument('--dsc_activation', type=str, default='lrelu:0.2', help='Activation in DCGAN discriminator')
    parser.add_argument('--device', type=str, default='cuda:0', help='PyTorch device')
    # visualization
    parser.add_argument('--visualize', action='store_true', default=False, help='Visualize a model')
    parser.add_argument('--n_toy_samples', type=int, default=50000, help='Number of samples to visualize for toy datasets')
    parser.add_argument('--save_toy_samples_to', type=str, default=None, help='If specified, save generated toy samples to file')
    parser.add_argument('--vis_path', type=str, default='', help='Path prefix to save visualization figures. Folder must end with /')

    args = parser.parse_args(argv) if not argv is None else parser.parse_args()

    if args.dataset == 'ts':
        loader = twoSpiralsLoader(args.datasize, args.batch_size, dynamic=args.dynamic)
        args.type = 'mlp'
    elif args.dataset == 'eg':
        loader = eightGaussiansLoader(args.datasize, args.batch_size, dynamic=args.dynamic)
        args.type = 'mlp'
    elif args.dataset == 'cb':
        loader = checkerboardLoader(args.datasize, args.batch_size, dynamic=args.dynamic)
        args.type = 'mlp'
    elif args.dataset == 'mnist':
        loader = mnistLoader(args.dataset_dir, args.download, args.batch_size)
        args.type = 'densenet'
        is_toy_dataset = False
    elif args.dataset == 'fashion':
        loader = fashionLoader(args.dataset_dir, args.download, args.batch_size)
        args.type = 'densenet'
        is_toy_dataset = False
    elif args.dataset == 'stacked-mnist':
        loader = stackedMnistLoader(args.dataset_dir, args.download, args.batch_size)
        args.type = 'dcgan'
        is_toy_dataset = False
    elif args.dataset == 'ncsn':
        loader = stackedMnistLoader(args.dataset_dir, args.download, args.batch_size)
        args.type = 'ncsn'
        is_toy_dataset = False
    else:
        print(f'Unknown dataset: {args.dataset}')
        return

    output_shape = list(loader.dataset[0][0].shape)
    print(f'Output shape: {output_shape}')

    output_length = loader.dataset[0][0].view(-1).shape[0]
    print(f'Output length: {output_length}')

    gen_activation = activation_from_name(args.gen_activation)
    dde_activation = activation_from_name(args.dde_activation)
    if args.type == 'mlp':
        model = DdeModel(MlpModule(output_length, args.n_hidden, args.n_layers, output_length, gen_activation, True, args.device),
                         MlpModule(output_length, args.n_hidden, args.n_layers, 1, dde_activation, False, args.device),
                         MlpModule(output_length, args.n_hidden, args.n_layers, 1, dde_activation, False, args.device))
    elif args.type == 'densenet':
        end_activation = nn.Sigmoid if args.dense_sigmoid else None
        model = DdeModel(DenseNetModule(args.n_input, args.n_gen_hidden, args.n_layers, output_length, gen_activation, end_activation, args.device),
                         DenseNetModule(output_length, args.n_hidden, args.n_layers, 1, dde_activation, None, args.device),
                         DenseNetModule(output_length, args.n_hidden, args.n_layers, 1, dde_activation, None, args.device))
    elif args.type == 'dcgan':
        dsc_activation = activation_from_name(args.dsc_activation)
        model = DdeModel(Generator(args.n_input, args.n_feat_gen, output_shape[0], args.gen_bn, args.device),
        Discriminator(args.n_feat_dsc, output_shape[0], dsc_activation, None, False, args.device),
        Discriminator(args.n_feat_dsc, output_shape[0], dsc_activation, None, False, args.device))
    elif args.type == 'ncsn':
        model = NetModel(NCSNDde(64, gen_activation, args.device))
    else:
        print(f'Unknown model type: {args.type}')

    if not args.load_from is None:
        model.load(args.load_from)
        print(f'Loaded model from {args.load_from}')
    
    if len(args.vis_path) > 0:
        os.makedirs(args.vis_path, exist_ok=True)

    if args.train:
        if args.type == 'ncsn':
            train_ncsn()
        else:
            train()
    
    if args.visualize:
        visualize(None)

if __name__ == "__main__":
    run(None)