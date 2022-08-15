import torch
import torchvision.utils as vutils
from torch_ema import ExponentialMovingAverage
import numpy as np
import os, math
import matplotlib.pyplot as plt
from PIL import Image
from .model.toy.models import ToyNet
from .model.unet_pp.models import NCSNpp
from .toy_data import *


#--------------------  Model Builder  --------------------#

def build_model(args, direction=None):
    assert args.data_type in ['toy', 'mnist', 'emnist', 'cifar10', 'custom']

    if args.data_type == 'toy':
        model = ToyNet(args, direction)
    
    elif args.data_type in ['mnist', 'emnist', 'cifar10']:
        model = NCSNpp(args, direction)

    elif args.data_type == 'custom':
        model = NCSNpp(args, direction)

    print(f"# of {direction} model parameters : {sum(p.numel() for p in model.parameters())}")
    
    return model


#--------------------  Optimizer Builder  --------------------#

def build_optim(args, model):
    lr, lr_min = args.sb_lr, args.sb_lr_min
    
    total_itr = args.n_ipf * args.sample_itr * args.num_itr
    if model.direction == "forward" and args.train_type == 'generation':
        total_itr = int(0.5 * total_itr)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=0)
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_itr, eta_min=lr_min)
    
    return optimizer, ema, scheduler


#--------------------  Gamma Schedule Builder  --------------------#

def gamma_sch(args):
    if args.train_type == 'generation':
        gammas = np.linspace(args.gamma['min'], args.gamma['max'], args.num_steps)
    else:
        gammas = np.linspace(args.gamma['min'], args.gamma['max'], args.num_steps // 2)
        gammas = np.concatenate([gammas, np.flip(gammas)])
    assert len(gammas) == args.num_steps

    return gammas


#--------------------  Utils for Visual Plotting  --------------------#

class ImPlotter(object):
    def __init__(self, args, im_dir='./imgs'):
        self.args = args
        if not os.path.exists(im_dir):
            os.mkdir(im_dir)
        self.im_dir = im_dir

    def save(self, img, path):
        if self.args.data_type == 'toy':
            plot_toy(img, path)
        else:
            vutils.save_image(img, path, normalize=True, range=(-1,1), nrow=int(math.sqrt(img.size(0))))

    def make_gif(self, plot_paths, im_dir, fb):
        frames = [Image.open(fn) for fn in plot_paths]
        frames[0].save(os.path.join(im_dir, f"{fb}.gif"), format='GIF', append_images=frames[1:], 
                       save_all=True, duration=125, loop=0)

    def plot(self, total_samples, fb, i=None):
        # Shape of total samples: [batch_size, num_steps, x_dim]
        name = f"{fb[0]}_{i}" if i is not None else f"{fb[0]}"
        im_dir = os.path.join(self.im_dir, name)
        if not os.path.exists(im_dir): os.makedirs(im_dir)

        num_steps = total_samples.shape[1]
        plot_paths = []

        plt.clf()
        filename_grid_png = os.path.join(im_dir, 'img_first.png')
        plot_paths.append(filename_grid_png)
        self.save(total_samples[:,0], filename_grid_png)

        for k in range(1, num_steps-1):
            filename_grid_png = os.path.join(im_dir, f'img_{k}.png')
            plot_paths.append(filename_grid_png)
            self.save(total_samples[:,k], filename_grid_png)

        filename_grid_png = os.path.join(im_dir, 'img_final.png')
        plot_paths.append(filename_grid_png)
        self.save(total_samples[:,-1], filename_grid_png)

        self.make_gif(plot_paths, im_dir, fb)

    def __call__(self, total_samples, fb, i=None):
        self.plot(total_samples, fb, i)