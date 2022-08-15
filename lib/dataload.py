import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from natsort import natsort
import imageio
import glob, sys, os
from .toy_data import *


#--------------------  Data Pair(P & Q) Builder & Sampler  --------------------#

def build_pair(args, train=True):
    assert args.train_type in ['generation', 'translation']
    if args.train_type == 'translation':
        assert args.q_data != 'noise'

    batch_size = args.train_batch_size * args.cache_num if train else args.test_batch_size
    kwargs = {"num_workers": 2, "pin_memory": True, "drop_last": True}

    if args.data_type == 'toy':
        p_data = load_toy(args.p_data, batch_size)
        if args.q_data != 'noise':
            q_data = load_toy(args.q_data, batch_size)
        elif args.q_data == 'noise':
            q_data = load_noise(batch_size)
    
    if args.data_type == 'mnist':
        assert args.q_data == 'noise'
        p_data = sample_data(DataLoader(load_mnist(args, train), batch_size, shuffle=True, **kwargs))
        q_data = sample_data(DataLoader(load_latent(next(p_data)), batch_size, shuffle=True, **kwargs))

    if args.data_type == 'emnist':
        p_data = sample_data(DataLoader(load_emnist(args, train), batch_size, shuffle=True, **kwargs))
        if args.q_data == 'mnist':
            q_data = sample_data(DataLoader(load_mnist(args, train), batch_size, shuffle=True, **kwargs))
        elif args.q_data == 'noise':
            q_data = sample_data(DataLoader(load_latent(next(p_data)), batch_size, shuffle=True, **kwargs))

    elif args.data_type == 'cifar10':
        assert args.q_data == 'noise'
        p_data = sample_data(DataLoader(load_cifar10(args, train), batch_size, shuffle=True, **kwargs))
        q_data = sample_data(DataLoader(load_latent(next(p_data)), batch_size, shuffle=True, **kwargs))

    if args.data_type == 'custom':
        transform = A.Compose([A.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)), 
                               A.Resize(args.img_size, args.img_size), 
                               A.HorizontalFlip(), ToTensorV2()])

        p_data = Load_Dataset(args, args.p_path, train=train, transform=transform)
        p_data = sample_data(DataLoader(p_data, batch_size, shuffle=True, **kwargs))

        if args.q_data != 'noise':
            if args.q_data == 'lr':
                transform_lr = A.Compose([A.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)), 
                               A.Resize(args.img_size // args.lr_scale, args.img_size // args.lr_scale),
                               A.Resize(args.img_size, args.img_size),
                               A.HorizontalFlip(), ToTensorV2()])
                q_data = Load_Dataset(args, args.p_path, train=train, transform=transform_lr)
            else:
                q_data = Load_Dataset(args, args.q_path, train=train, transform=transform)
            q_data = sample_data(DataLoader(q_data, batch_size, shuffle=True, **kwargs))
        
        elif args.q_data == 'noise':
            q_data = sample_data(DataLoader(load_latent(next(p_data)), batch_size, shuffle=True, **kwargs))

    return p_data, q_data


def sample_data(dataloader, is_not_cacheloader=True):
    while True:
        for batch in dataloader:
            # For dataloader with label: MNIST, CIFAR10
            if isinstance(batch, list) and is_not_cacheloader:
                yield batch[0]
            else:
                yield batch


class load_latent(Dataset):
    def __init__(self, sample_data):
        super(load_latent, self).__init__()
        self.data_shape = np.array(sample_data).shape[1:]
    def __len__(self):
        return  10000
    def __getitem__(self, item):
        return  torch.randn(*self.data_shape)

def load_mnist(args, train):
    transform = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(), 
                                    transforms.Normalize((0.5,), (0.5,))])
    data = datasets.MNIST('/home/sk851/data', transform=transform, train=train, download=True)
    return data

def load_emnist(args, train):
    transform = transforms.Compose([transforms.Resize(args.img_size), 
                                    lambda img: transforms.functional.rotate(img, -90),
                                    lambda img: transforms.functional.hflip(img),
                                    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    data = datasets.EMNIST('/home/sk851/data', split='letters', transform=transform, train=train, download=True)
    return data

def load_cifar10(args, train):
    transform = transforms.Compose([transforms.Resize(args.img_size), transforms.ToTensor(), 
                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    data = datasets.CIFAR10('/home/sk851/data', transform=transform, train=train, download=True)
    return data


#--------------------  Utils for Custom Data Load  --------------------#

def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))

class Load_Dataset(Dataset):
    def __init__(self, args, path_info, train=True, transform=None):
        super(Load_Dataset, self).__init__()
        self.args = args
        path = path_info.train if train else path_info.test
        self.paths = fiFindByWildcard(os.path.join(path, '*.*'))
        self.transform = transform

    def __len__(self):
        return  len(self.paths)

    def __getitem__(self, item):
        img = cv2.imread(self.paths[item])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']

        return  img