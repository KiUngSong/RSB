import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import abc


#--------------------  Class for Dataloder of Sampled Trajectory  --------------------#

def flatten_dim01(x):
    # [batch_size, num_steps, x_dim] -> [batch_size*num_steps, x_dim]
    return x.reshape(-1, *x.shape[2:])

class CacheLoader(Dataset):
    def __init__(self, args, xs, out, ts):
        super().__init__()
        self.args = args
        self.xs = flatten_dim01(xs)
        self.out = flatten_dim01(out)
        self.ts = ts.repeat(args.train_batch_size * args.cache_num)

        assert self.xs.shape[0] == (args.train_batch_size * args.cache_num * len(ts))
        assert self.ts.shape[0] == (args.train_batch_size * args.cache_num * len(ts))
        assert self.xs.shape == self.out.shape

    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, idx):
        x = self.xs[idx]
        out = self.out[idx]
        t = self.ts[idx]
        return x, out, t


#--------------------  Class for Trajectory Sampling  --------------------#

class Traj(nn.Module):
    def __init__(self, args, p, q, ts, gammas, device):
        super().__init__()
        self.args = args
        
        self.alpha = self.args.alpha
        self.beta = self.args.beta
        self.gammas = gammas

        self.p = p # data distribution
        self.q = q # prior distribution for generation or another data distribution

        self.ts = ts
        assert (ts[1:] > ts[:-1]).all(), 'time step must be strictly increasing'

        self.num_steps = args.num_steps  # num diffusion steps
        self.device = device

    @torch.no_grad()
    def sample(self, model, n=None, sample=False):
        sample_direction = model.direction
        assert sample_direction in ['forward','backward']

        init_dist = self.p if sample_direction=='forward' else self.q
        x = torch.Tensor(next(init_dist)).to(self.device) # [batch_size, x_dim]

        xs = torch.empty((x.shape[0], len(self.ts), *x.shape[1:])) # [batch_size, num_steps, x_dim]
        out = torch.empty_like(xs) # [batch_size, num_steps, x_dim]
        x_init = x

        gammas = self.gammas if sample_direction=='forward' else np.flip(self.gammas)

        for idx, t in enumerate(self.ts):
            if n == 0 and sample_direction=='forward' and not self.args.load:
                a = (idx + 1) / len(self.ts)
                t_old = np.sqrt(1 - a) * x + np.sqrt(a) * torch.randn_like(x)
            else:
                t_old = model(x, t)
            x_old = x.cpu()

            if sample & (idx == self.args.num_steps - 1):
                x = t_old
            else:
                x = t_old + np.sqrt(gammas[idx].item()) * torch.randn_like(x)

            if sample:
                pass
            else:
                if n == 0 and sample_direction=='forward' and not self.args.load:
                    t_new = np.sqrt(1 - a) * x + np.sqrt(a) * torch.randn_like(x)
                else:
                    t_new = model(x, t)

                    if idx != self.args.num_steps - 1 and self.args.alpha != 0:
                        t_new = (1 - self.alpha) * t_new + self.alpha * model(x, self.ts[idx + 1])

                target = (t_old - t_new).cpu() + x.cpu()
                target = target / (1 + self.beta) + self.beta * x_old / (1 + self.beta)
                out[:,idx,...] = target
            
            xs[:,idx,...] = x.cpu()

        if sample:
            return xs, x_init
        else:
            return xs, out, x_init