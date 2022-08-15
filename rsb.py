import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os, gc, math
import numpy as np
from lib.utils import *
from sampling import *
from lib.dataload import *


class RSB():
    def __init__(self, args):
        super(RSB, self).__init__()
        self.args = args
        self.device  = torch.device(f'cuda:{args.gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
        if torch.cuda.is_available(): torch.cuda.set_device(args.gpu_id)

        if not os.path.exists("./weights"):
            os.makedirs("./weights")

        # -------- build boundary distributions: p(data) & q(prior for generation) --------
        """ For image data, assume value range is [-1,1] """
        self.p_data, self.q_data = build_pair(args) # For train
        self.p_data_test, self.q_data_test = build_pair(args, train=False) # For test
        self.plotter = ImPlotter(args)

        # -------- build models: forward & backward --------
        self.f = build_model(args, 'forward').to(self.device) # p -> q
        self.b = build_model(args, 'backward').to(self.device) # q -> p
        
        self.ts = torch.linspace(0., 1., args.num_steps).to(self.device)
        # print(f"Timesteps:{['{:.3f}'.format(i) for i in self.ts.reshape(-1)]}")
        self.max_t = self.ts[-1].to(self.device)
        self.it_f, self.it_b, self.it_d_p, self.it_d_q = 0, 0, 0, 0

        self.gammas = gamma_sch(args)
        self.traj = Traj(args, self.p_data, self.q_data, self.ts, self.gammas, self.device)
        self.test_traj = Traj(args, self.p_data_test, self.q_data_test, self.ts, self.gammas, self.device)
        # print(f"Gamma Schedule:{['{:.3f}'.format(i) for i in self.gammas.reshape(-1)]}")

        if args.load:
            self.load_weight()


    # ---------------- Main Function for Train ---------------- #
    def train(self):
        self.writer = SummaryWriter()
        
        # -------- optimizer and scheduler for SB --------
        self.optim_f, self.ema_f, self.sch_f = build_optim(self.args, self.f)
        self.optim_b, self.ema_b, self.sch_b = build_optim(self.args, self.b)

        for n in tqdm(range(self.args.n_ipf)) if self.args.data_type == 'toy' else range(self.args.n_ipf):
            if n == 0: print("# -------- SB training started -------- #")
            self.ipf_step(n, update_direction='backward')
            self.ipf_step(n, update_direction='forward')


    def ipf_step(self, n, update_direction):
        update_model, sample_model = {
            'forward':  [self.f, self.b], # train forward, sample from backward
            'backward': [self.b, self.f], # train backward, sample from forward
        }.get(update_direction)

        optim_sb, ema_sb, sch_sb = self.get_optim(update_model)
        _, ema_sample, _ = self.get_optim(sample_model)

        if self.args.data_type != 'toy': print(f"{n + 1}th {update_direction} ipf update: ")

        if self.args.train_type == 'generation' and update_direction == 'forward':
            num_itr = int(self.args.num_itr * 0.5)
        else:
            num_itr = self.args.num_itr

        for _ in tqdm(range(self.args.sample_itr)) if self.args.data_type != 'toy' else range(self.args.sample_itr):
            # -------- prepare data for RSB --------
            with ema_sample.average_parameters():
                xs_sb, out_sb, x_init_sb = self.traj.sample(sample_model, n)

            ts_sb = self.ts.detach().cpu()
            train_xs = xs_sb.cpu()  # [batch_size, num_steps, x_dim]
            train_out = out_sb.cpu()  # [batch_size, num_steps, x_dim]

            # -------- empty cache for memory --------
            del xs_sb, out_sb
            gc.collect()
            torch.cuda.empty_cache()
            sb_loader = self.cacheloader(train_xs, train_out, ts_sb)
        
            for i in range(num_itr):
                # -------- update SB --------
                optim_sb.zero_grad()
                
                x, out, t = next(sb_loader)
                eval_t = self.max_t - t.to(self.device)

                pred = update_model(x.to(self.device), eval_t)
                sb_loss = F.mse_loss(pred, out.to(self.device))
                sb_loss.backward()

                torch.nn.utils.clip_grad_norm_(update_model.parameters(), 1.)
                optim_sb.step()
                ema_sb.update()
                sch_sb.step()

                self.log_loss(update_direction, sb_loss)
    
            del sb_loader, sb_loss
            gc.collect()
            torch.cuda.empty_cache()

        if (n + 1) % self.args.verbose_ipf == 0:
            self.test_plot(update_direction, n)
            self.save_weight(n + 1)
    

    # ---------------- Main Function for Test ---------------- #
    @torch.no_grad()
    def test(self):
        self.load_weight()
        test_plotter = ImPlotter(self.args, im_dir = './test')
        self.test_plot('backward', test_plotter=test_plotter)
        if self.args.train_type == 'generation':
            pass
        else:
            self.test_plot('forward', test_plotter=test_plotter)

    @torch.no_grad()
    def test_plot(self, sample_direction, n=None, train_type=None, test_plotter=None):
        model = self.b if sample_direction == 'backward' else self.f
        xs, x_init = self.test_traj.sample(model, sample=True)
        xs = torch.cat([x_init[:,None,:].cpu(), xs.cpu()], dim=1)
        
        if test_plotter is None:
            self.plotter(xs.cpu(), fb = sample_direction, i = n + 1)
        else:
            test_plotter(xs.cpu(), fb = sample_direction)


    # ---------------- Utils ---------------- #
    def get_optim(self, model):
        if model.direction == 'forward':
            return self.optim_f, self.ema_f, self.sch_f
        elif model.direction == 'backward':
            return self.optim_b, self.ema_b, self.sch_b

    def cacheloader(self, xs, out, ts):
        new_dl = CacheLoader(self.args, xs, out, ts)
        new_dl = DataLoader(new_dl, batch_size=self.args.train_batch_size)
        return sample_data(new_dl, is_not_cacheloader=False)

    # -------- for logging --------
    def log_loss(self, update_direction, sb_loss):
        it = self.it_f if update_direction == 'forward' else self.it_b
        self.writer.add_scalar(f"SB_Loss_{update_direction}", sb_loss.item(), it+1)

        if update_direction == 'forward':
            self.it_f += 1
        elif update_direction == 'backward':
            self.it_b += 1

    # -------- for save & load --------
    def load_weight(self):
        self.f.load_state_dict(torch.load(self.args.path.f_path))
        self.b.load_state_dict(torch.load(self.args.path.b_path))
        print("Model loaded successfully")

    def save_weight(self, n):
        path = f"./weights/iter_{n}"
        if not os.path.exists(path): os.makedirs(path)
        torch.save(self.f.state_dict(), path + "/f.pt")
        torch.save(self.b.state_dict(), path + "/b.pt")