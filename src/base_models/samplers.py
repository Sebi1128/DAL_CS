import pytorch_lightning as pl
import torch.nn as nn
from scipy.special import kl_div
import numpy as np
import torch
from tqdm import tqdm
import unittest
from torch import optim


class Base_Sampler(pl.LightningModule):
    def __init__(self, cfg_smp, device):
        super(Base_Sampler, self).__init__()
        self.cfg_smp = cfg_smp
        self.dev = device
        self.batch_size = 10
        self.trainable = False

    def sample(self, active_data, acq_size, model):
        raise NotImplementedError()

    def forward(self):
        return 0.0

class TrainableSampler(Base_Sampler):
    def __init__(self, cfg_smp, device, train_every_k):
        super(TrainableSampler, self).__init__(cfg_smp, device)
        self.trainable = True
        self.train_every_k = train_every_k
        self.optimizer = None


class Random(Base_Sampler):
    def __init__(self, cfg_smp, device):
        super().__init__(cfg_smp, device)

    def sample(self, active_data, acq_size, model):
        unlbld_idx = torch.where(torch.logical_not(active_data.lbld_mask))[0]
        sample_idx = np.random.choice(len(unlbld_idx), acq_size, replace=False)
        return unlbld_idx[sample_idx]

def gaussian_kl_div(mu_p, log_var_p, mu_q, log_var_q):
    """KL(Q||P) where Q, P ~ N(mu_1:k, diag(sigma2_1:k))"""
    return 0.5*(torch.exp(-log_var_q)*(torch.exp(log_var_p) + (mu_q-mu_p)**2) - 1 + log_var_q - log_var_p).sum(1)


def gaussian_symmetric_kl_div(mu_p, log_var_p, mu_q, log_var_q):
    """KL(Q||P)-KL(P||Q) where Q, P ~ N(mu_1:k, diag(sigma2_1:k))"""
    return 0.5*(torch.exp(-log_var_q)*(torch.exp(log_var_p) + (mu_q-mu_p)**2)
                + torch.exp(-log_var_p)*(torch.exp(log_var_q) + (mu_p-mu_q)**2) -2).sum(1)


class CAL(Base_Sampler):
    def __init__(self, cfg_smp, device):
        super().__init__(cfg_smp, device)

        self.n_neighs = cfg_smp['n_neighs']

        # This can be parametrized later easily
        self.dist_func = lambda y_l, y_p: kl_div(y_l.detach(), y_p.detach()).sum(1)

        if cfg_smp['neigh_dist'] == 'l2':
            self.neigh_dist_func = lambda p, A: torch.sum((p[..., 0] - A[..., 0]) ** 2, axis=1)
        elif cfg_smp['neigh_dist'] == 'kldiv':
            self.neigh_dist_func = lambda p, A: gaussian_kl_div(p[..., 0], p[..., 1], A[..., 0], A[..., 1])
        elif cfg_smp['neigh_dist'] == 'sym_kldiv':
            self.neigh_dist_func = lambda p, A: gaussian_symmetric_kl_div(p[..., 0], p[..., 1], A[..., 0], A[..., 1])
        else:
            raise ValueError("cfg_smp.neigh_dist set to {} which is not known".format(cfg_smp['neigh_dist']))

    def sample(self, active_data, acq_size, model):


        labeled_data = active_data.get_loader('labeled', batch_size=self.batch_size)
        unlabeled_data = active_data.get_loader('unlabeled', batch_size=self.batch_size)

        z_lab = []
        p_lab = []
        z_unlab = []
        p_unlab = []

        for x, _ in labeled_data:
            x = x.to(self.dev)
            z = model.latent_full(x)
            z_lab.append(z)
            p = model.classify(x)
            p_lab.append(p)
        for x, _ in unlabeled_data:
            x = x.to(self.dev)
            z = model.latent_full(x)
            z_unlab.append(z)
            p = model.classify(x)
            p_unlab.append(p)

        z_lab = torch.cat(z_lab)
        p_lab = torch.exp(torch.cat(p_lab))
        z_unlab = torch.cat(z_unlab)
        p_unlab = torch.exp(torch.cat(p_unlab))

        score = torch.zeros((len(p_unlab)))

        # NOTE This part is very slow, 
        # we can improve it using batch etc
        # or using CPU always maybe

        for i in range(len(p_unlab)):
            idxs_neigh = self.find_neighs(z_unlab[i].unsqueeze(0), z_lab, self.n_neighs)
            score[i] = self.dist_func(p_lab[idxs_neigh], p_unlab[i].unsqueeze(0)).mean()

        score *= -1
        _, querry_indices = torch.topk(score, acq_size)
        unlbld_idx = torch.where(torch.logical_not(active_data.lbld_mask))[0]

        return unlbld_idx[querry_indices]

    def find_neighs(self, p, A, n_neigh):
        dist = self.neigh_dist_func(p, A)
        return torch.argsort(dist)[:n_neigh]


class VAALSampler(TrainableSampler):
    def __init__(self, cfg_smp, device):
        super().__init__(cfg_smp, device, cfg_smp['train_every_k'])
        self.discriminator = Discriminator(self.cfg_smp['latent_dim'])
        self.bce_loss = nn.BCELoss()

    def forward(self, x):
        z_labeled = x[0]
        z_unlabeled = x[1]

        labeled_preds = self.discriminator(z_labeled)
        unlabeled_preds = self.discriminator(z_unlabeled)
        return labeled_preds, unlabeled_preds

    def sampler_loss(self, pred):
        labeled_preds = pred[0]
        unlabeled_preds = pred[1]

        lab_real_preds = torch.ones(labeled_preds.shape, device=self.dev)
        unlab_fake_preds = torch.zeros(unlabeled_preds.shape, device=self.dev)

        return self.bce_loss(labeled_preds, lab_real_preds) + self.bce_loss(unlabeled_preds, unlab_fake_preds)

    def model_loss(self, pred):
        labeled_preds = pred[0]
        unlabeled_preds = pred[1]

        lab_real_preds = torch.ones(labeled_preds.shape, device=self.dev)
        unlab_real_preds = torch.ones(unlabeled_preds.shape, device=self.dev)

        return self.bce_loss(labeled_preds, lab_real_preds) + self.bce_loss(unlabeled_preds, unlab_real_preds)

    def sample(self, active_data, acq_size, model):
        d_pool = active_data.unlabeled_trainset
        all_preds = []
        # TODO: Also work with batches!
        pbar = tqdm(d_pool)
        pbar.set_description(f"Sampling")
        for i, xt_p in enumerate(pbar):
            x = xt_p[0]
            x = x.to(self.dev)
            z = model.latent(x.unsqueeze(0))
            d = self.discriminator(z)
            all_preds.append(d)

        all_preds = torch.stack(all_preds).squeeze()
        all_preds = all_preds.view(-1)
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, acq_size)
        unlbld_idx = torch.where(torch.logical_not(active_data.lbld_mask))[0]

        return unlbld_idx[querry_indices]


class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN.
    Taken from https://github.com/sinhasam/vaal by Samarth Sinha et al."""
    def __init__(self, z_dim=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)


import torch.nn.init as init


def kaiming_init(m):
    """Taken from https://github.com/sinhasam/vaal by Samarth Sinha et al."""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


SAMPLER_DICT = {'cal': CAL, 'random': Random, 'vaal': VAALSampler}

def gaussian_kl_div_test():
    tc = unittest.TestCase()
    vec = torch.zeros(2)
    tc.assertAlmostEqual(float(gaussian_kl_div(mu_p=vec, log_var_p=vec, mu_q=vec, log_var_q=vec)), 0)

def gaussian_symmetric_kl_div_test():
    tc = unittest.TestCase()
    vec = torch.zeros(2)
    tc.assertAlmostEqual(float(gaussian_symmetric_kl_div(mu_p=vec, log_var_p=vec, mu_q=vec, log_var_q=vec)), 0)


if __name__ == '__main__':
    gaussian_kl_div_test()
    gaussian_symmetric_kl_div_test()
