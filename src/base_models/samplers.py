"""
Deep Active Learning with Contrastive Sampling

Deep Learning Project for Deep Learning Course (263-3210-00L)  
by Department of Computer Science, ETH Zurich, Autumn Semester 2021 

Authors:  
Sebastian Frey (sefrey@student.ethz.ch)  
Remo Kellenberger (remok@student.ethz.ch)  
Aron Schmied (aronsch@student.ethz.ch)  
Guney Tombak (gtombak@student.ethz.ch)  
"""

import torch.nn as nn
from scipy.special import kl_div
import numpy as np
import torch
from tqdm import tqdm
from torch import optim
from .model_utils import kaiming_init
from torch.nn.functional import normalize
from sklearn.decomposition import PCA


class BaseSampler(nn.Module):
    """Parent Sampler class for the samplers do not require training"""
    def __init__(self, cfg_smp, device):
        super(BaseSampler, self).__init__()
        self.cfg_smp = cfg_smp
        self.dev = device
        self.batch_size = 20 # not to confuse with #neighbors and #classes
        self.trainable = False

    def sample(self, active_data, acq_size, model):
        raise NotImplementedError()

    def forward(self):
        return 0.0

class TrainableSampler(BaseSampler):
    """Parent Sampler class for the samplers require training"""
    def __init__(self, cfg_smp, device):
        super(TrainableSampler, self).__init__(cfg_smp, device)
        self.trainable = True
        self.n_sub_epochs = cfg_smp['n_sub_epochs']

        self.optimizer = None

    def sampler_loss(self, pred):
        raise NotImplementedError()

    def model_loss(self, pred):
        raise NotImplementedError()


class Random(BaseSampler):
    """Random sampling class that randomly selects the indices transferred from unlabeled to labeled"""
    def __init__(self, cfg_smp, device):
        super().__init__(cfg_smp, device)

    def sample(self, active_data, acq_size, model):
        # indices of unlabeled indices in the training
        unlbld_idx = torch.where(torch.logical_not(active_data.lbld_mask))[0]
        # selecting randomly from the unlabeled without replacing
        sample_idx = np.random.choice(len(unlbld_idx), acq_size, replace=False)
        return unlbld_idx[sample_idx]

def gaussian_kl_div(mu_p, log_var_p, mu_q, log_var_q):
    """KL(Q||P) where Q, P ~ N(mu_1:k, diag(sigma2_1:k))"""
    return 0.5*(torch.exp(-log_var_q)*(torch.exp(log_var_p) + (mu_q-mu_p)**2) - 1 + log_var_q - log_var_p).sum(1)


def gaussian_symmetric_kl_div(mu_p, log_var_p, mu_q, log_var_q):
    """KL(Q||P)-KL(P||Q) where Q, P ~ N(mu_1:k, diag(sigma2_1:k))"""
    return 0.5*(torch.exp(-log_var_q)*(torch.exp(log_var_p) + (mu_q-mu_p)**2)
                + torch.exp(-log_var_p)*(torch.exp(log_var_q) + (mu_p-mu_q)**2) -2).sum(1)


class CAL(BaseSampler):
    """
    
    """
    def __init__(self, cfg_smp, device):
        super().__init__(cfg_smp, device)

        # number of neighbours to be considered
        self.n_neighs = cfg_smp['n_neighs']

        # distince function between probabilities: KL Divergence
        self.dist_func = lambda y_l, y_p: kl_div(y_l.cpu().detach(), y_p.cpu().detach()).sum(1)

        # distance function for finding neighbours
        if cfg_smp['neigh_dist'] == 'l2': # l2 distance
            self.neigh_dist_func = lambda p, A: torch.sum((p[..., 0] - A[..., 0]) ** 2, axis=1)
        elif cfg_smp['neigh_dist'] == 'kldiv': # Kl divergence
            self.neigh_dist_func = lambda p, A: gaussian_kl_div(p[..., 0], p[..., 1], A[..., 0], A[..., 1])
        elif cfg_smp['neigh_dist'] == 'sym_kldiv': # symmetric KL divergence
            self.neigh_dist_func = lambda p, A: gaussian_symmetric_kl_div(p[..., 0], p[..., 1], A[..., 0], A[..., 1])
        else:
            raise ValueError("cfg_smp.neigh_dist set to {} which is not known".format(cfg_smp['neigh_dist']))

    def sample(self, active_data, acq_size, model):
        """
        Sampling function which returns the best samples according to Contrastive Active Sampling
        It returns the indices in training to be set as labeled which were initially unlabeled.
        """

        # generating labeled and unlabeled loaders 
        labeled_data = active_data.get_loader('labeled', batch_size=self.batch_size)
        unlabeled_data = active_data.get_loader('unlabeled', batch_size=self.batch_size)

        # initializing the latent space and probability lists for labeled (lab) and unlabeled (unlab)
        z_lab, p_lab, z_unlab, p_unlab = list(), list(), list(), list()

        for x, _ in labeled_data: # for the labeled samples
            x = x.to(self.dev) # put input to the device (cpu/gpu)
            z = model.latent_param(x) # get latent parameters (mean, log variance)
            z_lab.append(z) # append latent parameters to the list
            p = model.classify(x) # get the classification result from the model
            p_lab.append(p) # append output probabilities to the list
        for x, _ in unlabeled_data: # for the unlabeled samples
            x = x.to(self.dev) # put input to the device (cpu/gpu)
            z = model.latent_param(x) # get latent parameters (mean, log variance)
            z_unlab.append(z) # append latent parameters to the list
            p = model.classify(x) # get the classification result from the model
            p_unlab.append(p) # append output probabilities to the list

        # transform lists of latent parameters into torch tensor for labeled and unlabeled
        z_lab = torch.cat(z_lab) 
        z_unlab = torch.cat(z_unlab)

        # transform lists of classification probabilities (normalized) into torch tensor for labeled and unlabeled
        p_lab = normalize(torch.exp(torch.cat(p_lab)), p=1)
        p_unlab = normalize(torch.exp(torch.cat(p_unlab)), p=1)

        score = torch.zeros((len(p_unlab))) # score tensor initialization

        for i in range(len(p_unlab)): # for each unlabeled sample
            # find the neighbours to be considered in latent space representation
            idxs_neigh = self.find_neighs(z_unlab[i].unsqueeze(0), z_lab, self.n_neighs)
            # calculating the score as the mean distance from the neighbours in classification probability space
            score[i] = self.dist_func(p_lab[idxs_neigh], p_unlab[i].unsqueeze(0)).mean()

        # finding the best neighbor indices
        _, querry_indices = torch.topk(score, acq_size) 

        # indices of unlabeled indices in the training
        unlbld_idx = torch.where(torch.logical_not(active_data.lbld_mask))[0] 

        # returning indices in training to be set as labeled which were initially unlabeled
        return unlbld_idx[querry_indices] 

    def find_neighs(self, p, A, n_neigh):
        """
        An algorithm to find nearest neighbors
        n_neigh : number of neighbours (K), 
        p       : the sample
        A       : possible neighbours
        """
        dist = self.neigh_dist_func(p, A)
        return torch.argsort(dist)[:n_neigh]

class VAALSampler(TrainableSampler):
    """VAAL sampler from https://github.com/sinhasam/vaal"""
    def __init__(self, cfg_smp, device):
        super().__init__(cfg_smp, device)
        self.discriminator = Discriminator(self.cfg_smp['latent_dim'])
        self.bce_loss = nn.BCELoss()

        optimizers = {'adam': optim.Adam, 'sgd': optim.SGD}
        self.optimizer = optimizers[cfg_smp['optimizer'].lower()](
            list(self.discriminator.parameters()),
            lr=cfg_smp['lr']
        )

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
            mu = model.latent_param(x.unsqueeze(0))[..., 0]
            d = self.discriminator(mu)
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
        kaiming_init(self)

    def forward(self, z):
        return self.net(z)


class CAL_PCA(BaseSampler):
    def __init__(self, cfg_smp, device):
        super().__init__(cfg_smp, device)

        self.n_neighs = cfg_smp['n_neighs']
        self.n_pca_comp = cfg_smp['n_pca_comp']

        self.dist_func = lambda y_l, y_p: kl_div(y_l.cpu().detach(), y_p.cpu().detach()).sum(1)

        self.neigh_dist_func = lambda p, A: torch.sum((p[...] - A[...]) ** 2, axis=1)


    def sample(self, active_data, acq_size, model):
        """
        Sampling function which returns the best samples according to Contrastive Active Sampling
        It returns the indices in training to be set as labeled which were initially unlabeled.
        """
        
        # Take all the training data for the PCA
        all_DL = active_data.get_loader('train_all', batch_size=len(active_data.base_trainset))
        all_iter = iter(all_DL)
        x_all, _ = next(all_iter)
        pca = PCA(n_components=self.n_pca_comp)
        pca_model = pca.fit(torch.reshape(x_all, (len(active_data.base_trainset), -1)))
        del x_all, _, all_DL, all_iter

        # generating labeled and unlabeled loaders 
        labeled_data = active_data.get_loader('labeled', batch_size=self.batch_size)
        unlabeled_data = active_data.get_loader('unlabeled', batch_size=self.batch_size)

        # initializing the latent space and probability lists for labeled (lab) and unlabeled (unlab)
        z_lab, p_lab, z_unlab, p_unlab = list(), list(), list(), list()

        for x, _ in labeled_data:  # for the labeled samples
            x = x.to(self.dev) # put input to the device (cpu/gpu)
            # Project the data to the PCA coordinates
            z = pca_model.transform(torch.reshape(x.cpu(), (self.batch_size, -1)))
            z_lab.append(torch.tensor(z, device=self.dev)) # append latent parameters to the list
            p = model.classify(x) # get the classification result from the model
            p_lab.append(p) # append output probabilities to the list
        for x, _ in unlabeled_data: # for the unlabeled samples
            x = x.to(self.dev) # put input to the device (cpu/gpu)
            # Project the data to the PCA coordinates
            z = pca_model.transform(torch.reshape(x.cpu(), (self.batch_size, -1))) 
            z_unlab.append(torch.tensor(z, device=self.dev)) # append latent parameters to the list
            p = model.classify(x) # get the classification result from the model
            p_unlab.append(p) # append output probabilities to the list


        # transform lists of latent parameters into torch tensor for labeled and unlabeled
        z_lab = torch.cat(z_lab) 
        z_unlab = torch.cat(z_unlab)

        # transform lists of classification probabilities (normalized) into torch tensor for labeled and unlabeled
        p_lab = normalize(torch.exp(torch.cat(p_lab)), p=1)
        p_unlab = normalize(torch.exp(torch.cat(p_unlab)), p=1)

        score = torch.zeros((len(p_unlab))) # score tensor initialization

        for i in range(len(p_unlab)): # for each unlabeled sample
            # find the neighbours to be considered in latent space representation
            idxs_neigh = self.find_neighs(z_unlab[i].unsqueeze(0), z_lab, self.n_neighs)
            # calculating the score as the mean distance from the neighbours in classification probability space
            score[i] = self.dist_func(p_lab[idxs_neigh], p_unlab[i].unsqueeze(0)).mean()

        # finding the best neighbor indices
        _, querry_indices = torch.topk(score, acq_size)

        # indices of unlabeled indices in the training
        unlbld_idx = torch.where(torch.logical_not(active_data.lbld_mask))[0]

        # returning indices in training to be set as labeled which were initially unlabeled
        return unlbld_idx[querry_indices]

    def find_neighs(self, p, A, n_neigh):
        """
        An algorithm to find nearest neighbors
        n_neigh : number of neighbours (K), 
        p       : the sample
        A       : possible neighbours
        """
        dist = self.neigh_dist_func(p, A)
        return torch.argsort(dist)[:n_neigh]

# dictionary containing sampler classes
SAMPLER_DICT = {'random': Random, 'cal': CAL, 'cal_pca': CAL_PCA, 'vaal': VAALSampler}

