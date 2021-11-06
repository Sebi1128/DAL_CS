from scipy.special import kl_div
import numpy as np
import torch
from tqdm import tqdm

class CAL():
    def __init__(self, cfg_smp):
        self.n_neighs = cfg_smp['n_neighs']

        # This can be parametrized later easily
        self.dist_func = lambda y_l, y_p: kl_div(y_l.cpu().detach().numpy().flatten(), 
                                                 y_p.cpu().detach().numpy().flatten()).sum()
        self.neigh_dist_func = lambda p, A: torch.sum((p - A)**2, axis=1)
        

    def __call__(self, active_data, acq_size, model, device):

        # REVIEW shoud we use the device as cpu always?
        D_lab = active_data.labeled_trainset
        D_pool = active_data.unlabeled_trainset

        Z_lab = list()

        for x, _ in D_lab:
            z = model.latent(x.to(device).unsqueeze(0))
            Z_lab.append(z) 

        Z_lab = torch.stack(Z_lab).squeeze()

        S_xp = np.zeros((len(D_pool)))

        # NOTE This part is very slow, 
        # we can improve it using batch etc
        # or using CPU always maybe

        pbar = tqdm(D_pool)
        pbar.set_description(f"Sampling")
        for i, xt_p in enumerate(pbar):
            x_p = xt_p[0].to(device).unsqueeze(0)
            
            s = np.zeros((self.n_neighs))
            
            z_p = model.latent(x_p)
            idxs_neigh = self.find_neighs(z_p, Z_lab, self.n_neighs)

            log_y_p = model.classify(x_p) 
            y_p = torch.exp(log_y_p) # Sum up to 1

            for j, idx in enumerate(idxs_neigh):
                x_l, t_l = D_lab[idx]
                x_l = x_l.to(device).unsqueeze(0)
                log_y_l = model.classify(x_l) 
                y_l = torch.exp(log_y_l) # Sum up to 1
                s[j] = self.dist_func(y_l, y_p)       
            
            S_xp[i] = np.mean(s)
            

        # REVIEW CHECK THE TRANSITION
        unlbl2lbl_idx = np.argsort(-1*S_xp)[:acq_size]
        unlbld_idx = torch.where(torch.logical_not(active_data.lbld_mask))[0]
        
        return unlbld_idx[unlbl2lbl_idx]

    def find_neighs(self, p, A, n_neigh):
        dist = self.neigh_dist_func(p, A)
        return torch.argsort(dist)[:n_neigh]

    
SAMPLER_DICT = {'cal'   : CAL}