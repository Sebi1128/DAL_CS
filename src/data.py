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

import sys
import math

from torchvision.datasets import CIFAR10, CIFAR100,  MNIST, FashionMNIST 
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
import random

from src.training_utils import TRANSFORMS_DICT

DATASETS_DICT = {
	'cifar10'	: CIFAR10,
	'cifar100'	: CIFAR100,
	'mnist'		: MNIST,
    'fmnist'	: FashionMNIST
}

def torch_rp_bool(k:int, n:int):
    """
    Gives a boolean array of size (n,) with 
    randomly selected indices of k True, (n-k) False 
    """
    x = torch.zeros(n, dtype=bool)
    perm = torch.randperm(n)
    idx = perm[:k]
    x[idx] = True
    return x

def seed_worker(worker_id):
    """Taken from https://pytorch.org/docs/stable/notes/randomness.html"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ActiveDataset():
    """
    A custom dataset consists of four disjoint sub-datasets:
    Training: (Labeled, Unlabeled), Validation, Test

    The 
    """
    def __init__(self, name, 
                 path=None, init_lbl_ratio=0.1,
                 val_ratio=0.1, transform=None,
                 seed=42):

        print(f'Constructing Active Dataset for {name}:')	

        # setting the parameters
        self.dataset_name = name
        self.dataset_path = './data/' + name if path is None else path
        self.transform = self._get_transform(transform)
        self.seed = seed

        self._take_datasets() # take the datasets from the corresponding torchvision loader 
        self._init_mask(init_lbl_ratio, val_ratio) # set initial self.lbld_mask
        self.update() # updating with the default set defined by self.lbld_mask

        self.iter_schedule = self.get_itersch()

    def _init_mask(self, init_lbl_ratio, val_ratio):
        if (val_ratio + init_lbl_ratio) > 1.0:
            sys.exit('The validation and initialization ratio sum should be less than 1.0!')

        # defining initial labeled ratio and validation ratio
        self.init_ratio = init_lbl_ratio
        self.val_ratio = val_ratio

        # number of samples in base trainset (every sample other than the test ones)
        n_btra = len(self.base_trainset)

        # number of samples for validation, training, and labeled training datasets
        n_val = int(n_btra * val_ratio)
        n_tra = int(n_btra - n_val)
        n_lbl = int(n_btra * init_lbl_ratio)

        # generating random validation and training sets
        self.val_mask = torch_rp_bool(n_val, n_btra)
        val_idx = torch.where(self.val_mask)[0]
        tra_idx = torch.where(torch.logical_not(self.val_mask))[0]

        self.validset = Subset(self.base_trainset, val_idx)
        self.trainset = Subset(self.base_trainset, tra_idx)

        # defining the indices corresponding to labeled samples in training set
        self.lbld_mask = torch_rp_bool(n_lbl, n_tra)


    def _take_datasets(self):
        """
        Returns datasets from the corresponding torchvision loader
        Datasets: CIFAR10, CIFAR100, MNIST, Fashion MNIST
        """
        DATASET = DATASETS_DICT[self.dataset_name]

        self.testset = DATASET(root=self.dataset_path, 
                                download=True,
                                train=False,
                                transform=self.transform)
        self.base_trainset = DATASET(root=self.dataset_path, 
                                        download=True,
                                        train=True,
                                        transform=self.transform)

    def _get_transform(self, transform):
        """Get the transforms defined for the datasets"""
        if isinstance(transform, str):
            if transform not in TRANSFORMS_DICT.keys():
                print(f'No transform known as {transform}')
                print('Returning None')
                return None
            else:
                return TRANSFORMS_DICT[transform.lower()]
        elif transform is None:
            return TRANSFORMS_DICT[self.dataset_name.lower()]

    def update(self, idx=list()):
        """
        Updating labeled and unlabeled set according to indices to be labeled set of training set.
        """
        if len(idx) > 0: # if the indices are not 
            self.lbld_mask[idx] = True

        # updating the ratio
        self.lbld_ratio = torch.true_divide(torch.sum(self.lbld_mask), len(self.lbld_mask))

        # updating labeled and unlabeled indices accordingly
        lbld_idx = torch.where(self.lbld_mask)[0]
        unlbld_idx = torch.where(torch.logical_not(self.lbld_mask))[0]

        # constructing subsets 
        self.labeled_trainset = Subset(self.trainset, lbld_idx)
        self.unlabeled_trainset = Subset(self.trainset, unlbld_idx)

    def get_loader(self, spec, batch_size, shuffle=True, num_workers=8):
        """
        Getting loader according to the desired sub-dataset
        """
        if spec.lower() == 'train': # training set
            dataset = self.trainset
        elif spec.lower() == 'test': # test set
            dataset = self.testset
        elif spec.lower() in ['valid', 'validation']: # validation set
            dataset = self.validset
        elif spec.lower() == 'labeled': # labeled samples in training dataset
            dataset = self.labeled_trainset
        elif spec.lower() == 'unlabeled': # unlabeled samples in training dataset
            dataset = self.unlabeled_trainset
        elif spec.lower() == 'train_all': # training set + validation set
            dataset = self.base_trainset
        else:
            sys.exit(f"No set known as {spec}, exiting...")

        'To have a reproducible dataset'
        g = torch.Generator() # for reproducibility
        g.manual_seed(self.seed) # for reproducibility

        loader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle,
                            num_workers=num_workers,
                            worker_init_fn=seed_worker, # for reproducibility
                            generator=g # for reproducibility
            )

        return loader

    def get_itersch(self, uniform=True, setL=None, setU=None):
        """
        Get iteration schedule (in an epoch)

        The method is mainly required to define whether the training (for reconstruction) 
        or labeled training (for classification) set to be   
        
        Generates an iterative module for training in favor of a homogeneous training procedure.
        First, it finds the smallest ratio between number of elements in setL and setU.
        e.g.: |setL | = 4, |setU| = 12 -> a = 1, b = 3 -> seq_len = 4

        The method is not reliable always. For caution, we used it with try/except 
        """
        
        setL = self.labeled_trainset if setL is None else setL

        # realize that we can use whole dataset as unlabeled
        setU = self.trainset if setU is None else setU 

        # finding simplest ratio using greatest common divider
        gcdLU = math.gcd(len(setL), len(setU))
        a, b = len(setL) // gcdLU, len(setU) // gcdLU

        # defining sequence length
        seq_len = a + b
        n_seq = len(setL + setU) // seq_len

        stack = torch.zeros(n_seq, seq_len, dtype=bool)

        if uniform: # if uniform, for each subset of a + b, the latest a of them are labeled 
            stack[:,:a] = True
        else: # if not uniform, a labeled ones are randomly given in sequence of length a + b
            for k in range(n_seq):
                stack[k,:] = torch_rp_bool(a, seq_len)

        iter_schedule = stack.flatten()
        assert torch.sum(iter_schedule) == len(setL)
        assert len(iter_schedule) == len(self.trainset) + len(self.labeled_trainset)

        return iter_schedule

