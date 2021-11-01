import sys
import math

from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Subset
import torch

from src.utils import TRANSFORMS_DICT

DATASETS_DICT = {
	'cifar10'	: CIFAR10,
	'cifar100'	: CIFAR100
}

def torch_rp_bool(k:int, n:int):
    '''
	Gives a boolean array of size (n,) with 
 	randomly selected indices of k True, (n-k) False 
    '''
    x = torch.zeros(n, dtype=bool)
    perm = torch.randperm(n)
    idx = perm[:k]
    x[idx] = True
    return x

class ActiveDataset():
	def __init__(self, dataset_name, 
              	 dataset_path=None, init_ratio=0.1,
                 val_ratio=0.1, transform=None):	

		self.dataset_name = dataset_name
		self.dataset_path = './data/' + dataset_name if dataset_path is None else dataset_path
		self.transform = self._get_transform(transform)
		
		self._take_datasets()
		self._init_mask(init_ratio, val_ratio)
		self.update()

	def _init_mask(self, init_ratio, val_ratio):

		if (val_ratio + init_ratio) > 1.0:
			sys.exit('The validation and initialization ratio sum should be less than 1.0!')

		self.init_ratio = init_ratio
		self.val_ratio = val_ratio
		
		n_btra = len(self.base_trainset)

		n_val = int(n_btra * val_ratio)
		n_tra = int(n_btra - n_val)
		n_lbl = int(n_btra * init_ratio)

		self.val_mask = torch_rp_bool(n_val, n_btra)
		val_idx = torch.where(self.val_mask)[0]
		tra_idx = torch.where(torch.logical_not(self.val_mask))[0]

		self.validset = Subset(self.base_trainset, val_idx)
		self.trainset = Subset(self.base_trainset, tra_idx)

		self.lbld_mask = torch_rp_bool(n_lbl, n_tra)


	def _take_datasets(self):
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

		if isinstance(transform, str):
			if transform not in TRANSFORMS_DICT.keys():
				print(f'No transform known as {transform}')
				print('Returning None')
				return None
			else:
				return TRANSFORMS_DICT[transform]
		else:
			return transform

	def update(self, idx=list()):

		if len(idx) > 0:
			self.lbld_mask[idx] = True

		self.lbld_ratio = torch.sum(self.lbld_mask) / len(self.lbld_mask)

		lbld_idx = torch.where(self.lbld_mask)[0]
		unlbld_idx = torch.where(torch.logical_not(self.lbld_mask))[0]

		self.labeled_trainset = Subset(self.trainset, lbld_idx)
		self.unlabeled_trainset = Subset(self.trainset, unlbld_idx)

	def get_loader(self, spec, batch_size, shuffle=True, num_workers=-1):
		if spec.lower == 'train':
			dataset = self.trainset
		elif spec.lower == 'test':
			dataset = self.testset
		elif spec.lower in ['valid', 'validation']:
			dataset = self.validset
		elif spec.lower == 'labeled':
			dataset = self.labeled_trainset
		elif spec.lower == 'unlabeled':
			dataset = self.unlabeled_trainset
		else:
			sys.exit(f"No set known as {spec}, exiting...")

		loader = DataLoader(dataset, 
                      	    batch_size=batch_size, 
                            shuffle=shuffle,
                            num_workers=num_workers)

		return loader

	def get_itersch(self, uniform=True, setA=None, setB=None):

		setA = self.labeled_trainset if setA is None else setA
		setB = self.unlabeled_trainset if setB is None else setB 
		
		gcdAB = math.gcd(len(setA), len(setB))
		a, b = len(setA) // gcdAB, len(setB) // gcdAB

		seq_len = a + b
		n_seq = len(self.trainset) // seq_len

		stack = torch.zeros(n_seq, seq_len, dtype=bool)

		if uniform:
			stack[:,:a] = True
		else:
			for k in range(n_seq):
				stack[k,:] = torch_rp_bool(a, seq_len)

		iter_schedule = stack.flatten()
		assert torch.sum(iter_schedule) == len(setA)
		assert len(iter_schedule) == len(self.trainset)

		return iter_schedule

