from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Subset
import torch

DATASETS_DICT = {
	'cifar10'	: CIFAR10,
	'cifar100'	: CIFAR100
}

class ActiveDataset():
	def __init__(self, dataset_name, 
              	 dataset_path='./data', init_ratio=0.1,
                 transform=None):	
		self.dataset_name = dataset_name
		self.dataset_path = dataset_path
  		
		self.transform = transform
		
		self._take_datasets()
		self._init_mask(init_ratio)
		self.update()

	def _init_mask(self, init_ratio):
		self.init_ratio = init_ratio
		
		n_ins = len(self.trainset)

		self.lbld_mask = torch.zeros(n_ins, dtype=bool)

		perm = torch.randperm(n_ins)
		idx = perm[:int(init_ratio*n_ins)]
		self.lbld_mask[idx] = True


	def _take_datasets(self):
		DATASET = DATASETS_DICT[self.dataset_name]

		self.testset = DATASET(root=self.dataset_path, 
                         	   download=True,
							   train=False,
                        	   transform=self.transform)
		self.trainset = DATASET(root=self.dataset_path, 
                         	    download=True,
							    train=True,
                        	    transform=self.transform)

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
		elif spec.lower == 'labeled':
			dataset = self.labeled_trainset
		elif spec.lower == 'unlabeled':
			dataset = self.unlabeled_trainset

		loader = DataLoader(dataset, 
                      	    batch_size=batch_size, 
                            shuffle=shuffle,
                            num_workers=num_workers)

		return loader
