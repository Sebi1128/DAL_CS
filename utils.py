import os
import random
from shutil import copyfile

import numpy as np
import torch
from pytorch_lightning.utilities.seed import seed_everything as light_seed


import wandb

SAVE_DIR = './save/'
SAVE_DIR_PARAM = SAVE_DIR + 'param/'
CONFIG_DIR = './config.py'


def config_defaulter(cfg):

    cfg.update({}, allow_val_change=True) 

    
    cfg = get_device(cfg)
    cfg = dataset_parametrizer(cfg)

    seed_everything(cfg.seed)
    
    return cfg


def get_device(cfg, verbose=True):

    if cfg.device.lower() in ['gpu', 'cuda']:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    cfg.update({'device': device}, allow_val_change=True) 

    if verbose:
        print(f"Device is {cfg.device}")

    return cfg

def dataset_parametrizer(cfg):

    name = cfg.dataset['name']

    if name == 'cifar10':

        cfg.enc['input_size'] = [32, 32]
        cfg.dec['output_size'] = [32, 32]
        cfg.cls['output_size'] = 10
        
    elif name == 'cifar100':

        cfg.enc['input_size'] = [32, 32]
        cfg.dec['output_size'] = [32, 32]
        cfg.cls['output_size'] = 100

    return cfg

def seed_everything(seed: int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    light_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    print(f"Seed has been set to {seed}...")

    
class ModelWriter():
    def __init__(self, cfg):
        self.name = wandb.run.name
        self.dir = './save/param/' + self.name + '/'
        os.makedirs(self.dir)
        copyfile(CONFIG_DIR, self.dir + 'config.py')

    def write(self, model, prefix=''):
        # maybe we should save epoch no too!
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save(model.state_dict(), self.dir + prefix + 'weights.pth')

    def load(self, model, prefix=''):
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        model.load_state_dict(torch.load(self.dir + prefix + 'weights.pth'))
        return model
        