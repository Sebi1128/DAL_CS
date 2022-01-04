import os
import random
from shutil import copyfile

import numpy as np
import torch
from pytorch_lightning.utilities.seed import seed_everything as light_seed
import copy

import wandb

from vae_training.pretrained_vae_loader import download_vae

SAVE_DIR = './save/'
SAVE_DIR_PARAM = SAVE_DIR + 'param/'
CONFIG_DIR = './config.py'

def config_lister(config):
    """
    For more than one seed, config_lister creates a list of configurations
    each with one of the seeds.
    """

    if isinstance(config['seed'], int):
        cfg_list = [config]
    else:
        seeds = [int(s) for s in config['seed'].split(',')]
        cfg_list = list()
        for s in seeds:
            cfg_s = copy.deepcopy(config)
            cfg_s['seed'] = s
            cfg_list.append(cfg_s)

    return cfg_list

def config_defaulter(cfg):
    """
    Regarding the default specifications of the datasets and models,
    config_defaulter 
    """

    # check whether Variational Autoencoder Data is downloaded 
    check_vae_download()

    cfg.update({}, allow_val_change=True) 

    try:
        a = cfg.visual_latent
    except:
        cfg.visual_latent = False

    cfg = get_device(cfg)
    cfg = dataset_parametrizer(cfg)

    seed_everything(cfg.seed)
    
    return cfg

def get_device(cfg, verbose=True):
    """
    Setting the device for the run.
    """
    if cfg.device.lower() in ['gpu', 'cuda']: # GPU
        # if gpu is wanted to be used and cuda is available 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else: #CPU
        device = torch.device("cpu")

    # updating the configuration
    cfg.update({'device': device}, allow_val_change=True) 

    if verbose: # 
        print(f"Device is {cfg.device}")

    return cfg

def dataset_parametrizer(cfg):
    """
    Setting model parameters according to the dataset.
    """

    name = cfg.dataset['name']

    if name == 'cifar10': # default settings for Cifar-10

        cfg.enc['input_size'] = [32, 32]
        cfg.dec['output_size'] = [32, 32]
        cfg.cls['output_size'] = 10
        cfg.cls['in_channels'] = 3
        
    elif name == 'cifar100': # default settings for Cifar-100

        cfg.enc['input_size'] = [32, 32]
        cfg.dec['output_size'] = [32, 32]
        cfg.cls['output_size'] = 100
        cfg.cls['in_channels'] = 3

    elif name == 'mnist': # default settings for MNIST

        cfg.enc['input_size'] = [32, 32]
        cfg.dec['output_size'] = [32, 32]
        cfg.cls['output_size'] = 10
        cfg.cls['in_channels'] = 1
        
    elif name == 'fmnist': # default settings for Fashion-MNIST

        cfg.enc['input_size'] = [32, 32]
        cfg.dec['output_size'] = [32, 32]
        cfg.cls['output_size'] = 10
        cfg.cls['in_channels'] = 1

    # channel settings for encoder and decoder
    cfg.enc['in_channels'] = cfg.cls['in_channels']
    cfg.dec['out_channels'] = cfg.cls['in_channels']
    
    return cfg

def check_vae_download():
    """
    Checking whether the variational autoencoder 
    """
    if not os.path.isdir('./vae_training/pretrained_models'):
        download_vae()

def seed_everything(seed: int):
    """
    Seeding everything for reproducibility. 
    Unfortunately, the models become very slow on GPU with deterministic algorithms.
    Hence, the method only works for the initial dataset partition but not for the models.

    To enable the total reproducibility, remove the last two commented lines of the code.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    light_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8:4096:2" # efficient but random

    # To have total reproducibility, please remove the following two lines:
    # torch.use_deterministic_algorithms(True) It does not work with GPU
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2" # reproducible but inefficient    
    
    print(f"Seed has been set to {seed}...")

    
class ModelWriter():
    """
    Custom model saver to the predefined folder with naming:
    save/param/<date>_<experiment_name>_<seed>_<W&B_ID>/<prefix>_weights.pth
    """
    def __init__(self, cfg):
        self.name = wandb.run.name
        self.dir = './save/param/' + self.name + '/'
        os.makedirs(self.dir)
        copyfile(CONFIG_DIR, self.dir + 'config.py')

    def write(self, model, prefix=''):
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save(model.state_dict(), self.dir + prefix + 'weights.pth')

    def load(self, model, prefix=''):
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html
        model.load_state_dict(torch.load(self.dir + prefix + 'weights.pth'))
        return model
        