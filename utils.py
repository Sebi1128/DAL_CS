import os
import random
from shutil import copyfile

import numpy as np
import torch
from pytorch_lightning.utilities.seed import seed_everything as light_seed

from src.base_models.encoders import ENCODER_DICT
from src.base_models.bottlenecks import BOTTLENECK_DICT
from src.base_models.decoders import DECODER_DICT
from src.base_models.classifiers import CLASSIFIER_DICT
from src.data import ActiveDataset

import wandb

SAVE_DIR = './save/'
SAVE_DIR_PARAM = SAVE_DIR + 'param/'
CONFIG_DIR = './config.py'


def config_defaulter(cfg):

    cfg.update({}, allow_val_change=True) 

    
    cfg = set_device(cfg)
    cfg = dimension_parametrizer(cfg)

    seed_everything(cfg.seed)
    
    return cfg


def set_device(cfg, verbose=True):

    if cfg.device.lower() in ['gpu', 'cuda']:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    cfg.update({'device': device}, allow_val_change=True) 

    if verbose:
        print(f"Device is {cfg.device}")

    return cfg

def dimension_parametrizer(cfg):

    dataset_name    = cfg.dataset['name']
    model_name      = cfg.autoenc['name']

    # hidden dimensions defined by model
    if model_name == 'base':
        hidden_dims = [128, 256]

    elif model_name == 'vaal':
        hidden_dims = [128, 256, 512, 1024]

    # input/output dimensions defined by dataset
    if dataset_name == 'cifar10':

        input_size = [3, 32, 32]
        output_size = 10
        
    elif dataset_name == 'cifar100':

        input_size = [3, 32, 32]
        output_size = 100

    elif dataset_name == 'mnist':

        input_size  = [1, 32, 32]
        output_size = 10

    elif dataset_name == 'fashion_mnist':

        input_size  = [1, 32, 32]
        output_size = 10

    cfg = set_dataset_dimensions(cfg, input_size, output_size, hidden_dims)
    
    return cfg

def set_dataset_dimensions(cfg, input_size, output_size, hidden_dims):

    cfg.autoenc.update({
        'input_size'    : input_size,
        'hidden_dims'   : hidden_dims
    })
    cfg.cls.update({
        'input_size': input_size,
        'output_size': output_size
    })

    active_dataset = ActiveDataset(cfg.dataset['name'], 
                               init_lbl_ratio=cfg.dataset['init_lbl_ratio'],
                               val_ratio=cfg.dataset['val_ratio'])

    dummy_data_loader = active_dataset.get_loader('train', batch_size=cfg.batch_size)

    encoder = ENCODER_DICT[cfg.autoenc['name']](cfg.autoenc)
    for (x, t) in dummy_data_loader:

        t = encoder(x)

        break

    cfg.autoenc.update({
            'feature_dim': list(t.size())[1:]
        })
    
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
        