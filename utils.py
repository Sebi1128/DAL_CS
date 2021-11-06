import torch

def config_defaulter(cfg):

    cfg.update({}, allow_val_change=True) 

    cfg = get_device(cfg)
    cfg = dataset_parametrizer(cfg)
    
    return cfg


def get_device(cfg, verbose=True):

    if cfg.device.lower() in ['gpu', 'cuda']:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    cfg.update({'device': device}, 
                   allow_val_change=True) 

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

