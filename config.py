import wandb

# Start a W&B run
wandb.init(project="Deep Learning Project", 
           entity="active_learners",
           mode="disabled")
cfg = wandb.config

# Dataset and Active Learning
cfg.dataset = { 'name'              : 'cifar10',
                'init_lbl_ratio'    : 0.1,
                'val_ratio'         : 0.1,}

cfg.update_ratio = 0.05
cfg.n_runs = 9
cfg.n_neighs = 10 # will change and add other sampler parameters

# Run Hyperparameters 
cfg.batch_size = 50
cfg.n_epochs = 2

cfg.optimizer = 'adam'
cfg.learning_rate = 0.001
cfg.momentum = 0.0

# System
cfg.device = 'gpu'

# Architecture

cfg.enc = {'name'   : 'base'}
cfg.btk = {'name'   : 'base'}
cfg.dec = {'name'   : 'base'}
cfg.cls = {'name'   : 'base'}