import wandb

# Start a W&B run
wandb.init(project="Deep Learning Project", 
           entity="active_learners",
           mode="disabled")
cfg = wandb.config

# Seed

cfg.seed = 42

# Dataset and Active Learning
cfg.dataset = { 'name'              : 'cifar10',
                'init_lbl_ratio'    : 0.1,
                'val_ratio'         : 0.1,}

cfg.update_ratio = 0.05
cfg.n_runs = 9 # should be one more than the number of sampling updates

cfg.smp = {'name': 'cal', 'n_neighs': 10, 'neigh_dist': 'kldiv'}
#cfg.smp = {'name': 'random'}
#cfg.smp = {'name': 'vaal', 'latent_dim': 256, 'lr': 0.001, 'n_sub_epochs': 1}

# Run Hyperparameters 
cfg.batch_size = 10
cfg.n_epochs = 20

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

