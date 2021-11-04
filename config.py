import wandb

# Start a W&B run
wandb.init(project='Deep Learning Project', 
           entity="active_learners")
cfg = wandb.config

# Dataset
cfg.dataset = { 'name'              : 'cifar10',
                'init_lbl_ratio'    : 0.1,
                'val_ratio'         : 0.1,
            }


# Hyperparameters 
cfg.batch_size = 50
cfg.n_epochs = 10

cfg.optimizer = 'adam'
cfg.learning_rate = 0.001
cfg.n_neighs = 10

# Active Learning
cfg.update_ratio = 0.05
cfg.n_runs = 9

# System
cfg.device = 'gpu'
