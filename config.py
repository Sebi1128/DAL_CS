#%%

import wandb

# 1. Start a W&B run
wandb.init(project='Deep Learning Project')
cfg = wandb.config

# 2. Dataset
cfg.dataset = 'cifar10'

# 3. Hyperparameters 
cfg.batch_size = 100
cfg.n_epochs = 10

cfg.optimizer = 'adam'
cfg.learning_rate = 0.001

# 
cfg.device = 'gpu'

# %%
