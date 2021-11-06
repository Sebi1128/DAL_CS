from torch import optim
from src.data import ActiveDataset
from src.model import Net, Sampler
from src.training import epoch_run
from utils import config_defaulter, ModelWriter
from config import cfg

import wandb

def main(cfg):

    cfg = config_defaulter(cfg)
    model_writer = ModelWriter(cfg)

    active_dataset = ActiveDataset(cfg.dataset['name'], 
                               init_lbl_ratio=cfg.dataset['init_lbl_ratio'],
                               val_ratio=cfg.dataset['val_ratio'])
    
    step_acq_size = int(cfg.update_ratio * len(active_dataset.base_trainset))

    model = Net(cfg).to(cfg.device)
    sampler = Sampler(cfg.smp)
    wandb.watch(model)
    
    if cfg.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    elif cfg.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)

    for run_no in range(cfg.n_runs):
        epoch_run(model, active_dataset, optimizer, run_no, model_writer, cfg)

        if run_no < (cfg.n_runs - 1):
            train2lbl_idx = sampler(active_dataset, step_acq_size, model, cfg.device)
            active_dataset.update(train2lbl_idx)

    wandb.run.finish()
        
if __name__ == "__main__":
    main(cfg)