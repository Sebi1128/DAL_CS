from torch import optim
from src.data import ActiveDataset
from src.model import Net
from src.base_models.samplers import SAMPLER_DICT
from src.training import epoch_run
from utils import config_defaulter, ModelWriter
from config import get_wandb_config

import wandb

def main(cfg):
    cfg = config_defaulter(cfg)
    model_writer = ModelWriter(cfg)

    active_dataset = ActiveDataset(cfg.dataset['name'], 
                               init_lbl_ratio=cfg.dataset['init_lbl_ratio'],
                               val_ratio=cfg.dataset['val_ratio'])
    
    step_acq_size = int(cfg.update_ratio * len(active_dataset.base_trainset))

    for run_no in range(cfg.n_runs):
        # reinit model & sampler in every run as in VAAL
        model = Net(cfg).to(cfg.device)
        sampler = SAMPLER_DICT[cfg.smp['name']](cfg.smp, cfg.device)

        wandb.watch(model)
        wandb.watch(sampler)

        if sampler.trainable:
            sampler.optimizer = optim.Adam(sampler.parameters(), lr=cfg.smp['lr'])

        if cfg.optimizer.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
        elif cfg.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)

        epoch_run(model, sampler, active_dataset, optimizer, run_no, model_writer, cfg)

        if run_no < (cfg.n_runs - 1):
            train2lbl_idx = sampler.sample(
                active_dataset,
                step_acq_size,
                model
            )
            active_dataset.update(train2lbl_idx)

    wandb.run.finish()


if __name__ == "__main__":
    cfg = get_wandb_config()
    main(cfg)