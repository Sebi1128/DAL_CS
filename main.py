import argparse
from src.data import ActiveDataset
from src.model import Net
from src.base_models.samplers import SAMPLER_DICT
from src.training import epoch_run
from utils import config_defaulter, ModelWriter
from datetime import datetime

import yaml
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
        sampler = SAMPLER_DICT[cfg.smp['name']](cfg.smp, cfg.device).to(cfg.device)

        #wandb.watch(model, log="gradients", log_freq=1000, log_graph=(True))
        #wandb.watch(sampler, log="gradients", log_freq=1000, log_graph=(True))

        epoch_run(model, sampler, active_dataset, run_no, model_writer, cfg)

        if run_no < (cfg.n_runs - 1):
            train2lbl_idx = sampler.sample(
                active_dataset,
                step_acq_size,
                model
            )
            active_dataset.update(train2lbl_idx)

    wandb.run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config file', default='configs/cal_sampling_config.yaml')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    wandb.init(config=config,
               project="Deep Learning Project", entity="active_learners")

    cfg = wandb.config
    run_name = datetime.now().strftime("%Y_%m_%d_%H%M")[2:] + '_' + cfg.experiment_name + '_' + wandb.run.id
    wandb.run.name = run_name

    main(cfg)