"""
Deep Active Learning with Contrastive Sampling

Deep Learning Project for Deep Learning Course (263-3210-00L)  
by Department of Computer Science, ETH Zurich, Autumn Semester 2021 

Authors:  
Sebastian Frey (sefrey@student.ethz.ch)  
Remo Kellenberger (remok@student.ethz.ch)  
Aron Schmied (aronsch@student.ethz.ch)  
Guney Tombak (gtombak@student.ethz.ch)  
"""

import argparse
from src.data import ActiveDataset
from src.model import Net
from src.base_models.samplers import SAMPLER_DICT
from src.training import run
from utils import config_defaulter, config_lister, ModelWriter
from datetime import datetime

import yaml
import wandb


def main(cfg):
    # updating configuration (cfg) object with default parameters  
    cfg = config_defaulter(cfg)

    # constructing a model writer
    model_writer = ModelWriter(cfg)

    # constructing an active dataset
    active_dataset = ActiveDataset(cfg.dataset['name'], 
                               init_lbl_ratio=cfg.dataset['init_lbl_ratio'],
                               val_ratio=cfg.dataset['val_ratio'],
                               seed = cfg.seed)

    # defining number of samples to be added from unlabeled to the labeled dataset
    step_acq_size = int(cfg.update_ratio * len(active_dataset.base_trainset))

    for run_no in range(cfg.n_runs): # for loop over runs (with different labeled ratios)

        # reinitialize model & sampler in every run
        model = Net(cfg).to(cfg.device)
        sampler = SAMPLER_DICT[cfg.smp['name']](cfg.smp, cfg.device).to(cfg.device)

        # run with a specific ratio of labeled/unlabeled samples
        # training the models with validation and test at the end
        run(model, sampler, active_dataset, run_no, model_writer, cfg)

        # sampling
        if run_no < (cfg.n_runs - 1): # no need for sampling in the end
            # 
            train2lbl_idx = sampler.sample(active_dataset, step_acq_size, model)
            # updating the dataset: setting some trainset values to be 
            active_dataset.update(train2lbl_idx)

    # finishing the Weights and Biases run
    wandb.run.finish()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config file', default='configs/default_config.yaml')
    args = parser.parse_args()

    # getting parameters from the defined config.yaml file
    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # for more than one seed, config_lister creates list consists of configs with the specified seeds
    cfg_list = config_lister(config)

    for config in cfg_list: # for each seed value, a config file has been created by config_lister

        # initializing Weights and Biases Run with the data from config data of specified yaml file

        ########################################################
        # Please set this part according to your preferences
        ########################################################
        wandb.init(config=config)
        ########################################################
        # To use offline: mode = 'offline'
        # To disable    : mode = 'disabled'
        # Run with a specific project name: project='<project_name>'
        ########################################################
        
        # renaming configuration in favor of ease
        cfg = wandb.config 

        # setting the name of the run for Weights and Biases
        # with the format: <date>_<experiment_name>_<seed>_<W&B_ID>
        wandb.run.name = datetime.now().strftime("%Y_%m_%d_%H%M")[2:] + '_' \
                         + cfg.experiment_name + '_' + str(cfg.seed) + '_' + wandb.run.id

        main(cfg) # running an experiment with the specified configuration