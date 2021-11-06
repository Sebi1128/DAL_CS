from tqdm import tqdm
import torch
from torch import optim
from src.data import ActiveDataset
from src.model import Net, train_epoch, validate
from src.samplers import cal
from utils import config_defaulter, ModelWriter
from config import cfg

import wandb

def main(cfg):

    cfg = config_defaulter(cfg)
    model_writer = ModelWriter(cfg)

    active_dataset = ActiveDataset(cfg.dataset['name'], 
                               init_lbl_ratio=cfg.dataset['init_lbl_ratio'],
                               val_ratio=cfg.dataset['val_ratio'])
    
    step_acq_size = cfg.update_ratio * len(active_dataset.trainset)

    model = Net(cfg).to(cfg.device)
    wandb.watch(model)
    
    if cfg.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    elif cfg.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)

    for run_no in range(cfg.n_runs):
        pbar = tqdm(range(cfg.n_epochs))
        #TODO: Write a code that saves the parameters with lowest validation loss
        for epoch_no in pbar:
            c_train_loss, r_train_loss = train_epoch(model, active_dataset, optimizer, 
                                                     batch_size=cfg.batch_size, device=cfg.device)
            c_valid_loss, r_valid_loss = validate(model, active_dataset, 
                                                  batch_size=cfg.batch_size, device=cfg.device)

            info_text = f"{run_no+1}|C|R Train: {c_train_loss:.5f}|{r_train_loss:.5f}"
            info_text += f"  Valid: {c_valid_loss:.5f}|{r_valid_loss:.5f}"
            pbar.set_description(info_text, refresh=True)

            wandb.log({"r_train_loss": r_train_loss, "c_train_loss": c_train_loss})
            wandb.log({"r_valid_loss": r_valid_loss, "c_valid_loss": c_valid_loss})

            if not (epoch_no or run_no): # initialization
                r_best_valid_loss = r_valid_loss
                c_best_valid_loss = c_valid_loss
                model_writer.write(model, 'c_')
                model_writer.write(model, 'r_')
            else:
                if r_valid_loss < r_best_valid_loss:
                    r_best_epoch_no = epoch_no
                if c_valid_loss < c_best_valid_loss:
                    c_best_epoch_no = epoch_no

            if not epoch_no:
                r_best_epoch_no = -1
                c_best_epoch_no = -1
                    

        print(f"Best Classification Loss \t{c_best_valid_loss} with Epoch No {c_best_epoch_no} for Run {run_no}")
        print(f"Final Reconstruction Loss \t{r_best_valid_loss} with Epoch No {r_best_epoch_no} for Run {run_no}")

        if run_no < (cfg.n_runs - 1):
            train2lbl_idx = cal(active_dataset, step_acq_size, 
                                model, cfg.n_neighs, cfg.device)
            active_dataset.update(train2lbl_idx)

    wandb.run.finish()
        
if __name__ == "__main__":
    main(cfg)