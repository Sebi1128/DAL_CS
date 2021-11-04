from tqdm import tqdm
import torch
from torch import optim
from src.data import ActiveDataset
from src.model import Net, train_epoch, validate
from src.samplers import cal
from utils import config_defaulter
from config import cfg

import wandb

def main(cfg):

    cfg = config_defaulter(cfg)

    active_dataset = ActiveDataset(cfg.dataset['name'], 
                               init_lbl_ratio=cfg.dataset['init_lbl_ratio'],
                               val_ratio=cfg.dataset['val_ratio'])
    
    step_acq_size = cfg.update_ratio * len(active_dataset.trainset)

    if cfg.device.lower() in ['gpu', 'cuda']:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Device is {device}")

    model = Net().to(device)
    wandb.watch(model)
    
    if cfg.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    elif cfg.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)

    for run_no in range(cfg.n_runs):
        pbar = tqdm(range(cfg.n_epochs))
        #TODO: Write a code that saves the parameters with lowest validation loss
        for k in pbar:
            c_train_loss, r_train_loss = train_epoch(model, active_dataset, optimizer, 
                                                        batch_size=cfg.batch_size, device=device)
            c_valid_loss, r_valid_loss = validate(model, active_dataset, 
                                                    batch_size=cfg.batch_size, device=device)

            info_text = f"C|R Train Loss: {c_train_loss:.5f}|{r_train_loss:.5f}"
            info_text += f"C|R Valid Loss: {c_valid_loss:.5f}|{r_valid_loss:.5f}"
            pbar.set_description(info_text, refresh=True)

            wandb.log({"r_train_loss": r_train_loss, "c_train_loss": c_train_loss})
            wandb.log({"r_valid_loss": r_valid_loss, "c_valid_loss": c_valid_loss})

        print(f"Final Classification loss: \t{c_valid_loss}")
        print(f"Final Reconstruction loss: \t{r_valid_loss}")

        if run_no < (cfg.n_runs - 1):
            train2lbl_idx = cal(active_dataset, step_acq_size, 
                                model, cfg.n_neighs, device)
            active_dataset.update(train2lbl_idx)
        
        
if __name__ == "__main__":
    main(cfg)