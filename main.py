
from utils import load_cfg, pprint_color, get_parser
from tqdm import tqdm
import torch
from torch import optim
from src.data import ActiveDataset
from src.model import Net, train_epoch
from config import cfg
import wandb

def main(cfg):

    ad = ActiveDataset(cfg.dataset)

    if cfg.device.lower() == 'gpu':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f"Device is {device}")

    model = Net().to(device)
    if cfg.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    elif cfg.optimizer.lower() == 'sgd':
        optimizer =optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=cfg.momentum)
        
    pbar = tqdm(range(cfg.n_epochs))
    for k in pbar:
        c_loss, r_loss = train_epoch(model, ad, optimizer, 
                                     batch_size=cfg.batch_size, device=device)
        pbar.set_description(f"C|R Loss: {c_loss:.5f}|{r_loss:.5f}", refresh=True)
        wandb.log({"r_loss": r_loss})
        wandb.log({"c_loss": c_loss})

    print(f"Final Classification loss: \t{c_loss}")
    print(f"Final Reconstruction loss: \t{r_loss}")
        
if __name__ == "__main__":
    main(cfg)