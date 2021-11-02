
from utils import load_cfg, pprint_color, get_parser
from tqdm import tqdm
import torch
from torch import optim
from src.data import ActiveDataset
from src.model import Net, train_epoch

def main(yaml_filepath):
    """Example."""
    cfg = load_cfg(yaml_filepath)

    # Print the configuration - just to make sure that you loaded what you
    # wanted to load
    pprint_color(cfg)

    ad = ActiveDataset('cifar10', transform='cifar10')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    pbar = tqdm(range(3))
    for k in pbar:
        c_loss, r_loss = train_epoch(model, ad, optimizer, batch_size=50, device=device)
        pbar.set_description(f"C|R Loss: {c_loss:.5f}|{r_loss:.5f}", refresh=True)

    print(f"Final Classification loss: \t{c_loss}")
    print(f"Final Reconstruction loss: \t{r_loss}")
        
if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args.filename)