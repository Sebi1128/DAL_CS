from collections import OrderedDict
from torch.nn import functional as F
from torch import nn
import torch
from pytorch_lightning.core.lightning import LightningModule
from src.data import ActiveDataset

class Net(LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3,20,5, padding=2)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(20,64,5, padding=2)),
            ('relu2', nn.ReLU()),
        ]))

        self.bnck1 = nn.Linear(32*32*64, 256)
        self.relu1 = nn.ReLU()
        self.bnck2 = nn.Linear(256, 32*32*64)
        self.relu2 = nn.ReLU()
        
        self.decoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64,20,5, padding=2)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(20,3,5, padding=2)),
        ]))
        
        self.classifier = nn.Sequential(OrderedDict([
            ('linr1', nn.Linear(256, 64)),
            ('relu1', nn.ReLU()),
            ('linr2', nn.Linear(64, 10)),
        ]))

        self.c_loss = nn.NLLLoss()
        self.r_loss = nn.MSELoss()

    def forward(self, x, classify=True, reconstruct=True):
        batch_size, channels, height, width = x.size()

        # Encoder
        x = self.encoder(x)
        x = x.view(batch_size, -1)

        # Bottleneck
        x = self.bnck1(x)
        z = self.relu1(x) # Latent Space
        
        x = self.bnck2(z)
        x = self.relu2(x)

        if classify:
            c = self.classifier(z)
            c = F.log_softmax(c, dim=1) # Classification
        else:
            c = None

        if reconstruct:
            r = x.view(batch_size, -1, height, width)
            r = self.decoder(r) # Reconstruction
        else:
            r = None

        return z, r, c

    def latent(self, x):
        z, _, _ = self.forward(x, classify=False, reconstruct=False)
        return z

    def reconstruct(self, x):
        _, r, _ = self.forward(x, classify=False)
        return r

    def classify(self, x):
        _, _, c = self.forward(x, reconstruct=False)
        return c

def train_epoch(model, activData, optimizer, batch_size, device):

    model.train()
    torch.set_grad_enabled(True)
    
    iter_schedule = activData.get_itersch(uniform=False)

    lbld_DL = activData.get_loader('labeled', batch_size=batch_size)
    unlbld_DL = activData.get_loader('unlabeled', batch_size=batch_size)

    lbl_iter = iter(lbld_DL)
    unlbl_iter = iter(unlbld_DL)

    n_epochs = len(activData.trainset) // batch_size

    c_losses = list()
    r_losses = list()
    
    for is_labeled in iter_schedule[:n_epochs]:

        if is_labeled:
            x, t = next(lbl_iter)
            x = x.to(device)
            t = t.to(device)
            c = model.classify(x)
            loss = model.c_loss(c, t)
            c_losses.append(loss)
        else:
            x, _ = next(unlbl_iter)
            x = x.to(device)
            r = model.reconstruct(x)
            loss = model.r_loss(r.flatten(), x.flatten())
            r_losses.append(loss)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mean_c_loss = torch.mean(torch.tensor(c_losses))
    mean_r_loss = torch.mean(torch.tensor(r_losses))

    return mean_c_loss, mean_r_loss

        


    
