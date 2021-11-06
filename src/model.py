from collections import OrderedDict
from torch.nn import functional as F
from torch import nn
import torch
from pytorch_lightning.core.lightning import LightningModule
from src.data import ActiveDataset

from src.base_models.encoders import ENCODER_DICT
from src.base_models.bottlenecks import BOTTLENECK_DICT
from src.base_models.decoders import DECODER_DICT
from src.base_models.classifiers import CLASSIFIER_DICT


class Net(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        # x -> e
        self.encoder = ENCODER_DICT[cfg.enc['name']](cfg.enc)
        # e -> b, z
        self.bottleneck = BOTTLENECK_DICT[cfg.btk['name']](cfg.btk)
        # b -> r
        self.decoder = DECODER_DICT[cfg.dec['name']](cfg.dec)
        # b -> c
        self.classifier = CLASSIFIER_DICT[cfg.cls['name']](cfg.cls)

        self.c_loss = self.classifier.loss
        self.r_loss = self.decoder.loss

    def forward(self, x, classify=True, reconstruct=True):

        # Encoder
        x = self.encoder(x)

        # Bottleneck
        x, z = self.bottleneck(x)  

        if classify: # Classification
            c = self.classifier(z)
        else:
            c = None

        if reconstruct: # Reconstruction
            r = self.decoder(x) 
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

def train_epoch(model, active_data, optimizer, batch_size, device):

    model.train()
    torch.set_grad_enabled(True)
    
    iter_schedule = active_data.get_itersch(uniform=False)

    lbld_DL = active_data.get_loader('labeled', batch_size=batch_size)
    unlbld_DL = active_data.get_loader('unlabeled', batch_size=batch_size)

    lbl_iter = iter(lbld_DL)
    unlbl_iter = iter(unlbld_DL)

    n_epochs = len(active_data.trainset) // batch_size

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

        
def validate(model, active_data, batch_size, device):

    model.eval()
    torch.set_grad_enabled(False)

    valid_DL = active_data.get_loader('validation', batch_size=batch_size)

    c_losses = list()
    r_losses = list()
    
    for x, t in valid_DL:
            
        x = x.to(device)
        t = t.to(device)
        c = model.classify(x)
        loss = model.c_loss(c, t)
        c_losses.append(loss)

    for x, t in valid_DL:

        x = x.to(device)
        t = t.to(device)

        r = model.reconstruct(x)
        loss = model.r_loss(r.flatten(), x.flatten())
        r_losses.append(loss)
        
    mean_c_loss = torch.mean(torch.tensor(c_losses))
    mean_r_loss = torch.mean(torch.tensor(r_losses))

    return mean_c_loss, mean_r_loss
