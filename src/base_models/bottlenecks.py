from collections import OrderedDict
from torch.nn import functional as F
from torch import nn
import torch
from pytorch_lightning.core.lightning import LightningModule

# Dummy Bottleneck
class Base_Bottleneck(nn.Module):
    def __init__(self, cfg_btk):
        super().__init__()

        #REVIEW Z should be before or ReLU or Not?
        
        self.linr1 = nn.Linear(32*32*64, 256)
        self.relu1 = nn.ReLU()
        self.linr2 = nn.Linear(256, 32*32*64)
        self.relu2 = nn.ReLU()

    def forward(self, x, latent=True, output=True):

        x = x.view(x.size(0), -1)
        z = self.linr1(x)
        if output:
            x = self.relu1(z)
            x = self.linr2(x)
            y = self.relu2(x)
        else:
            y = None
        
        return y, z

    def latent(self, x):
        _, z = self.forward(x, latent=True, output=False)
        return z

    def output(self, x):
        y, _ = self.forward(x, latent=False, output=True)
        return y
        

BOTTLENECK_DICT = {'base'  : Base_Bottleneck}