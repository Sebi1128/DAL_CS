from collections import OrderedDict
from torch.nn import functional as F
from torch import nn
import torch
from pytorch_lightning.core.lightning import LightningModule


# Dummy Encoder
class Base_Encoder(nn.Module):
    def __init__(self, cfg_enc):
        super().__init__()
        
        self.input_size = cfg_enc['input_size']
        self.encoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3,20,5, padding=2)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(20,64,5, padding=2)),
            ('relu2', nn.ReLU()),
        ]))

    def forward(self, x):
        return self.encoder(x)

ENCODER_DICT = {'base'  : Base_Encoder}