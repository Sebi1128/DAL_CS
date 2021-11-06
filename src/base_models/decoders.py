from collections import OrderedDict
from torch.nn import functional as F
from torch import nn
import torch
from pytorch_lightning.core.lightning import LightningModule

# Dummy Decoder
class Base_Decoder(nn.Module):
    def __init__(self, cfg_dec):
        super().__init__()

        self.output_size = cfg_dec['output_size']
        self.decoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64,20,5, padding=2)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(20,3,5, padding=2)),
        ]))

        self.loss = nn.MSELoss() 
        # should define a loss corresponding to the output

    def forward(self, x):
        x = x.view(x.size(0), -1, 
                   self.output_size[0],
                   self.output_size[1])
        return self.decoder(x)

DECODER_DICT = {'base'  : Base_Decoder}