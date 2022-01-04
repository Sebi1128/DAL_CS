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

from collections import OrderedDict
from torch import nn
import torch
from .model_utils import kaiming_init


class VAAL_Encoder(nn.Module):
    """The encoder described in the """
    def __init__(self, cfg_enc):
        super(VAAL_Encoder, self).__init__()

        hidden_dims = [128, 256, 512, 1024]
        in_channels = cfg_enc['in_channels']
        layers = []
        for dim in hidden_dims:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, dim, kernel_size=4, stride=2, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(),
                )
            )
            in_channels = dim

        self.encoder = nn.Sequential(*layers)
        kaiming_init(self)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return x


class Base_Encoder(nn.Module):
    """A dummy encoder class with two convolutional layers for testing"""
    def __init__(self, cfg_enc):
        super(Base_Encoder, self).__init__()

        self.input_size = cfg_enc['input_size']
        in_channels = cfg_enc['in_channels']
        self.encoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, 20, 5, padding=2)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(20, 64, 5, padding=2)),
            ('relu2', nn.ReLU()),
        ]))

    def forward(self, x):
        return self.encoder(x)

# dictionary containing encoder classes
ENCODER_DICT = {"base": Base_Encoder, "vaal": VAAL_Encoder}
