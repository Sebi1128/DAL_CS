from collections import OrderedDict
from torch.nn import functional as F
from torch import nn
import torch
from torch.nn.modules.batchnorm import BatchNorm2d


class VAAL_Encoder(nn.Module):
    def __init__(self, cfg_enc):
        super(VAAL_Encoder, self).__init__()

        hidden_dims = cfg_enc['hidden_dims']
        in_channels = cfg_enc['input_size'][0]
        layers = []
        for i, dim in enumerate(hidden_dims):
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

    def forward(self, x):
        x = self.encoder(x)
        #x = torch.flatten(x, 1)
        return x


class Base_Encoder(nn.Module):
    def __init__(self, cfg_enc):
        super(Base_Encoder, self).__init__()
        hidden_dims = cfg_enc['hidden_dims']
        in_channels = cfg_enc['input_size'][0]
        
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

    def forward(self, x):
        x = self.encoder(x)
        #x = torch.flatten(x, 1)
        return x


ENCODER_DICT = {"base": Base_Encoder, "vaal": VAAL_Encoder}
