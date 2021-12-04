from collections import OrderedDict
from torch.nn import functional as F
from torch import nn
import torch
from torch.nn.modules.batchnorm import BatchNorm2d


class VAAL_Encoder(nn.Module):
    def __init__(self, cfg_enc):
        super(VAAL_Encoder, self).__init__()

        hidden_dims = [128, 256, 512, 1024]
        in_channels = 3
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
        x = torch.flatten(x, 1)
        return x


class Base_Encoder(nn.Module):
    def __init__(self, cfg_enc):
        super(Base_Encoder, self).__init__()

        self.input_size = cfg_enc['input_size']
        self.encoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 20, 5, padding=2)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(20, 64, 5, padding=2)),
            ('relu2', nn.ReLU()),
        ]))

    def forward(self, x):
        return self.encoder(x)


ENCODER_DICT = {"base": Base_Encoder, "vaal": VAAL_Encoder}
