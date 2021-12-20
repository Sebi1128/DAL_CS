from collections import OrderedDict
import math

from torch.nn import functional as F
from torch import nn
import torch
from functools import partial
from pytorch_lightning.core.lightning import LightningModule


class Base_Decoder(nn.Module):
    def __init__(self, cfg_dec):
        super(Base_Decoder, self).__init__()
        out_channels = cfg_dec['input_size'][0]
        image_size = cfg_dec['input_size'][1] * cfg_dec['input_size'][2]
        hidden_dims = list(reversed(cfg_dec['hidden_dims']))
        kld_weight = cfg_dec['kld_weight']

        self.feature_dim = cfg_dec['feature_dim']
        
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU(),
                )
            )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1
                )
            )
        )

        self.decoder = nn.Sequential(*layers)
        self.loss = partial(vae_loss, kld_weight=kld_weight, image_size=image_size)

    def forward(self, x):
        x = x.view(-1, *self.feature_dim)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


class VAAL_Decoder(nn.Module):
    def __init__(self, cfg_dec):
        super(VAAL_Decoder, self).__init__()
        self.kld_weight = cfg_dec['kld_weight']

        image_size = math.prod(cfg_dec['input_size'][1:])
        hidden_dims = cfg_dec['hidden_dims'][::-1]
        out_channels = cfg_dec['input_size'][0]

        self.feature_dim = cfg_dec['feature_dim']
        
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU(),
                )
            )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    out_channels,
                    kernel_size=2,
                    stride=2,
                )
            )
        )

        self.decoder = nn.Sequential(*layers)
        self.loss = partial(vae_loss, kld_weight=self.kld_weight, image_size=image_size)

    def forward(self, x):
        x = x.view(-1, *self.feature_dim)
        x = self.decoder(x)
        x = torch.sigmoid(x) 
        #REVIEW Why?
        return x


def mse_loss(recon, x, *args, **kwargs):
    loss = F.mse_loss(recon, x)
    return {'loss': loss}


def vae_loss(recon, x, *args, **kwargs):
    mu = args[0]
    logvar = args[1]
    kld_weight = kwargs.get('kld_weight', 1)
    spatial_size = kwargs['image_size']
    recon_loss = F.binary_cross_entropy(recon.view(-1, spatial_size), x.view(-1, spatial_size), reduction='mean')
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar -
                          mu ** 2 - logvar.exp(), dim=1), dim=0)
    loss = recon_loss + kld_weight * kld_loss
    return {'loss': loss, 'reconstruction_loss': recon_loss}


DECODER_DICT = {"base": Base_Decoder, "vaal": VAAL_Decoder}
