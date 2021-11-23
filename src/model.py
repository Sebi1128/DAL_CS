import torch
from torch import nn
#from torchsummary import summary

from src.base_models.encoders import ENCODER_DICT
from src.base_models.bottlenecks import BOTTLENECK_DICT
from src.base_models.decoders import DECODER_DICT
from src.base_models.classifiers import CLASSIFIER_DICT
from src.base_models.samplers import SAMPLER_DICT

class Net(nn.Module):
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
        #summary(self.classifier, input_size=(3, 32, 32))

        self.c_loss = self.classifier.loss
        self.r_loss = self.decoder.loss

    def forward(self, x, classify=True, reconstruct=True):
        # Keep for classification
        x_save = x

        # Encoder
        x = self.encoder(x)

        # Bottleneck
        x, latent = self.bottleneck(x)
        z = latent[0]

        if classify: # Classification
            c = self.classifier(x_save, z, x)
        else:
            c = None

        if reconstruct: # Reconstruction
            r = self.decoder(x) 
        else:
            r = None

        return latent, r, c

    def latent(self, x):
        latent, _, _ = self.forward(x, classify=False, reconstruct=False)
        return latent[0]

    def latent_full(self, x):
        """ returns latent mean and log var"""
        latent, _, _ = self.forward(x, classify=False, reconstruct=False)
        mu = latent[1]
        logvar = latent[2]
        return torch.stack((mu, logvar), -1)

    def reconstruct(self, x):
        latent, r, _ = self.forward(x, classify=False)
        return r, latent

    def classify(self, x):
        _, _, c = self.forward(x, reconstruct=False)
        return c
