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

import torch
from torch import nn
import torch.optim as optim
from src.base_models.vae_wrapper import VAEWrapper

from src.base_models.encoders import ENCODER_DICT
from src.base_models.bottlenecks import BOTTLENECK_DICT
from src.base_models.decoders import DECODER_DICT
from src.base_models.classifiers import CLASSIFIER_DICT
from src.base_models.samplers import SAMPLER_DICT

class Net(nn.Module):
    """
    General model that uses base models defined in src/base_models
    The structure of the Net consists of 4 (+1) units:
        - Encoder: Which serves as the encoder part of a defined variational autoencoder 
        - Decoder: Which serves as the decoder part of a defined variational autoencoder
        - Bottleneck: The connection between encoder and decoder, which outputs latent space 
        - Classifier: The method classifies the samples 
                      using sample itself, latent space, and bottleneck output
    """
    def __init__(self, cfg):
        super().__init__()
        self.use_off_the_shelf_vae = cfg.embedding['use_off_the_shelf_vae']
        if self.use_off_the_shelf_vae: # Using pretrained Variational Autoencoder
            vae = VAEWrapper(cfg)
            self.encoder = vae.get_encoder()
            # e -> b, z
            self.bottleneck = vae.get_bottleneck()
            # b -> r
            self.decoder = vae.get_decoder()
        else: # Using Variational Autoencoder with online training
            # x -> e
            self.encoder = ENCODER_DICT[cfg.enc['name']](cfg.enc)
            # e -> b, z
            self.bottleneck = BOTTLENECK_DICT[cfg.btk['name']](cfg.btk)
            # b -> r
            self.decoder = DECODER_DICT[cfg.dec['name']](cfg.dec)
            # b -> c
        self.classifier = CLASSIFIER_DICT[cfg.cls['name']](cfg.cls)

        # defining two types of loss functions: classification and reconstruction
        self.c_loss = self.classifier.loss
        self.r_loss = self.decoder.loss

        # defining optimizer and learning rate both for reconstruction and classifier
        optimizers = {'adam': optim.Adam, 'sgd': optim.SGD}
        self.optimizer_embedding = optimizers[cfg.embedding['optimizer'].lower()](
            list(self.encoder.parameters()) + list(self.bottleneck.parameters()) + list(self.decoder.parameters()),
            lr=cfg.embedding['lr']
        )
        self.optimizer_classifier = optimizers[cfg.cls['optimizer'].lower()](
            self.classifier.parameters(),
            lr=cfg.cls['lr'],
            weight_decay=5e-4,
            momentum=0.9
        )

    def forward(self, x, classify=True, reconstruct=True):
        # Keep for classification
        x_save = x

        # Encoder
        x = self.encoder(x)

        # Bottleneck
        x, latent = self.bottleneck(x)
        mu = latent[1]

        if classify: # Classification
            c = self.classifier(x_save, mu, x)
        else:
            c = None

        if reconstruct: # Reconstruction
            r = self.decoder(x)
        else:
            r = None

        return latent, r, c

    def latent(self, x):
        """
        Returns the latent variable 
        constructed by the encoder part of the defined VAE
        """
        latent, _, _ = self.forward(x, classify=False, reconstruct=False)
        z = latent[0]
        return z

    def latent_mu(self, x):
        """
        Returns the mean of the latent variable 
        constructed by the encoder part of the defined VAE
        """
        latent, _, _ = self.forward(x, classify=False, reconstruct=False)
        mu = latent[1]
        return mu

    def latent_param(self, x):
        """
        Returns the mean and logvar of the latent variable 
        constructed by the encoder part of the defined VAE
        """
        latent, _, _ = self.forward(x, classify=False, reconstruct=False)
        mu = latent[1]
        logvar = latent[2]
        return torch.stack([mu, logvar], -1)


    def reconstruct(self, x):
        """
        Returns the reconstruction result by the defined VAE
        """
        latent, r, _ = self.forward(x, classify=False)
        return r, latent

    def classify(self, x):
        """
        Returns the classification result
        """
        _, _, c = self.forward(x, reconstruct=False)
        return c
