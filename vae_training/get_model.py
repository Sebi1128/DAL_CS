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

from pl_bolts.models.autoencoders import VAE
from torch.nn import Conv2d
import torch

def get_model(cfg):
    vae = VAE(input_height=cfg['im_height'], latent_dim=cfg['latent_dim'])

    if cfg['use_pretrained_cifar_enc']:
        pretrained_vae = VAE(input_height=cfg['im_height'])
        pretrained_vae = pretrained_vae.from_pretrained('cifar10-resnet18')
        pretrained_vae.freeze() # freezing the parameters since we do not train but only use
        vae.encoder = pretrained_vae.encoder

    if cfg['im_channels'] != 3: # the standard version takes input channels 3 due to RGB
        vae.encoder.conv1 = Conv2d(cfg['im_channels'], 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        vae.decoder.conv1 = Conv2d(64, cfg['im_channels'], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    if cfg['load_from_checkpoint'] is not None: # if there is a specified load for parameters
        checkpoint = torch.load(cfg['load_from_checkpoint'], map_location='cpu')
        vae.load_state_dict(checkpoint['state_dict'])
    return vae