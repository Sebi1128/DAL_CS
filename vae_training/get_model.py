from pl_bolts.models.autoencoders import VAE
from torch.nn import Conv2d
import torch

def get_model(cfg):
    vae = VAE(input_height=cfg['im_height'], latent_dim=cfg['latent_dim'])

    if cfg['use_pretrained_cifar_enc']:
        pretrained_vae = VAE(input_height=cfg['im_height'])
        pretrained_vae = pretrained_vae.from_pretrained('cifar10-resnet18')
        pretrained_vae.freeze()
        vae.encoder = pretrained_vae.encoder

    if cfg['im_channels'] != 3:
        vae.encoder.conv1 = Conv2d(cfg['im_channels'], 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        vae.decoder.conv1 = Conv2d(64, cfg['im_channels'], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    if cfg['load_from_checkpoint'] is not None:
        checkpoint = torch.load(cfg['load_from_checkpoint'], map_location='cpu')
        vae.load_state_dict(checkpoint['state_dict'])
    return vae