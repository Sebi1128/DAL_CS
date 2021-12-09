from collections import OrderedDict
from torch.nn import functional as F
from torch import nn
import torch
import numpy


class Bottleneck(nn.Module):
    def __init__(self, cfg_btk):
        super(Bottleneck, self).__init__()

    def forward(self, x, latent, output):
        raise NotImplementedError()

    def latent(self, x):
        _, latent = self.forward(x, latent=True, output=False)
        return latent

    def output(self, x):
        y, _ = self.forward(x, latent=False, output=True)
        return y


class VAE_Bottleneck(Bottleneck):
    def __init__(self, cfg_btk):
        super(VAE_Bottleneck, self).__init__(cfg_btk)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        if mu.is_cuda:
            std.cuda()
            eps.cuda()
        z = mu + (eps * std)
        return z


class Base_Bottleneck(VAE_Bottleneck):
    def __init__(self, cfg_btk):
        super(Base_Bottleneck, self).__init__(cfg_btk)
        z_dim = cfg_btk['z_dim']
        feature_dim = numpy.prod(cfg_btk['feature_dim'])

        self.fc_mu = nn.Linear(feature_dim, z_dim)
        self.fc_logvar = nn.Linear(feature_dim, z_dim)

        self.out = nn.Linear(z_dim, feature_dim)

    def forward(self, x, latent=True, output=True):
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        y = None
        if output:
            y = self.out(z)

        return y, [z, mu, logvar]


class VAAL_Bottleneck(VAE_Bottleneck):
    def __init__(self, cfg_btk):
        super(VAAL_Bottleneck, self).__init__(cfg_btk)
        self.z_dim = cfg_btk['z_dim']
        self.fc_mu = nn.Linear(1024 * 2 * 2, self.z_dim)
        self.fc_logvar = nn.Linear(1024 * 2 * 2, self.z_dim)

        # VAAL uses some strange decoder
        self.out = nn.Linear(self.z_dim, 1024 * 4 * 4)

    def forward(self, x, latent=True, output=True):
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        y = None
        if output:
            y = self.out(z)

        return y, [z, mu, logvar]


BOTTLENECK_DICT = {"base": Base_Bottleneck, "vaal": VAAL_Bottleneck}
