from collections import OrderedDict
from torch.nn import functional as F
from torch import nn
import torch
from .model_utils import kaiming_init


class Bottleneck(nn.Module):
    """Parent class for Bottleneck"""
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


class Base_Bottleneck(Bottleneck):
    """A dummy bottleneck class with two fully connected layers for testing"""
    def __init__(self, cfg_btk):
        super(Base_Bottleneck, self).__init__(cfg_btk)

        # REVIEW Z should be before or ReLU or Not?
        self.z_dim = cfg_btk['z_dim']
        self.linr1 = nn.Linear(32 * 32 * 64, self.z_dim)
        self.relu1 = nn.ReLU()
        self.linr2 = nn.Linear(self.z_dim, 32 * 32 * 64)
        self.relu2 = nn.ReLU()

    def forward(self, x, latent=True, output=True):

        x = x.view(x.size(0), -1)
        z = self.linr1(x)
        if output:
            x = self.relu1(z)
            x = self.linr2(x)
            y = self.relu2(x)
        else:
            y = None

        return y, z


class VAAL_Bottleneck(Bottleneck):
    
    def __init__(self, cfg_btk):
        super(VAAL_Bottleneck, self).__init__(cfg_btk)
        self.z_dim = cfg_btk['z_dim']
        self.fc_mu = nn.Linear(1024 * 2 * 2, self.z_dim)
        self.fc_logvar = nn.Linear(1024 * 2 * 2, self.z_dim)

        self.out = nn.Linear(self.z_dim, 1024 * 4 * 4) # VAAL uses some strange decoder
        kaiming_init(self)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn(*mu.size())
        if mu.is_cuda:
            std = std.cuda()
            eps = eps.cuda()
        z = mu + (eps * std)
        return z

    def forward(self, x, latent=True, output=True):
        mu, logvar = self.fc_mu(x), self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        y = None
        if output:
            y = self.out(z)

        return y, [z, mu, logvar]


BOTTLENECK_DICT = {"base": Base_Bottleneck, "vaal": VAAL_Bottleneck}
