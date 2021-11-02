#%%

from collections import OrderedDict
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

class Net(LightningModule):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3,20,5, padding=2)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(20,64,5, padding=2)),
            ('relu2', nn.ReLU()),
        ]))

        self.bnck1 = nn.Linear(32*32*64, 256)
        self.relu1 = nn.ReLU()
        self.bnck2 = nn.Linear(256, 32*32*64)
        self.relu2 = nn.ReLU()
        
        self.decoder = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(64,20,5, padding=2)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(20,3,5, padding=2)),
        ]))
        
        self.classifier = nn.Sequential(OrderedDict([
            ('linr1', nn.Linear(256, 64)),
            ('relu1', nn.ReLU()),
            ('linr2', nn.Linear(64, 10)),
        ]))

    def forward(self, x, classify=True, reconstruct=True):
        batch_size, channels, height, width = x.size()

        # Encoder
        x = self.encoder(x)
        x = x.view(batch_size, -1)

        # Bottleneck
        x = self.bnck1(x)
        z = self.relu1(x) # Latent Space
        
        x = self.bnck2(z)
        x = self.relu2(x)

        if classify:
            c = self.classifier(z)
            c = F.log_softmax(c, dim=1) # Classification
        else:
            c = None

        if reconstruct:
            r = x.view(batch_size, -1, height, width)
            r = self.decoder(r) # Reconstruction
        else:
            None

        return z, r, c      

