from pytorch_lightning.core.lightning import LightningModule

from src.base_models.encoders import ENCODER_DICT
from src.base_models.bottlenecks import BOTTLENECK_DICT
from src.base_models.decoders import DECODER_DICT
from src.base_models.classifiers import CLASSIFIER_DICT
from src.base_models.samplers import SAMPLER_DICT

class Net(LightningModule):
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

        self.c_loss = self.classifier.loss
        self.r_loss = self.decoder.loss

    def forward(self, x, classify=True, reconstruct=True):

        # Encoder
        x = self.encoder(x)

        # Bottleneck
        x, z = self.bottleneck(x)  

        if classify: # Classification
            c = self.classifier(z)
        else:
            c = None

        if reconstruct: # Reconstruction
            r = self.decoder(x) 
        else:
            r = None

        return z, r, c

    def latent(self, x):
        z, _, _ = self.forward(x, classify=False, reconstruct=False)
        return z

    def reconstruct(self, x):
        _, r, _ = self.forward(x, classify=False)
        return r

    def classify(self, x):
        _, _, c = self.forward(x, reconstruct=False)
        return c


class Sampler():
    def __init__(self, cfg_smp):
        self.base_sampler = SAMPLER_DICT[cfg_smp['name']](cfg_smp)

    def __call__(self, active_data, acq_size, model, device):
        return self.base_sampler(active_data, acq_size, model, device)

