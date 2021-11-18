from collections import OrderedDict
from torch.nn import functional as F
from torch import nn
import torch
from pytorch_lightning.core.lightning import LightningModule


class Base_Decoder(nn.Module):
    def __init__(self, cfg_dec):
        super(Base_Decoder, self).__init__()

        self.output_size = cfg_dec["output_size"]
        self.decoder = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(64, 20, 5, padding=2)),
                    ("relu1", nn.ReLU()),
                    ("conv2", nn.Conv2d(20, 3, 5, padding=2)),
                ]
            )
        )

        self.loss = nn.MSELoss()
        # should define a loss corresponding to the output

    def forward(self, x):
        x = x.view(x.size(0), -1, self.output_size[0], self.output_size[1])
        return self.decoder(x)


class VAAL_Decoder(nn.Module):
    def __init__(self, cfg_dec):
        super(VAAL_Decoder, self).__init__()

        hidden_dims = [1024, 512, 256, 128]
        out_channels = 3
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
                    kernel_size=1
                )
            )
        )

        self.decoder = nn.Sequential(*layers)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = x.view(-1, 1024, 4, 4)
        x = self.decoder(x)
        return x


DECODER_DICT = {"base": Base_Decoder, "vaal": VAAL_Decoder}
