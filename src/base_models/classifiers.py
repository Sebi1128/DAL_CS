from collections import OrderedDict
from torch.nn import functional as F
from torch import nn
import torch


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class Classifier_VAAL(nn.Module):

    def __init__(self, cfg_cls):
        super(Classifier_VAAL, self).__init__()
        self.cfg_cls = cfg_cls
        self.output_size = self.cfg_cls['output_size']
        # Make CNN layers of VGG16 with batch normalization
        self.features = make_layers(cfgs['D'], batch_norm=True, 
                                    in_channels=self.cfg_cls['in_channels'])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        if self.cfg_cls['name'] == 'vaal_with_latent':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7 +  self.cfg_cls['z_dim'], 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, self.output_size),
            )
        elif  self.cfg_cls['name'] == 'vaal':
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, self.output_size),
            )
        else:
            raise NotImplementedError()

        self._initialize_weights()

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, z, x_bottleneck):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if  self.cfg_cls['name'] == 'vaal_with_latent':
            z = torch.flatten(z, 1)
            x = self.classifier(torch.cat((x, z), dim=1))
        elif self.cfg_cls['name'] == 'vaal':
            x = self.classifier(x)
        else:
            raise NotImplementedError()
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg_layer, batch_norm=False, in_channels=3):
    layers = []
    for v in cfg_layer:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# Dummy Classifier
class Base_Classifier_Dummy(nn.Module):
    def __init__(self, cfg_cls):
        super().__init__()
        self.cfg_cls = cfg_cls
        self.output_size = self.cfg_cls['output_size']
        self.classifier = nn.Sequential(OrderedDict([
            ('linr1', nn.Linear(256, 64)),
            ('relu1', nn.ReLU()),
            ('linr2', nn.Linear(64, self.output_size)),
        ]))

        self.loss = nn.NLLLoss() 
        # should define a loss corresponding to the output

    def forward(self, x, z, x_bottleneck):
        z = z.view(z.size(0), -1)
        z = self.classifier(z)
        c = F.log_softmax(z, dim=1)
        return c

CLASSIFIER_DICT = {
    'vaal': Classifier_VAAL,
    'vaal_with_latent': Classifier_VAAL,
    'base': Base_Classifier_Dummy
}