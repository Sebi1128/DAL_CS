import torch.nn as nn
import torch.nn.init as init


def kaiming_init(net):
    """taken from https://github.com/sinhasam/vaal/blob/master/model.py"""
    for module in net.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            init.kaiming_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.weight.data.fill_(1)
            if module.bias is not None:
                module.bias.data.fill_(0)

