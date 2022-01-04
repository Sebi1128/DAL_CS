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

