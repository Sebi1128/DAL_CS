from collections import OrderedDict
from torch.nn import functional as F
from torch import nn
import torch
from pytorch_lightning.core.lightning import LightningModule

# Dummy Classifier
class Base_Classifier(nn.Module):
    def __init__(self, cfg_cls):
        super().__init__()
        
        self.output_size = cfg_cls['output_size']
        self.classifier = nn.Sequential(OrderedDict([
            ('linr1', nn.Linear(256, 64)),
            ('relu1', nn.ReLU()),
            ('linr2', nn.Linear(64, self.output_size)),
        ]))

        self.loss = nn.NLLLoss() 
        # should define a loss corresponding to the output

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        c = F.log_softmax(x, dim=1)
        return c

CLASSIFIER_DICT = {'base'  : Base_Classifier}