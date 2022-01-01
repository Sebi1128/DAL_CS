import torchvision.transforms as transforms
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

# Constructing standard transformations for datasets

#transform_cifar10 = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#    ])
# as in https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/transforms/dataset_normalizations.py
transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    cifar10_normalization()
    ])

transform_cifar100=transforms.Compose([
    transforms.ToTensor(),
    cifar10_normalization()
    ])

transform_mnist=transforms.Compose([
    transforms.CenterCrop((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

transform_fmnist=transforms.Compose([
    transforms.CenterCrop((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,)),
    ])

TRANSFORMS_DICT = {
    'cifar10'   : transform_cifar10,
    'cifar100'  : transform_cifar100,
    'mnist'     : transform_mnist,
    'fmnist'    : transform_fmnist,
    }

