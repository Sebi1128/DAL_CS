import torchvision.transforms as transforms

# Constructing standard transformations for datasets

transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_cifar100=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
    ])

transform_mnist = transforms.Compose([
    transforms.ToTensor()
])

transform_fashion_mnist= transforms.Compose([
    transforms.ToTensor()
])

TRANSFORMS_DICT = {
    'cifar10'   : transform_cifar10,
    'cifar100'  : transform_cifar100,
    'mnist'     : transform_mnist,
    'fashion_mnist' : transform_fashion_mnist
    }

