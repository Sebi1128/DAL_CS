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

import yaml
import argparse
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torchvision
from torch.utils.data import DataLoader
from pl_bolts.datamodules import CIFAR10DataModule, BinaryMNISTDataModule, FashionMNISTDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from get_model import get_model


def get_dataset(cfg):
    res = None

    transforms = []
    if cfg['do_random_crop']:
        transforms += [torchvision.transforms.RandomCrop(32, padding=4)]
    if cfg['do_horizontal_flip']:
        transforms += [torchvision.transforms.RandomHorizontalFlip()]

    if cfg['name'] == 'CIFAR10':
        train_transforms = torchvision.transforms.Compose(
            transforms + [
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

        cifar10_dm = CIFAR10DataModule(
            data_dir=cfg['data_dir'],
            batch_size=cfg['batch_size'],
            num_workers=cfg['n_workers'],
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            val_transforms=test_transforms,
        )
        res = {'train_dataloaders': cifar10_dm}

    elif cfg['name'] == 'CIFAR100':
        train_transforms = torchvision.transforms.Compose(
            transforms + [
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

        cifar100_train = torchvision.datasets.CIFAR100(
            root=cfg['data_dir'],
            train=True,
            download=True,
            transform=train_transforms
        )
        cifar100_test = torchvision.datasets.CIFAR100(
            root=cfg['data_dir'],
            train=False,
            download=True,
            transform=test_transforms
        )

        train_loader = DataLoader(
            cifar100_train,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['n_workers']
        )
        test_loader = DataLoader(
            cifar100_test,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['n_workers']
        )
        res = {'train_dataloaders': train_loader, 'val_dataloaders': test_loader}

    elif cfg['name'] == 'MNIST':
        train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.CenterCrop((32, 32))
            ] + transforms + [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.CenterCrop((32, 32)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        mnist_dm = BinaryMNISTDataModule(
            data_dir=cfg['data_dir'],
            batch_size=cfg['batch_size'],
            num_workers=cfg['n_workers'],
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            val_transforms=test_transforms,
        )
        res = {'train_dataloaders': mnist_dm}
    elif cfg['name'] == 'Fashion_MNIST':
        train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.CenterCrop((32, 32))
            ] + transforms + [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (1,)),
            ]
        )

        test_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.CenterCrop((32, 32)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (1,)),
            ]
        )

        fashion_mnist_dm = FashionMNISTDataModule(
            data_dir=cfg['data_dir'],
            batch_size=cfg['batch_size'],
            num_workers=cfg['n_workers'],
            train_transforms=train_transforms,
            test_transforms=test_transforms,
            val_transforms=test_transforms,
        )
        res = {'train_dataloaders': fashion_mnist_dm}

    return res

def train(cfg):
    run_name = cfg['run_name']
    experiment_name = run_name + datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")

    wandb_logger = WandbLogger(project="VAE_training")
    wandb_logger.experiment.name = experiment_name + '_' + wandb_logger.experiment.id
    wandb_logger.experiment.config.update(cfg)

    callbacks = []
    callbacks += [EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min")]
    callbacks += [ModelCheckpoint(
        monitor='val_loss',
        dirpath='vae_checkpoints/{}'.format(experiment_name),
        filename='sample-mnist-{epoch:02d}-{val_loss:.2f}'
    )]
    callbacks += [LearningRateMonitor(logging_interval="step")]

    trainer = Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=cfg['n_epochs'],
        accelerator='auto',
        devices='auto',
        callbacks=callbacks,
        logger=wandb_logger,
        strategy=cfg['strategy']
    )

    vae = get_model(cfg['vae'])

    dataset = get_dataset(cfg['dataset'])

    trainer.fit(vae, **dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config file', default='configs/CIFAR10_vae.yaml')
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    train(config)