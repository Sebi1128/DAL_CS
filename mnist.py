import os
from typing import Optional
import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms


class BaselineClassifier(pl.LightningModule):

    def __init__(self):
        super(BaselineClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=4 * 4 * 64, out_features=10)

    def forward(self, x):
        # conv 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxPool1(x)
        # conv 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxPool2(x)
        # fc 1
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)

        x = F.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        acc = FM.accuracy(logits, y)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor)
        MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor)

    def setup(self, stage: Optional[str] = None):
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_train = MNIST(os.getcwd(), train=True, download=False, transform=transform)
        mnist_test = MNIST(os.getcwd(), train=False, download=False, transform=transform)

        mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)

def main():
    data_module = MNISTDataModule()
    model = BaselineClassifier()
    trainer = pl.Trainer(max_epochs=20)

    trainer.fit(model, data_module)
    print("Finished training")
    trainer.test(test_dataloaders=data_module)


if __name__ == "__main__":
    main()