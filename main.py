import os
from typing import Optional
import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision import transforms

class SimpleClassifier(pl.LightningModule):

    def __init__(self):
        super(SimpleClassifier, self).__init__()

        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

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


class FastBaselineClassifier(pl.LightningModule):

    def __init__(self):
        super(FastBaselineClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, x):
        #batch_size, channels, height, width = x.size()

        # conv 1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # conv 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # fc1
        x = x.reshape(-1, 12 * 4 * 4)
        x = self.fc1(x)
        x = F.relu(x)
        # fc2
        x = self.fc2(x)
        x = F.relu(x)
        # output
        x = self.out(x)

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


class BaselineClassifier(pl.LightningModule):

    def __init__(self):
        super(BaselineClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm4 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=512)
        self.batchnorm5 = nn.BatchNorm1d(512)
        self.out = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        #batch_size, channels, height, width = x.size() 

        # conv 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)
        # conv 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.dropout(x, p=0.25)

        # conv 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)
        # conv 4
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchnorm4(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.dropout(x, p=0.25)

        # fc1
        x = x.reshape(-1, 64 * 4 * 4)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.batchnorm5(x)
        x = F.dropout(x, p=0.5)

        # output
        x = self.out(x)

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


class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        FashionMNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor)
        FashionMNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor)

    def setup(self, stage: Optional[str] = None):
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_train = FashionMNIST(os.getcwd(), train=True, download=False, transform=transform)
        mnist_test = FashionMNIST(os.getcwd(), train=False, download=False, transform=transform)

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
    data_module = FashionMNISTDataModule()
    model = FastBaselineClassifier()
    trainer = pl.Trainer(max_epochs=20)

    trainer.fit(model, data_module)
    print("Finished training")
    trainer.test(test_dataloaders=data_module)


if __name__ == "__main__":
    main()