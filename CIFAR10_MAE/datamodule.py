import os
import lightning as L
from lightning import LightningDataModule
import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import CIFAR10

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

class DataRepeater(object):
    def __init__(self, dataset, size=None):
        self.data = dataset
        self.size = size
        if self.size is None:
            self.size = len(dataset)
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        i = idx % len(self.data)
        return self.data[i]


class Cifar10DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size = 64,
        num_workers = 4,
        pin_memory = True,
        size = 224,
        augment = True,
        num_samples = None,
    ):
        super().__init__()
        self.augment = augment
        self.data_dir = PATH_DATASETS
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataclass = CIFAR10
        self.num_class = 10
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.num_samples = num_samples
        
    def train_dataloader(self):
        return data.DataLoader(
            dataset = self.data_train,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
            shuffle = True,
            drop_last = True
        )

    def test_dataloader(self):
        return data.DataLoader(
            dataset = self.data_test,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
            shuffle = False,
            drop_last = False
        )
        
    def prepare_data(self):
        self.dataclass(self.data_dir, download=True)
        

    def transforms(self, val=False):
        if not val:
            tform = transforms.Compose([
                transforms.Resize(self.size),
                transforms.Pad(self.size[0] // 8, padding_mode='reflect'),
                transforms.RandomAffine((-10, 10), (0, 1/8), (1, 1.2)),
                transforms.CenterCrop(self.size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5,)*3, (0.5,)*3)
            ])
        else:
            tform = transforms.Compose([
                transforms.Resize(self.size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,)*3, (0.5,)*3)
            ])
        return tform

    def setup(self, stage=None):
        self.data_train = self.dataclass(
            root = self.data_dir,
            train = True, 
            transform = self.transforms(not self.augment)
        )
        self.data_test = self.dataclass(
            root = self.data_dir,
            train = False, 
            transform = self.transforms(True)
        )
    