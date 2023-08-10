import os
import lightning as L
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

def init_dataloader():
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]),
        ]
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]),
        ]
    )   

    train_dataset = CIFAR10(root=PATH_DATASETS, train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root=PATH_DATASETS, train=True, transform=test_transform, download=True)
    L.seed_everything(42)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    L.seed_everything(42)
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])

    test_set = CIFAR10(root=PATH_DATASETS, train=False, transform=test_transform, download=True)

    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)
    
    return train_loader, val_loader, test_loader
