import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import torch

def get_loaders(batch_size, augment=False, first_1000=False):
    if augment:
        train_t = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
    else:
        train_t = T.ToTensor()

    test_t = T.ToTensor()

    train_set = CIFAR10(root="data", train=True, download=True, transform=train_t)
    test_set = CIFAR10(root="data", train=False, download=True, transform=test_t)

    if first_1000:
        train_set = Subset(train_set, list(range(1000)))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, val_loader
