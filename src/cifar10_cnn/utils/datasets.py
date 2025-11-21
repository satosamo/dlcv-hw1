import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np


CIFAR10_STATS = ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) # https://github.com/kuangliu/pytorch-cifar/issues/19


# randomly masks out a square patch of the image
class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return img * mask


def get_loaders(batch_size, augment_mode=False, first=None):
    
    normalize = T.Normalize(*CIFAR10_STATS)
    base_transform = [T.ToTensor(), normalize]
    if augment_mode == "basic":
        train_t = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
    elif augment_mode == "advanced":
        train_t = T.Compose([
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            normalize
        ])
    elif augment_mode == "best":
        train_t = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            normalize, 
            Cutout(n_holes=1, length=16)
        ])
    else: # augment_mode=none
        train_t = T.Compose(base_transform)

    test_t = T.Compose([T.ToTensor(), normalize])

    train_set = CIFAR10(root="data", train=True, download=True, transform=train_t)
    test_set = CIFAR10(root="data", train=False, download=True, transform=test_t)

    if first is not None:
        train_set = Subset(train_set, list(range(first)))


    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        test_set,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader
