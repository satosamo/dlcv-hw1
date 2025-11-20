import torch
from torch import nn

class Cifar10_CNN_Base(nn.Module):
    def __init__(self, activation=nn.ReLU):
        super().__init__()

        act = activation()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            act,
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            act,
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            act,
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            act,
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x