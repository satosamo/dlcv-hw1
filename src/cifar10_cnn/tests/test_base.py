import torch
from cifar10_cnn.models.base import Cifar10_CNN_Base
from cifar10_cnn.train import train_model
from cifar10_cnn.utils.datasets import get_loaders

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, val_loader = get_loaders(batch_size=64, first_1000=True)

model = Cifar10_CNN_Base().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

hist = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=15,
    device=device
)

torch.save(model, "results/baseline.pth")
