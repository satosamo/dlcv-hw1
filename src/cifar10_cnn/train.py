import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)

        running_loss += loss.item() * x.size(0)
        _, pred = out.max(1)
        correct += pred.eq(y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs, device):
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        # training loss, training accuracy, val. loss, val. accuracy
        tl, ta = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        vl, va = evaluate(model, val_loader, loss_fn, device)

        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)
        hist["train_acc"].append(ta)
        hist["val_acc"].append(va)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"train_acc={ta:.3f} val_acc={va:.3f}")

    return hist
