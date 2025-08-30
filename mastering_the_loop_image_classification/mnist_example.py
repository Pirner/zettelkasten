import numpy as np

import pandas as pd
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

from models.vit import ViT

np.random.seed(0)
torch.manual_seed(0)


def main():
    # Loading data
    transform = ToTensor()

    train_set = MNIST(root='./../datasets', train=True, download=True, transform=transform)
    test_set = MNIST(root='./../datasets', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    d_model = 16
    img_size = (1, 28, 28)
    n_heads = 4
    n_layers = 3

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    # model = MyViT((1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
    model = ViT(
        n_layers=n_layers,
        chw=img_size,
        n_patches=7,
        d_model=d_model,
        n_heads=n_heads,
        n_classes=10,
    )
    model = model.to(device)
    N_EPOCHS = 100
    LR = 0.0001

    # Training loop
    performance = []
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in range(N_EPOCHS):
        print('[INFO] running {}|{}'.format(epoch + 1, N_EPOCHS))
        train_loss = 0.0
        correct, total = 0, 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")
        print(f"Train accuracy: {correct / total * 100:.2f}%")
        performance.append({
            'train_loss': train_loss,
            'train_accuracy': correct / total * 100,
        })

    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")
    df = pd.DataFrame(performance)
    df.to_csv('train_logs.csv')


if __name__ == '__main__':
    main()
