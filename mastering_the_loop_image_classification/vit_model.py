import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import Adam
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
import numpy as np


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, img_size, patch_size, n_channels):
        super().__init__()

        self.d_model = d_model  # Dimensionality of Model
        self.img_size = img_size  # Image Size
        self.patch_size = patch_size  # Patch Size
        self.n_channels = n_channels  # Number of Channels

        self.linear_project = nn.Conv2d(self.n_channels, self.d_model, kernel_size=self.patch_size,
                                        stride=self.patch_size)

    # B: Batch Size
    # C: Image Channels
    # H: Image Height
    # W: Image Width
    # P_col: Patch Column
    # P_row: Patch Row
    def forward(self, x):
        x = self.linear_project(x)  # (B, C, H, W) -> (B, d_model, P_col, P_row)

        x = x.flatten(2)  # (B, d_model, P_col, P_row) -> (B, d_model, P)

        x = x.transpose(1, 2)  # (B, d_model, P) -> (B, P, d_model)

        return x


class ViT(nn.Module):
    def __init__(self, chw=(1, 28, 28), n_patches=7):
        super(ViT, self).__init__()
        # self.patch_embedding = PatchEmbedding(self.d_model, self.img_size, self.patch_size, self.n_channels)
        # Attributes
        self.chw = chw  # (C, H, W)
        self.n_patches = n_patches

        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"

    def patchify(self, images, n_patches):
        n, c, h, w = images.shape

        assert h == w, "Patchify method is implemented for square images only"

        patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
        patch_size = h // n_patches

        for idx, image in enumerate(images):
            for i in range(n_patches):
                for j in range(n_patches):
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    patches[idx, i * n_patches + j] = patch.flatten()
        return patches

    def forward(self, images):
        patches = self.patchify(images, self.n_patches)
        exit(0)
