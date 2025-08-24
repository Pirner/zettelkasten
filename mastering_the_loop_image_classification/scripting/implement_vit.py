import torch

from vit_model import ViT


if __name__ == '__main__':
    # Current model
    model = ViT(chw=(1, 28, 28), n_patches=7, d_model=16)

    x = torch.randn(7, 1, 28, 28) # Dummy images
    print(model(x).shape) # torch.Size([7, 49, 16])
