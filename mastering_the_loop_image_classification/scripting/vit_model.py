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

        self.linear_project = nn.Conv2d(
            self.n_channels,
            self.d_model,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) # Classification Token

        # Creating positional encoding
        pe = torch.zeros(max_seq_length, d_model)

        for pos in range(max_seq_length):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
                else:
                    pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Expand to have class token for every image in batch
        tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)

        # Adding class tokens to the beginning of each embedding
        x = torch.cat((tokens_batch,x), dim=1)

        # Add positional encoding to embeddings
        x = x + self.pe

        return x


class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.head_size = head_size

        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)

    def forward(self, x):
        # Obtaining Queries, Keys, and Values
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Dot Product of Queries and Keys
        attention = Q @ K.transpose(-2,-1)

        # Scaling
        attention = attention / (self.head_size ** 0.5)

        attention = torch.softmax(attention, dim=-1)

        attention = attention @ V

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.head_size = d_model // n_heads

        self.W_o = nn.Linear(d_model, d_model)

        self.heads = nn.ModuleList([AttentionHead(d_model, self.head_size) for _ in range(n_heads)])

    def forward(self, x):
        # Combine attention heads
        out = torch.cat([head(x) for head in self.heads], dim=-1)

        out = self.W_o(out)

        return out


class TransformerEncoder(nn.Module):
  def __init__(self, d_model, n_heads, r_mlp=4):
    super().__init__()
    self.d_model = d_model
    self.n_heads = n_heads

    # Sub-Layer 1 Normalization
    self.ln1 = nn.LayerNorm(d_model)

    # Multi-Head Attention
    self.mha = MultiHeadAttention(d_model, n_heads)

    # Sub-Layer 2 Normalization
    self.ln2 = nn.LayerNorm(d_model)

    # Multilayer Perception
    self.mlp = nn.Sequential(
        nn.Linear(d_model, d_model*r_mlp),
        nn.GELU(),
        nn.Linear(d_model*r_mlp, d_model)
    )

  def forward(self, x):
    # Residual Connection After Sub-Layer 1
    out = x + self.mha(self.ln1(x))

    # Residual Connection After Sub-Layer 2
    out = out + self.mlp(self.ln2(out))

    return out


class ViT(nn.Module):
    def __init__(
            self,
            n_layers,
            n_classes,
            chw=(1, 28, 28),
            n_patches=7,
            d_model=9,
            n_heads=3
    ):
        super(ViT, self).__init__()

        # Attributes
        self.chw = chw  # (C, H, W)
        self.n_patches = n_patches
        self.n_channels, self.h, self.w = chw
        self.im_size = (self.h, self.w)
        self.n_heads = n_heads
        self.n_classes = n_classes

        self.patch_size = self.h // n_patches
        self.d_model = d_model

        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_embedding = PatchEmbedding(self.d_model, self.im_size, self.patch_size, self.n_channels)

        # classification token -> v_class, learnable parameter
        self.class_token = nn.Parameter(torch.rand(1, self.d_model))

        # positional embedding
        # 3) Positional embedding
        # self.pos_embed = nn.Parameter(torch.tensor(self.get_positional_embeddings(self.n_patches ** 2 + 1)))
        # self.pos_embed.requires_grad = False
        self.max_seq_length = self.n_patches ** 2 + 1
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_seq_length)
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoder(self.d_model, self.n_heads) for _ in range(n_layers)])

        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.n_classes),
            nn.Softmax(dim=-1)
        )

    def get_positional_embeddings(self, sequence_length):
        result = torch.ones(sequence_length, self.d_model)
        for i in range(sequence_length):
            for j in range(self.d_model):
                result[i][j] = np.sin(i / (10000 ** (j / self.d_model))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / self.d_model)))
        return result

    def create_patches(self, images):
        n, c, h, w = images.shape

        assert h == w, "Patchify method is implemented for square images only"

        patches = torch.zeros(n, self.n_patches ** 2, h * w * c // self.n_patches ** 2)
        patch_size = h // self.n_patches

        for idx, image in enumerate(images):
            for i in range(self.n_patches):
                for j in range(self.n_patches):
                    patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                    patches[idx, i * self.n_patches + j] = patch.flatten()
        return patches

    def forward(self, images):
        # patches = self.create_patches(images)
        patches = self.patch_embedding(images)  # use this for a more optimized version to create patches.
        x = self.positional_encoding(patches)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])
        return x
