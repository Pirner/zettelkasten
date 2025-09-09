from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    # model specific
    model_type: str
    im_height: int
    im_width: int
    patch_size: int
    n_classes: int
    vit_dim: int
    vit_depth: int
    vit_heads: int
    vit_mlp_dim: int

    # training specific
    epochs: int
    learning_rate: float
    batch_size: int
    device: str

    # dataset and logging
    train_dataset_path: str
    val_dataset_path: str
    test_dataset_path: str
    experiment_path: str

    # optional values
    backbone: Optional[str] = None
