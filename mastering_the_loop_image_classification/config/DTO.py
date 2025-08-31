from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # model specific
    model_type: str
    d_model: int
    im_height: int
    im_width: int
    n_heads: int
    n_layers: int
    n_classes: int
    n_patches: int

    # training specific
    epochs: int
    learning_rate: float
    batch_size: int
