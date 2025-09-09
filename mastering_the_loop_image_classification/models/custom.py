from torch import nn
from torchvision import models

from config.DTO import TrainingConfig


class CustomModel(nn.Module):
    def __init__(self, config: TrainingConfig):
        super(CustomModel, self).__init__()
        self.config = config
        model_func = getattr(models, self.config.backbone)
        # self.model_ft = models.resnet18(weights='IMAGENET1K_V1')
        self.model_ft = model_func(weights='IMAGENET1K_V1')
        num_ftrs = self.model_ft.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
        self.model_ft.fc = nn.Linear(num_ftrs, self.config.n_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.model_ft(x)
        x = self.softmax(x)
        return x
