from torch import nn
from torchvision import models

from config.DTO import TrainingConfig


class CustomModel(nn.Module):
    def __init__(self, config: TrainingConfig):
        super(CustomModel, self).__init__()
        self.config = config
        model_func = getattr(models, self.config.backbone)
        self.model_ft = model_func(weights='IMAGENET1K_V1')
        if 'convnext' in self.config.backbone.lower():
            self.model_ft.classifier[2] = nn.Linear(self.model_ft.classifier[2].in_features, len(self.config.classes))
        elif 'vit' in self.config.backbone.lower():
            self.model_ft.heads.head = nn.Linear(self.model_ft.heads.head.in_features, len(self.config.classes))
        else:
            num_ftrs = self.model_ft.fc.in_features
            self.model_ft.fc = nn.Linear(num_ftrs, len(self.config.classes))
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.model_ft(x)
        # x = self.softmax(x)
        return x
