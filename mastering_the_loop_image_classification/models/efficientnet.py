import torch

from config.DTO import TrainingConfig


class Efficientnet(torch.nn.Module):
    def __init__(self, model, n_classes, dropout=0.2):
        super(Efficientnet, self).__init__()
        self.model = model

        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        self.classifier = torch.nn.Linear(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(self.n_ouputs_last_layer, n_classes),
        )
