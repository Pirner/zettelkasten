import torch
import torchmetrics

from my_ai.metric import ClassificationMetric


class Accuracy(ClassificationMetric):
    def get_name(self) -> str:
        return 'accuracy'

    def __init__(self, n_classes: int, device='cuda'):
        self.n_classes = n_classes
        self.accuracy = torchmetrics.Accuracy(num_classes=n_classes, average='macro', task='multiclass').to(device)

    def calculate(self, ground_truth: torch.Tensor, prediction: torch.Tensor):
        """
        compute the resulting metric
        :param ground_truth: -"-
        :param prediction: -"-
        :return:
        """
        # Calculate accuracy
        # correct = (prediction == ground_truth).sum().item()
        # total = ground_truth.size(0)
        # accuracy = correct / total
        accuracy = self.accuracy(prediction, ground_truth)
        return accuracy
