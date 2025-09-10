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

        # 1. Get predicted class indices
        # _, predicted = torch.max(prediction, 1)  # shape: [batch_size]

        # 2. Compare with true labels
        # correct = (predicted == labels).sum().item()

        # # 3. Calculate accuracy
        # accuracy = correct / labels.size(0)
        _, gt_maxed = torch.max(ground_truth, 1)
        accuracy = self.accuracy(prediction, gt_maxed)
        return accuracy
