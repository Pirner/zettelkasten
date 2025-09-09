from abc import ABC, abstractmethod

import torch


class ClassificationMetric(ABC):

    @abstractmethod
    def calculate(self, ground_truth: torch.Tensor, prediction: torch.Tensor):
        """
        calculate the metric behind
        :param ground_truth: the label of the dataset
        :param prediction: the result of the model
        :return:
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
