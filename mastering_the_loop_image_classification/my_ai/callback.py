from abc import ABC, abstractmethod
from typing import Any

from training.trainer import ImageClassificationTrainer


class Callback(ABC):
    """Abstract base class for training callbacks, similar to PyTorch Lightning."""

    @abstractmethod
    def on_init_start(self, trainer: ImageClassificationTrainer) -> None:
        """Called at the beginning of trainer initialization."""
        pass

    @abstractmethod
    def on_init_end(self, trainer: ImageClassificationTrainer) -> None:
        """Called at the end of trainer initialization."""
        pass

    @abstractmethod
    def on_train_start(self, trainer: ImageClassificationTrainer) -> None:
        """Called before training begins."""
        pass

    @abstractmethod
    def on_train_end(self, trainer: ImageClassificationTrainer) -> None:
        """Called after training ends."""
        pass

    @abstractmethod
    def on_epoch_start(self, trainer: ImageClassificationTrainer) -> None:
        """Called at the beginning of each epoch."""
        pass

    @abstractmethod
    def on_epoch_end(self, trainer: ImageClassificationTrainer) -> None:
        """Called at the end of each epoch."""
        pass

    @abstractmethod
    def on_batch_start(self, trainer: ImageClassificationTrainer) -> None:
        """Called at the beginning of each batch."""
        pass

    @abstractmethod
    def on_batch_end(self, trainer: ImageClassificationTrainer) -> None:
        """Called at the end of each batch."""
        pass

    @abstractmethod
    def on_validation_start(self, trainer: ImageClassificationTrainer) -> None:
        """Called before validation starts."""
        pass

    @abstractmethod
    def on_validation_end(self, trainer: ImageClassificationTrainer) -> None:
        """Called after validation ends."""
        pass

    @abstractmethod
    def on_exception(self, trainer: ImageClassificationTrainer) -> None:
        """Called if an exception is raised during training."""
        pass
