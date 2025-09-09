from dataclasses import asdict
import os
import json

import torch

from config.DTO import TrainingConfig
from my_ai.callback import Callback
from training.trainer import ImageClassificationTrainer


class ModelCheckpointer(Callback):
    def __init__(self, config: TrainingConfig):
        """
        basic callback for models to checkpoint
        :param config: configuration for the training run
        """
        self.config = config

    def on_train_start(self, trainer: ImageClassificationTrainer) -> None:
        with open(os.path.join(self.config.experiment_path, 'config.json'), "w") as file:
            json.dump(asdict(self.config), file, indent=4)

    def on_init_start(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_init_end(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_train_end(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_epoch_start(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_epoch_end(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_batch_start(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_batch_end(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_validation_start(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_validation_end(self, trainer: ImageClassificationTrainer) -> None:
        """
        run this method at the end of validation
        :param trainer: the trainer which runs everything
        :return:
        """
        torch.save(trainer.model.state_dict(), os.path.join(self.config.experiment_path, "model_weights.pth"))

    def on_exception(self, trainer: ImageClassificationTrainer) -> None:
        pass