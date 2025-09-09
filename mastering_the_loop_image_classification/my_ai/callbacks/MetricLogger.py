import os

import pandas as pd

from config.DTO import TrainingConfig
from my_ai.callback import Callback
from training.trainer import ImageClassificationTrainer


class MetricLogger(Callback):
    def __init__(self, config: TrainingConfig):
        self.config = config

    def on_epoch_end(self, trainer: ImageClassificationTrainer) -> None:
        df = pd.DataFrame(trainer.logs)
        df.to_csv(os.path.join(self.config.experiment_path, 'metrics.csv'))

    def on_init_start(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_init_end(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_train_start(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_train_end(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_epoch_start(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_batch_start(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_batch_end(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_validation_start(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_validation_end(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_exception(self, trainer: ImageClassificationTrainer) -> None:
        pass
