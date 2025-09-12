from torch.utils.tensorboard import SummaryWriter

from my_ai.callback import Callback
from training.trainer import ImageClassificationTrainer


class TensorboardLogger(Callback):
    experiment_path: str
    writer = None

    def __init__(self, experiment_path: str):
        """
        initialize the tensorboard logger
        :param experiment_path: where to log the tensorboard information
        """
        self.experiment_path = experiment_path
        self.writer = SummaryWriter(log_dir=self.experiment_path)

    def on_train_start(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_init_start(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_init_end(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_train_end(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_epoch_start(self, trainer: ImageClassificationTrainer) -> None:
        pass

    def on_epoch_end(self, trainer: ImageClassificationTrainer) -> None:
        """
        on epoch end write all the logs into the tensorboard entry with the writer
        :param trainer: image classification trainer which runs the job
        :return:
        """
        for key, item in trainer.logs.items():
            self.writer.add_scalar(key, item[-1], trainer.epoch)
        self.writer.flush()

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