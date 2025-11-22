import os
from pprint import pprint

from torch.utils.data import DataLoader
import yaml

from config.DTO import TrainingConfig
from my_ai.callbacks.ModelCheckpointer import ModelCheckpointer
from my_ai.callbacks.MetricLogger import MetricLogger
from my_ai.callbacks.TensorboardLogger import TensorboardLogger
from my_ai.ImClassDataset import ClassificationDataset
from my_ai.metrics.Accuracy import Accuracy
from my_ai.transformation import DataTransformation
from training.trainer import ImageClassificationTrainer


def main():
    config_path = 'train_configs/base_routine.yaml'
    config_path = 'train_configs/implementing.yaml'
    with open(config_path) as stream:
        try:
            y_config = yaml.safe_load(stream)
            config = TrainingConfig(**y_config)
        except yaml.YAMLError as exc:
            print(exc)

    n = len(os.listdir(config.experiment_path))
    config.experiment_path = os.path.join(config.experiment_path, '{:03d}'.format(n))
    os.makedirs(config.experiment_path)
    pprint('[INFO] training model with: {}'.format(config))
    # construct my_ai loaders
    train_dataset = ClassificationDataset(
        dataset_path=config.train_dataset_path,
        classes=config.classes,
        transforms=DataTransformation.get_train_transforms(im_h=config.im_height, im_w=config.im_width),
        config=config,
    )

    val_dataset = ClassificationDataset(
        dataset_path=config.val_dataset_path,
        classes=config.classes,
        transforms=DataTransformation.get_val_transforms(im_h=config.im_height, im_w=config.im_width),
        config=config,
    )
    test_dataset = ClassificationDataset(
        dataset_path=config.test_dataset_path,
        classes=config.classes,
        transforms=DataTransformation.get_val_transforms(im_h=config.im_height, im_w=config.im_width),
        config=config,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    trainer = ImageClassificationTrainer(config=config)
    trainer.callbacks = [
        ModelCheckpointer(config=config),
        MetricLogger(config=config),
        TensorboardLogger(experiment_path=config.experiment_path),
    ]
    trainer.metrics = [
        Accuracy(n_classes=len(config.classes)),
    ]
    trainer.build_model()
    trainer.train_model(train_loader=train_dataloader, val_loader=val_dataloader)
    trainer.test_model(test_loader=test_dataloader)


if __name__ == '__main__':
    main()
