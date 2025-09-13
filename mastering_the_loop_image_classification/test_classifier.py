import argparse
import json
import os

from config.DTO import TrainingConfig
from training.trainer import ImageClassificationTrainer


def create_args():
    parser = argparse.ArgumentParser(description='Model Testing Args parser.')
    parser.add_argument(
        '--run_path',
        type=str,
        help='path to the source directory of the model experiment run'
    )

    parser.add_argument(
        '--dataset_path',
        type=str,
        help='path to the testing dataset'
    )

    args = parser.parse_args()
    return args


def main():
    args = create_args()

    with open(os.path.join(args.run_path, 'config.json')) as f:
        d = json.load(f)
        config = TrainingConfig(**d)
    trainer = ImageClassificationTrainer(config=config)
    trainer.build_model()

    test_dataset = ClassificationDataset(
        dataset_path=config.test_dataset_path,
        classes=config.classes,
        transforms=DataTransformation.get_val_transforms(im_h=config.im_height, im_w=config.im_width),
        config=config,
    )
    print('[INFO] finished testing model')


if __name__ == '__main__':
    main()
