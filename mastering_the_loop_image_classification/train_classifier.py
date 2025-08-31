from pprint import pprint

import yaml

from config.DTO import TrainingConfig


def main():
    config_path = 'train_configs/implementing.yaml'
    with open(config_path) as stream:
        try:
            y_config = yaml.safe_load(stream)
            config = TrainingConfig(**y_config)
        except yaml.YAMLError as exc:
            print(exc)

    pprint('[INFO] training model with: {}'.format(config))
    # all training job parameters
    # d_model = 16
    # img_size = (1, 28, 28)
    # n_heads = 4
    # n_layers = 3
    # n_classes = 10
    # n_patches = 7
    # N_EPOCHS = 100
    # LR = 0.0001
    # batch_size = 32


if __name__ == '__main__':
    main()
