from config.DTO import TrainingConfig
from models.vit import ViT


class ImageClassificationTrainer:
    model = None

    def __init__(self, config: TrainingConfig):
        """
        main class for image classification training.
        :param config: the trainer configuration for the trainer to perform training of an image classification model
        """
        self.config = config

    def build_model(self):
        """
        build the model according to the configuration
        :return:
        """
        if self.config.model_type.lower() == 'vit':
            self.model = ViT(
                n_layers=self.config.n_layers,
                chw=(3, self.config.im_height, self.config.im_width),
                n_patches=self.config.n_patches,
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                n_classes=self.config.n_classes,
            )
        else:
            raise Exception('unknown model type: ', self.config.model_type)

    def train_model(self, train_loader, val_loader):
        """
        train the model according to the configuration with the given data
        :param train_loader: loader for the training data
        :param val_loader: loader for the validation data
        :return:
        """
        raise NotImplementedError('foobar')

    def test_model(self, test_loader):
        """
        test the model to achieve a score
        :param test_loader: loader for testing data
        :return:
        """
        raise NotImplementedError('foobar')