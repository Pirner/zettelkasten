import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

from config.DTO import TrainingConfig
from models.vit import ViT
from models.custom import CustomModel


class ImageClassificationTrainer:
    model = None
    loss_fn = None
    optimizer = None

    def __init__(self, config: TrainingConfig):
        """
        main class for image classification training.
        :param config: the trainer configuration for the trainer to perform training of an image classification model
        """
        self.config = config
        self.loss_fn = CrossEntropyLoss()

    def build_model(self):
        """
        build the model according to the configuration
        :return:
        """
        if self.config.model_type.lower() == 'vit':
            self.model = ViT(
                image_size=self.config.im_height,
                patch_size=self.config.patch_size,
                num_classes=self.config.n_classes,
                dim=self.config.vit_dim,
                depth=self.config.vit_dim,
                heads=self.config.vit_heads,
                mlp_dim=self.config.vit_mlp_dim,
                dropout=0.1,
                emb_dropout=0.1
            )
            # self.model = ViT(
            # n_layers=self.config.n_layers,
            # chw=(3, self.config.im_height, self.config.im_width),
            # n_patches=self.config.n_patches,
            # d_model=self.config.d_model,
            # n_heads=self.config.n_heads,
            # n_classes=self.config.n_classes,
            # )
        else:
            self.model = CustomModel(config=self.config)
            # raise Exception('unknown model type: ', self.config.model_type)

    def _run_train_epoch(self, train_loader):
        """
        run a training epoch with the my_ai loader
        :param train_loader: for training my_ai
        :return:
        """
        train_losses = []

        for batch in tqdm(train_loader, desc='Running Training epoch'):
            x, y = batch
            x, y = x.to(self.config.device), y.to(self.config.device)
            y_hat = self.model(x)
            loss = self.loss_fn(y_hat, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_losses.append(loss)

        train_losses = torch.tensor(train_losses)
        avg_train_loss = torch.mean(train_losses)
        print(avg_train_loss)

    def train_model(self, train_loader, val_loader):
        """
        train the model according to the configuration with the given my_ai
        :param train_loader: loader for the training my_ai
        :param val_loader: loader for the validation my_ai
        :return:
        """
        self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.model = self.model.to(self.config.device)

        for e in range(self.config.epochs):
            print('[INFO] running epoch {}/{}'.format(e + 1, self.config.epochs))
            self._run_train_epoch(train_loader)

        raise NotImplementedError('foobar')

    def test_model(self, test_loader):
        """
        test the model to achieve a score
        :param test_loader: loader for testing my_ai
        :return:
        """
        raise NotImplementedError('foobar')