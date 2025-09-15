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
    logs = None
    callbacks = None
    metrics = None
    epoch: int

    def __init__(self, config: TrainingConfig):
        """
        main class for image classification training.
        :param config: the trainer configuration for the trainer to perform training of an image classification model
        """
        self.config = config
        self.loss_fn = CrossEntropyLoss()
        self.reset_trainer()

    def reset_trainer(self):
        self.logs = {
            'train_loss': [],
            'val_loss': [],
        }
        self.epoch = 0

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
        running_metrics = {}
        for m in self.metrics:
            running_metrics['train_{}'.format(m.get_name())] = []

        for batch in tqdm(train_loader, desc='Running Training epoch'):
            x, y = batch
            x, y = x.to(self.config.device), y.to(self.config.device)
            y_hat = self.model(x)
            loss = self.loss_fn(y_hat, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_losses.append(loss)

            for m in self.metrics:
                m_res = m.calculate(y, y_hat)
                running_metrics['train_{}'.format(m.get_name())].append(m_res)

        train_losses = torch.tensor(train_losses)
        avg_train_loss = torch.mean(train_losses)
        for key, value in running_metrics.items():
            avg_val = torch.mean(torch.tensor(value))
            self.logs[key].append(avg_val)
        self.logs['train_loss'].append(avg_train_loss)

    def _run_val_epoch(self, val_loader):
        """
        run an epoch with the validation data loader
        :param val_loader: data loader for validation data
        :return:
        """
        val_losses = []
        running_metrics = {}
        for m in self.metrics:
            running_metrics['val_{}'.format(m.get_name())] = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Running Validation Epoch'):
                x, y = batch
                x, y = x.to(self.config.device), y.to(self.config.device)
                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)
                val_losses.append(loss)

                for m in self.metrics:
                    m_res = m.calculate(y, y_hat)
                    running_metrics['val_{}'.format(m.get_name())].append(m_res)

        val_losses = torch.tensor(val_losses)
        avg_val_loss = torch.mean(val_losses)
        for key, value in running_metrics.items():
            avg_val = torch.mean(torch.tensor(value))
            self.logs[key].append(avg_val)
        self.logs['val_loss'].append(avg_val_loss)

        for cb in self.callbacks:
            cb.on_validation_end(self)

    def train_model(self, train_loader, val_loader):
        """
        train the model according to the configuration with the given my_ai
        :param train_loader: loader for the training my_ai
        :param val_loader: loader for the validation my_ai
        :return:
        """
        for cb in self.callbacks:
            cb.on_train_start(self)

        for m in self.metrics:
            self.logs['train_{}'.format(m.get_name())] = []
            self.logs['val_{}'.format(m.get_name())] = []

        self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.model = self.model.to(self.config.device)

        for e in range(self.config.epochs):
            print('\n[INFO] running epoch {}/{}'.format(e + 1, self.config.epochs))
            self._run_train_epoch(train_loader)
            self._run_val_epoch(val_loader)

            self.epoch = e
            for cb in self.callbacks:
                cb.on_epoch_end(self)

    def test_model(self, test_loader):
        """
        test the model to achieve a score
        :param test_loader: loader for testing my_ai
        :return:
        """
        test_losses= []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Running Validation Epoch'):
                x, y = batch
                x, y = x.to(self.config.device), y.to(self.config.device)
                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)
                test_losses.append(loss)

        test_losses = torch.tensor(test_losses)
        avg_test_loss = torch.mean(test_losses)
        # self.logs['val_loss'].append(avg_test_loss)
        print('[INFO] test_loss: {}'.format(avg_test_loss))
