import os
from typing import List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from my_ai.DTO import DataPoint


class ClassificationDataset(Dataset):
    """
    Classification dataset implementation to train, validate and test deep learning models.
    This dataset is responsible for loading the dataset that we store in a proper format.
    """
    df: pd.DataFrame
    data: List[DataPoint]
    dataset_path: str
    n_classes: int
    transforms = None

    def __init__(
            self,
            dataset_path: str,
            n_classes: int,
            transforms,
            config=None,
    ):
        """
        instantiate classification dataset
        :param dataset_path: the path to the dataset location, assumed is a folder with images that contains .jpg files
        and an annotation.csv file
        :param n_classes: number of classes represented through the dataset
        :param transforms: the transformations to apply to the image while loading the my_ai
        :param config: configuration
        """
        self.dataset_path = dataset_path
        self.n_classes = n_classes
        self.transforms = transforms
        self.config = config

        self._init_dataset()

    def _init_dataset(self):
        """
        initialize the my_ai points to make the dataset ready
        :return:
        """
        self.df = pd.read_csv(os.path.join(self.dataset_path, 'annotations.csv'))
        self.data = []
        for index, row in self.df.iterrows():
            label_id = row['Retinopathy grade']
            label = np.zeros(self.n_classes, dtype=np.float32)
            label[label_id] = 1.
            im_path = os.path.join(self.dataset_path, 'images', row['Image name'])
            dp = DataPoint(
                im_path=im_path,
                label=label,
            )
            if self.config is not None:
                dp.im_height = self.config.im_height
                dp.im_width = self.config.im_width
            self.data.append(dp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data_point = self.data[item]
        im_data = data_point.get_im_data()
        label = data_point.label
        if self.transforms is None:
            return im_data, label
        else:
            im_transformed = self.transforms(image=im_data)['image']
            return im_transformed, label
