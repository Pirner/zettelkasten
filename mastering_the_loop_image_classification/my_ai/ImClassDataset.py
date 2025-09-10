import os
import glob
from typing import List, Dict

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
    classes: Dict[str, int]
    transforms = None

    def __init__(
            self,
            dataset_path: str,
            classes: Dict[str, int],
            transforms,
            config=None,
    ):
        """
        instantiate classification dataset
        :param dataset_path: the path to the dataset location, assumed is a folder with images that contains .jpg files
        and an annotation.csv file
        :param classes: represented through the dataset
        :param transforms: the transformations to apply to the image while loading the my_ai
        :param config: configuration
        """
        self.dataset_path = dataset_path
        self.classes = classes
        self.transforms = transforms
        self.config = config

        self._init_dataset()

    def _init_dataset(self):
        """
        initialize the my_ai points to make the dataset ready
        :return:
        """
        self.data = []
        im_paths = glob.glob(os.path.join(self.dataset_path, '**/*.jpg'), recursive=True)
        for c_name, c_id in self.classes.items():
            c_images = list(filter(lambda x: c_name.lower() in x.lower(), im_paths))
            # c_images = c_images[:500]
            label = np.zeros(len(self.classes), dtype=np.float32)
            label[c_id] = 1.
            for im_path in c_images:
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
