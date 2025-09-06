from dataclasses import dataclass
from typing import Optional

import numpy as np

from vision.io import VisionIO


@dataclass
class DataPoint:
    label: np.array
    im_data: Optional[np.array] = None
    im_path: Optional[str] = None

    def get_im_data(self):
        if self.im_data is None and self.im_path is None:
            raise Exception('image my_ai point has no image my_ai available')
        if self.im_data is not None:
            return self.im_data
        else:
            self.im_data = VisionIO.read_im(im_path=self.im_path)
            return self.im_data
