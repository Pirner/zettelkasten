import albumentations as A
from albumentations.pytorch import ToTensorV2


class DataTransformation:
    @staticmethod
    def get_train_transforms(im_h: int, im_w: int):
        """
        get training my_ai transforms
        :param im_h: image height
        :param im_w: image width
        :return:
        """
        t_train = A.Compose(
            [
                # A.SmallestMaxSize(max_size=160),
                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                # A.RandomCrop(height=128, width=128),
                # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                # A.RandomBrightnessContrast(p=0.5),
                A.Resize(height=im_h, width=im_w, p=1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        return t_train

    @staticmethod
    def get_val_transforms(im_h: int, im_w: int):
        """
        get validation my_ai transforms
        :param im_h: image height
        :param im_w: image width
        :return:
        """
        t_val = A.Compose(
            [
                A.Resize(height=im_h, width=im_w, p=1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        return t_val
