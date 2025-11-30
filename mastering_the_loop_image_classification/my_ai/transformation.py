import albumentations as A
from albumentations.pytorch import ToTensorV2


class DataTransformation:
    @staticmethod
    def get_train_transforms(im_h: int, im_w: int, use_aug=False):
        """
        get training my_ai transforms
        :param im_h: image height
        :param im_w: image width
        :param use_aug: whether to use augmentions or not
        :return:
        """

        if use_aug:
            t_train = A.Compose(
                [
                    # A.SmallestMaxSize(max_size=160),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.2),
                    A.RandomCrop(height=im_h * 2, width=im_w * 2, p=0.2),
                    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),
                    A.RandomBrightnessContrast(p=0.2),
                    A.HorizontalFlip(p=0.2),
                    A.Affine(p=0.2, scale=0.8, shear=5, translate_percent=0.1, rotate=20),
                    A.Blur(blur_limit=3, p=0.1),
                    A.OpticalDistortion(p=0.1),
                    A.GridDistortion(p=0.1),
                    A.HueSaturationValue(p=0.1),
                    A.Resize(height=im_h, width=im_w, p=1),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            t_train = A.Compose(
                [
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
