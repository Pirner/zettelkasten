import cv2


class VisionIO:
    @staticmethod
    def read_im(im_path: str):
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im
