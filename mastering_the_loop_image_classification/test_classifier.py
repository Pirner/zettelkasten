import argparse
import json
import os

import numpy as np
import onnxruntime as ort
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

from config.DTO import TrainingConfig
from my_ai.ImClassDataset import ClassificationDataset


def create_args():
    parser = argparse.ArgumentParser(description='Model Testing Args parser.')
    parser.add_argument(
        '--run_path',
        type=str,
        help='path to the source directory of the model experiment run'
    )

    parser.add_argument(
        '--dataset_path',
        type=str,
        help='path to the testing dataset'
    )

    args = parser.parse_args()
    return args


def normalize_numpy(img):
    """
    Normalize a NumPy image array using ImageNet mean and std.

    Args:
        img (np.ndarray): Image array of shape (H, W, 3) with values in [0, 1].

    Returns:
        np.ndarray: Normalized image.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img / 255.
    img = ((img - mean) / std)
    img = np.transpose(img, (2, 0, 1))
    return img.astype(np.float32)


# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference


def main():
    args = create_args()

    with open(os.path.join(args.run_path, 'config.json')) as f:
        d = json.load(f)
        config = TrainingConfig(**d)
    print('[INFO] load model')
    session = ort.InferenceSession(os.path.join(args.run_path, "model.onnx"))
    input_name = session.get_inputs()[0].name

    # load data from disk
    test_dataset = ClassificationDataset(
        dataset_path=args.dataset_path,
        classes=config.classes,
        transforms=None,
        config=config,
    )

    gt_labels = []
    predictions = []
    n_correct = 0
    n_total = len(test_dataset)

    for tdp in tqdm(test_dataset, total=len(test_dataset)):
        x, y = tdp
        x = normalize_numpy(x)
        x = np.expand_dims(x, axis=0)
        y_hat = session.run(None, {input_name: x})[0][0]
        y_hat = softmax(y_hat)

        pred_class = np.argmax(y_hat)
        gt_l = np.argmax(y)

        gt_labels.append(gt_l)
        predictions.append(pred_class)

    n_total = len(test_dataset)
    n_correct = np.sum(np.array(gt_labels) == np.array(predictions))
    accuracy = n_correct / n_total
    print('[INFO] scored {0} accuracy'.format(accuracy * 100))

    n_classes = len(config.classes)
    cm = confusion_matrix(gt_labels, predictions, labels=range(n_classes))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=list(config.classes.keys()))
    disp.plot()
    plt.show()
    print('[INFO] finished testing model')


if __name__ == '__main__':
    main()
