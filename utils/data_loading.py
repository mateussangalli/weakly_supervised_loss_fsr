import os
import numpy as np
import pandas as pd
from skimage.io import imread, imsave


def save_label(label, filename):
    """
    encodes a label image into a label image and saves it
    """
    if len(label.shape) == 3:
        label = np.argmax(label, -1)
    out = np.zeros_like(label)
    out[label == 1] = 0
    out[label == 0] = 127
    out[label == 2] = 255
    out = out.astype(np.uint8)
    imsave(filename, out)


def read_label(filename, one_hot=False):
    """
    reads a greyscale image encoding a label image and decodes it
    """
    label_grey = imread(filename)
    label = np.zeros(label_grey.shape + (3,), dtype=np.float32)
    label[label_grey == 0, 1] = 1.0
    label[label_grey == 127, 0] = 1.0
    label[label_grey == 255, 2] = 1.0
    if one_hot:
        return label
    else:
        return np.argmax(label, -1)


def read_dataset(data_root, subset, image_list=None, min_size=(256, 256)):
    """
    reads the dataset as a list of (image, label) tuples
    """
    data_dir = os.path.join(data_root, subset)
    if image_list is None:
        filenames = os.listdir(os.path.join(data_dir, "images"))
    elif isinstance(image_list, str):
        df = pd.read_csv(image_list)
        df = list(df['name'])
    else:
        filenames = image_list

    dataset = list()
    for fname in filenames:
        image = imread(os.path.join(data_dir, "images", fname))
        if image.shape[0] < min_size[0] or image.shape[1] < min_size[1]:
            continue
        label = read_label(os.path.join(data_dir, "labels", fname))
        dataset.append((image, label))
    return dataset


def read_dataset_pseudo(data_root, subset, labels_dir, image_list=None, min_size=(256, 256)):
    """
    reads the dataset as a list of (image, label) tuples, but the labels come from another folder
    """
    data_dir = os.path.join(data_root, subset)
    if image_list is None:
        filenames = os.listdir(os.path.join(data_dir, "images"))
    elif isinstance(image_list, str):
        df = pd.read_csv(image_list)
        df = list(df['name'])
    else:
        filenames = image_list

    dataset = list()
    for fname in filenames:
        image = imread(os.path.join(data_dir, "images", fname))
        if image.shape[0] < min_size[0] or image.shape[1] < min_size[1]:
            continue
        label = read_label(os.path.join(labels_dir, fname))
        dataset.append((image, label))
    return dataset
