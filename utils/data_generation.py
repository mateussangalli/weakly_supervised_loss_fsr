
import keras.backend as K
import numpy as np

from .crop_selection import select_crop, select_crop_full


def generator_from_list(list_im, list_gt, list_weight=None):
    """
    generator from a list of images and a list of their respective labels
    Args:
        list_im: list of input images (rank 3 arrays)
        list_gt: list of input labels (rank 3 arrays H x W x 1)

    Yields:
        x: cropped array
        y: cropped label
    """
    if list_weight is not None:
        for im, gt, w in zip(list_im, list_gt, list_weight):
            im = im.astype(np.float32) / 255.
            yield im, gt, w
    else:
        for im, gt in zip(list_im, list_gt):
            im = im.astype(np.float32) / 255.
            yield im, gt


def generator_train_crops_full(
        list_im,
        list_gt,
        crop_size=256,
        crops_per_image=64,
        image_data_format=K.image_data_format(),
        max_scale=1.,
        aug=False,
        list_weight=None):
    """
    generator object from a list of images and a list of their respective labels
    if aug = True performs scale augmentation and returns crops with varying sizes

    Args:
        listIm: list of input images (rank 3 arrays)
        listGT: list of input labels (rank 3 arrays H x W x 1)
        crop_size: size of the cropped images
        crops_per_image: number of crops per image / label pair
        image_data_format: 'channels_first' or 'channels_last'
        max_scale: maximal scale for cropped regtion scaling
        aug: boolean specifying whether to use augmentation or not when selecting crop

    Yields:
        x: cropped array
        y: cropped label
    """
    if list_weight is None:
        for im, gt in zip(list_im, list_gt):
            im = im.astype(np.float32) / 255.
            for i in range(crops_per_image):
                x, y = select_crop_full(
                    im, gt,
                    dict(crop_size=crop_size,
                         max_scale=max_scale),
                    aug=aug)
                if image_data_format == 'channels_first':
                    x = np.transpose(x, (2, 0, 1))
                yield x, y
    else:
        for im, gt, w in zip(list_im, list_gt, list_weight):
            im = im.astype(np.float32) / 255.
            for i in range(crops_per_image):
                x, y = select_crop_full(
                    im, gt,
                    dict(crop_size=crop_size,
                         max_scale=max_scale),
                    aug=aug)
                if image_data_format == 'channels_first':
                    x = np.transpose(x, (2, 0, 1))
                weight = w * \
                    np.ones([crop_size, crop_size, 1], dtype=np.float32)
                yield x, y, weight


def generator_train_crops(
        list_im,
        list_gt,
        crop_size=256,
        crops_per_image=64,
        image_data_format=K.image_data_format(),
        max_scale=1.,
        aug=False,
        list_weight=None):
    """
    generator object from a list of images and a list of their respective labels
    if aug = True performs scale augmentation and returns crops with varying sizes

    Args:
        listIm: list of input images (rank 3 arrays)
        listGT: list of input labels (rank 3 arrays H x W x 1)
        crop_size: size of the cropped images
        crops_per_image: number of crops per image / label pair
        image_data_format: 'channels_first' or 'channels_last'
        max_scale: maximal scale for cropped regtion scaling
        aug: boolean specifying whether to use augmentation or not when selecting crop

    Yields:
        x: cropped array
        y: cropped label
    """
    if list_weight is None:
        for im, gt in zip(list_im, list_gt):
            im = im.astype(np.float32) / 255.
            for i in range(crops_per_image):
                x, y = select_crop(im, gt,
                                   dict(crop_size=crop_size,
                                        max_scale=max_scale),
                                   aug=aug)
                if image_data_format == 'channels_first':
                    x = np.transpose(x, (2, 0, 1))
                yield x, y
    else:
        for im, gt, w in zip(list_im, list_gt, list_weight):
            im = im.astype(np.float32) / 255.
            for i in range(crops_per_image):
                x, y = select_crop(im, gt,
                                   dict(crop_size=crop_size,
                                        max_scale=max_scale),
                                   aug=aug)
                if image_data_format == 'channels_first':
                    x = np.transpose(x, (2, 0, 1))
                weight = w * \
                    np.ones([crop_size, crop_size, 1], dtype=np.float32)
                yield x, y, weight


def generator_val_crops(
        listIm,
        listGT,
        crop_size=256,
        crops_per_image=64,
        image_data_format=K.image_data_format()):
    """
    generator object from a list of images and a list of their respective labels

    Args:
        listIm: list of input images (rank 3 arrays)
        listGT: list of input labels (rank 3 arrays H x W x 1)
        crop_size: size of the cropped images
        crops_per_image: number of crops per image / label pair
        image_data_format: 'channels_first' or 'channels_last'

    Yields:
        x: cropped array
        y: cropped label
    """
    for im, gt in zip(listIm, listGT):
        im = im.astype(np.float32) / 255.
        for i in range(crops_per_image):
            x, y = select_crop(im, gt,
                               dict(crop_size=crop_size))
            if image_data_format == 'channels_first':
                x = np.transpose(x, (2, 0, 1))
                y = np.transpose(y, (2, 0, 1))
            yield x, y
