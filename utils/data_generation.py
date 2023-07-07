import numpy as np
import tensorflow as tf

from .crop_selection import select_crop
from .data_augmentation import (RandomRotation, ColorJitter, ColorTransfer,
                                random_horizontal_flip, resize_inputs)


def crop_generator(
    data, crop_size, crops_per_image, scale_range, yield_label=True, normalize=True
):
    """
    Takes a list of (image, label) pairs and yields crops of the images.
    Parameters:
        data (List[Tuple[np.ndarray, np.ndarray]]): a list of (image, label)
            pairs, where image is a float32 or integer array of rank 3 and
            label is an integer array of rank 2.
        crop_size (int or Tuple[int, int]): height and width of the cropped
            region.
            if integer, cropped region is a square.
        crops_per_image (int): how many crops per image.
        scale_range (Tuple[float, float]): minimum and maximum value for res-
            scaling cropped region.
        yield_label (bool): where to yield only the image or the label too.
        normalize (bool): whether to normalize the image from [0..255] valued
            to [0., 1.] valued.
    Yields:
        im_crop (np.ndarray): cropped image
        gt_crop (np.ndarray): optional. Cropped label.
    """
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    for image, label in data:
        if normalize:
            image = image.astype(np.float32) / 255.0
        for i in range(crops_per_image):
            s = np.random.uniform(
                low=np.log(scale_range[0]), high=np.log(scale_range[1])
            )
            s = np.exp(s)
            new_size = (int(crop_size[0] * s), int(crop_size[1] * s))
            im_crop, gt_crop = select_crop(image, label, new_size)
            if yield_label:
                yield im_crop, gt_crop
            else:
                yield im_crop


def get_tf_val_dataset(data):
    def gen_val():
        for image, label in data:
            image = image.astype(np.float32) / 255.0
            yield image, label

    ds_val = tf.data.Dataset.from_generator(
        gen_val, output_types=(tf.float32, tf.int32)
    )
    ds_val = ds_val.map(
        lambda im, gt: (im, tf.one_hot(gt, 3)), num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_val = ds_val.batch(1)
    ds_val = ds_val.repeat()
    return ds_val


def get_tf_train_dataset(data, params):
    if params["min_scale"] < 0.0:
        scale_range = (1.0 / params["max_scale"], params["max_scale"])
    else:
        scale_range = (params["min_scale"], params["max_scale"])

    def gen_train():
        return crop_generator(
            data, params["crop_size"], params["crops_per_image"], scale_range
        )

    samples_per_epoch = len(data) * params["crops_per_image"]

    ds_train = tf.data.Dataset.from_generator(
        gen_train,
        output_types=(tf.float32, tf.int32),
        output_shapes=((None, None, 3), (None, None)),
    )
    ds_train = ds_train.shuffle(samples_per_epoch)
    ds_train = ds_train.map(
        lambda im, gt: resize_inputs(im, gt, params["crop_size"]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds_train = ds_train.map(
        RandomRotation(params["rotation_angle"]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if "hue_jitter" in params and "sat_jitter" in params and "val_jitter" in params:
        color_jitter = ColorJitter(
            params["hue_jitter"], params["sat_jitter"], params["val_jitter"])
        ds_train = ds_train.map(
            lambda x, y: (color_jitter(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds_train = ds_train.map(
        lambda im, gt: (im, tf.one_hot(gt, 3)), num_parallel_calls=tf.data.AUTOTUNE
    )
    ds_train = ds_train.map(
        random_horizontal_flip,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if "noise_value" in params:
        noise_shape = (params["crop_size"], params["crop_size"], 3)
        ds_train = ds_train.map(
            lambda x, y: (x + tf.random.normal(noise_shape, 0., params["noise_value"]), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    if "color_transfer_probability" in params:
        color_transfer = ColorTransfer(
            params["color_transfer_means"],
            params["color_transfer_probability"])
        ds_train = ds_train.map(
            lambda x, y: (color_transfer(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    ds_train = ds_train.batch(params["batch_size"])

    return ds_train
