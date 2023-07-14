import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.data_generation import get_online_dataset
from utils.data_loading import read_dataset
from utils.utils import crop_to_multiple_of
from labeled_images import LABELED_IMAGES, LABELED_IMAGES_VAL

SC = 0
LED = 2

parser = argparse.ArgumentParser()
# data dir arguments
parser.add_argument("--run_id", type=str, default='')
parser.add_argument("--data_root", type=str, default="~/weak_supervision_data")
parser.add_argument("--runs_dir", type=str, default="no_pseudo_runs")

# training arguments
# WARN: make sure that batch_size_labeled divides batch_size_unlabeled
parser.add_argument("--num_images_labeled", type=int, default=3)
parser.add_argument("--rotation_angle", type=float, default=np.pi/8.)
parser.add_argument("--hue_jitter", type=float, default=0.)
parser.add_argument("--sat_jitter", type=float, default=0.)
parser.add_argument("--val_jitter", type=float, default=0.)

# crop generator arguments
parser.add_argument("--crop_size", type=int, default=256)
parser.add_argument("--crops_per_image_unlabeled", type=int, default=1)
parser.add_argument("--min_scale", type=float, default=-1.0)
parser.add_argument("--max_scale", type=float, default=1.3)

args = parser.parse_args()


# load data
data_train = read_dataset(args.data_root, "train", LABELED_IMAGES[:args.num_images_labeled])
data_unlabeled = read_dataset(args.data_root, "train")
# just to be sure...
data_unlabeled = [(x, np.zeros_like(y)) for x, y in data_unlabeled]

data_val = read_dataset(args.data_root, "val", LABELED_IMAGES_VAL)
data_val = [
    (crop_to_multiple_of(im, 32),
     crop_to_multiple_of(gt, 32)) for (im, gt) in data_val
]


steps_per_epoch = len(data_unlabeled)

params_labeled = {
    "min_scale": args.min_scale,
    "max_scale": args.max_scale,
    "rotation_angle": args.rotation_angle,
    "color_transfer_probability": .9,
    "noise_value": .02,
}
params_unlabeled = {
    "min_scale": args.min_scale,
    "max_scale": args.max_scale,
    "rotation_angle": args.rotation_angle,
    "color_transfer_probability": .9,
    "noise_value": .02,
}

# load labeled dataset and take the right amount of batches per epoch
ds_train_labeled: tf.data.Dataset = get_online_dataset(data_train, params_labeled)
ds_train_labeled = ds_train_labeled.take(steps_per_epoch).repeat()

# load unlabeled dataset, remove labels and take the right amount of batches per epoch
ds_train_unlabeled: tf.data.Dataset = get_online_dataset(data_unlabeled, params_unlabeled)
ds_train_unlabeled = ds_train_unlabeled.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE)
ds_train_unlabeled = ds_train_unlabeled.take(steps_per_epoch).repeat()

ds_zip = tf.data.Dataset.zip((ds_train_labeled, ds_train_unlabeled))
ds_train = ds_zip.prefetch(tf.data.AUTOTUNE)

for (im_l, gt), im_u in ds_train:
    plt.subplot(131)
    plt.title('labeled')
    plt.imshow(im_l[0, ...])
    plt.subplot(132)
    plt.title('ground truth')
    plt.imshow(gt[0, ...])
    plt.subplot(133)
    plt.title('unlabeled')
    plt.imshow(im_u[0, ...])
    plt.show()
    assert im_l.shape[1] % 64 == 0
    assert gt.shape[1] == im_l.shape[1]
    assert im_u.shape[1] % 64 == 0

    assert im_l.shape[2] % 64 == 0
    assert gt.shape[2] == im_l.shape[2]
    assert im_u.shape[2] % 64 == 0
