import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.data_generation import get_tf_train_dataset, get_single_crop_dataset
from utils.data_loading import read_dataset
from utils.utils import crop_to_multiple_of
from labeled_images import LABELED_IMAGES, LABELED_IMAGES_VAL

SC = 0
LED = 2

parser = argparse.ArgumentParser()
# data dir arguments
parser.add_argument("--data_root", type=str, default="~/weak_supervision_data")
parser.add_argument("--runs_dir", type=str, default="no_pseudo_runs")

# training arguments
# WARN: make sure that batch_size_labeled divides batch_size_unlabeled
parser.add_argument("--crop_position", nargs='+', type=int, default=(100, 100))
parser.add_argument("--batch_size_labeled", type=int, default=32)
parser.add_argument("--batch_size_unlabeled", type=int, default=96)
parser.add_argument("--rotation_angle", type=float, default=np.pi/8.)
parser.add_argument("--hue_jitter", type=float, default=0.)
parser.add_argument("--sat_jitter", type=float, default=0.)
parser.add_argument("--val_jitter", type=float, default=0.)
parser.add_argument("--color_transfer_probability", type=float, default=0.5)
parser.add_argument("--noise_value", type=float, default=0.02)

# crop generator arguments
parser.add_argument("--crop_size", type=int, default=256)
parser.add_argument("--crops_per_image_unlabeled", type=int, default=1)
parser.add_argument("--min_scale", type=float, default=-1.0)
parser.add_argument("--max_scale", type=float, default=1.3)

args = parser.parse_args()


# load data
# load data
crop_pos = tuple(args.crop_position)
data_train = read_dataset(args.data_root, "train", LABELED_IMAGES[:1])
image_train = data_train[0][0][crop_pos[0]:crop_pos[0] + args.crop_size, crop_pos[1]:crop_pos[1] + args.crop_size, :]
image_train = image_train.astype(np.float32) / 255.
label_train = data_train[0][1][crop_pos[0]:crop_pos[0] + args.crop_size, crop_pos[1]:crop_pos[1] + args.crop_size]
label_train = label_train.astype(np.int32)

data_unlabeled = read_dataset(args.data_root, "train")
# just to be sure...
data_unlabeled = [(x, np.zeros_like(y)) for x, y in data_unlabeled]

data_val = read_dataset(args.data_root, "val", LABELED_IMAGES_VAL)
data_val = [
    (crop_to_multiple_of(im, 32),
     crop_to_multiple_of(gt, 32)) for (im, gt) in data_val
]


samples_per_epoch_unlabeled = args.crops_per_image_unlabeled * len(data_unlabeled)
steps_per_epoch_unlabeled = int(np.ceil(samples_per_epoch_unlabeled / args.batch_size_unlabeled))
steps_per_epoch = steps_per_epoch_unlabeled

# set params for the data iterator
params = {
    "min_scale": args.min_scale,
    "max_scale": args.max_scale,
    "crop_size": args.crop_size,
    "rotation_angle": args.rotation_angle,
    "hue_jitter": args.hue_jitter,
    "sat_jitter": args.sat_jitter,
    "val_jitter": args.val_jitter,
    "color_transfer_probability": args.color_transfer_probability,
    "noise_value": args.noise_value
}
params_labeled = params.copy()
params_labeled['batch_size'] = args.batch_size_labeled

params_unlabeled = params.copy()
params_unlabeled['crops_per_image'] = args.crops_per_image_unlabeled
params_unlabeled['batch_size'] = args.batch_size_unlabeled

# load labeled dataset and take the right amount of batches per epoch
ds_train_labeled: tf.data.Dataset = get_single_crop_dataset(image_train, label_train, params_labeled)
ds_train_labeled = ds_train_labeled.take(steps_per_epoch).repeat()

# load unlabeled dataset, remove labels and take the right amount of batches per epoch
ds_train_unlabeled: tf.data.Dataset = get_tf_train_dataset(data_unlabeled, params_unlabeled)
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
