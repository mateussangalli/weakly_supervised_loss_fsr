import argparse
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.data_generation import get_tf_train_dataset
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
parser.add_argument("--batch_size_labeled", type=int, default=32)
parser.add_argument("--batch_size_unlabeled", type=int, default=96)
parser.add_argument("--rotation_angle", type=float, default=np.pi/8.)

# crop generator arguments
parser.add_argument("--crop_size", type=int, default=160)
# WARN: please choose a number of crops_per_image_unlabeled such that
#  num_images_unlabeled * crops_per_image_unlabeled * batch_size_labeled is a multiple of
#  num_images_labeled * batch_size_unlabeled
parser.add_argument("--crops_per_image_unlabeled", type=int, default=1)
parser.add_argument("--min_scale", type=float, default=-1.0)
parser.add_argument("--max_scale", type=float, default=1.3)

args = parser.parse_args()

if args.min_scale < 0.0:
    scale_range = (1.0 / args.max_scale, args.max_scale)
else:
    scale_range = (args.min_scale, args.max_scale)

if args.run_id == '':
    weight_str = f'{args.max_weight:.4f}'.replace('.', 'p')
    run_name = f'weight{weight_str}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
else:
    run_name = args.run_id
run_dir = os.path.join(args.runs_dir, run_name)

# load data
data_train = read_dataset(args.data_root, "train", LABELED_IMAGES[args.num_images_labeled])
data_unlabeled = read_dataset(args.data_root, "train")
# just to be sure...
data_unlabeled = [(x, np.zeros_like(y)) for x, y in data_unlabeled]

data_val = read_dataset(args.data_root, "val", LABELED_IMAGES_VAL)
data_val = [
    (crop_to_multiple_of(im, 2**args.depth),
     crop_to_multiple_of(gt, 2**args.depth)) for (im, gt) in data_val
]


# NOTE: we want the same number of labeled and unlabeled data batches
# NOTE: the number of batches is ceil(len(data) * crops_per_image / batch_size)
# NOTE: Everything else is determined, so we must find crops_per_image_labeled
crops_per_image_labeled = len(data_unlabeled)
crops_per_image_labeled *= args.crops_per_image_unlabeled
crops_per_image_labeled *= args.batch_size_labeled
crops_per_image_labeled //= len(data_train) * args.batch_size_unlabeled

samples_per_epoch_label = crops_per_image_labeled * len(data_train)
samples_per_epoch_unlabeled = args.crops_per_image_unlabeled * \
    len(data_unlabeled)
steps_per_epoch_labeled = int(np.ceil(samples_per_epoch_label / args.batch_size_labeled))
steps_per_epoch_unlabeled = int(np.ceil(samples_per_epoch_unlabeled / args.batch_size_unlabeled))
# verify that everything is at it should be
# if steps_per_epoch_unlabeled != steps_per_epoch_labeled:
#     msg1 = f'labeled: \n crops/image: {crops_per_image_labeled},' + \
#         'num images: {len(data_train)}, batch_size: {args.batch_size_labeled}'
#     msg2 = f'unlabeled: \n crops/image: {args.crops_per_image_unlabeled},' + \
#         ' num images: {len(data_unlabeled)}, batch_size: {args.batch_size_unlabeled}'
#     raise ValueError(
#         'number of steps is wrong! \n' + msg1 + '\n' + msg2
#     )
steps_per_epoch = min(steps_per_epoch_unlabeled, steps_per_epoch_labeled)

params_labeled = {
    "min_scale": args.min_scale,
    "max_scale": args.max_scale,
    "crop_size": args.crop_size,
    "rotation_angle": args.rotation_angle,
    "crops_per_image": crops_per_image_labeled,
    "batch_size": args.batch_size_labeled
}
params_unlabeled = {
    "min_scale": args.min_scale,
    "max_scale": args.max_scale,
    "crop_size": args.crop_size,
    "rotation_angle": args.rotation_angle,
    "crops_per_image": args.crops_per_image_unlabeled,
    "batch_size": args.batch_size_unlabeled
}

# load labeled dataset and take the right amount of batches per epoch
ds_train_labeled: tf.data.Dataset = get_tf_train_dataset(data_train, params_labeled)
ds_train_labeled = ds_train_labeled.take(steps_per_epoch).repeat()

# load unlabeled dataset, remove labels and take the right amount of batches per epoch
ds_train_unlabeled: tf.data.Dataset = get_tf_train_dataset(data_unlabeled, params_unlabeled)
ds_train_unlabeled = ds_train_unlabeled.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE)
ds_train_unlabeled = ds_train_unlabeled.take(steps_per_epoch).repeat()

ds_zip = tf.data.Dataset.zip((ds_train_labeled, ds_train_unlabeled))
ds_train = ds_zip.prefetch(tf.data.AUTOTUNE)

for (im_l, gt), im_u in ds_train:
    plt.suplot(131)
    plt.imshow(im_l)
    plt.suplot(132)
    plt.imshow(gt)
    plt.suplot(133)
    plt.imshow(im_u)
