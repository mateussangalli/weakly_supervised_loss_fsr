import argparse
import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from keras.callbacks import CSVLogger
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam

from utils.combined_loss import CombinedLoss
from utils.data_augmentation import resize_inputs
from utils.data_generation import crop_generator
from utils.data_loading import read_dataset, read_dataset_pseudo
from utils.directional_relations import PRPDirectionalPenalty
from utils.jaccard_loss import OneHotMeanIoU
from utils.unet import UNetBuilder
from utils.utils import crop_to_multiple_of

parser = argparse.ArgumentParser()
# data dir arguments
parser.add_argument("pseudo_labels_dir", type=str)
parser.add_argument("--data_root", type=str, default="../prp_loreal_data")
parser.add_argument("--runs_dir", type=str, default="pseudo_labels_runs")

# training arguments
# WARN: make sure that batch_size_labeled divides batch_size_pseudo
parser.add_argument("--batch_size_labeled", type=int, default=4)
parser.add_argument("--batch_size_pseudo", type=int, default=12)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--starting_lr", type=float, default=1e-5)
parser.add_argument("--val_freq", type=int, default=1)

# crop generator arguments
parser.add_argument("--crop_size", type=int, default=192)
# WARN: please choose a number of crops_per_image_pseudo is a
#   multiple of the number of labeled train data (currently 5)
parser.add_argument("--crops_per_image_pseudo", type=int, default=5)
parser.add_argument("--min_scale", type=float, default=-1.0)
parser.add_argument("--max_scale", type=float, default=1.3)

# architecture arguments
parser.add_argument("--filters_start", type=int, default=8)
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--bn_momentum", type=float, default=0.85)

# loss function arguments
parser.add_argument("--max_weight", type=float, default=1.0)
parser.add_argument("--increase_epochs", type=int, default=5)
parser.add_argument("--strel_size", type=int, default=3)
parser.add_argument("--strel_spread", type=int, default=2)
parser.add_argument("--strel_iterations", type=int, default=10)

# verbose
parser.add_argument("--verbose", type=int, default=2)

args = parser.parse_args()

if args.min_scale < 0.0:
    scale_range = (1.0 / args.max_scale, args.max_scale)
else:
    scale_range = (args.min_scale, args.max_scale)

weight_str = f'{args.max_weight:.4f}'.replace('.', 'p')
run_name = f'weight{weight_str}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
run_dir = os.path.join(args.runs_dir, run_name)

# load data
train_images = os.listdir(os.path.join(args.data_root, "train", "images"))
train_images_labeled = [train_images[3], train_images[4],
                        train_images[5], train_images[34], train_images[64]]

data_train = read_dataset(args.data_root, "train", train_images_labeled)
data_pseudo = read_dataset_pseudo(
    args.data_root, "train", args.pseudo_labels_dir)
data_val = read_dataset(args.data_root, "val")
data_val = [
    (crop_to_multiple_of(im, 2**args.depth),
     crop_to_multiple_of(gt, 2**args.depth)) for (im, gt) in data_val
]

# NOTE: we want the same number of labeled and pseudo-labeled data batches
# the number of batches is ceil(len(data) * crops_per_image / batch_size)
# Everything else is determined, so we must find crops_per_image_labeled as follows
crops_per_image_labeled = len(data_pseudo)
crops_per_image_labeled *= args.crops_per_image_pseudo
crops_per_image_labeled *= args.batch_size_labe
crops_per_image_labeled //= len(data_train) * args.batch_size_labeled


def gen_train_labeled():
    return crop_generator(data_train, args.crop_size, crops_per_image_labeled, scale_range)


def gen_train_pseudo():
    return crop_generator(data_pseudo, args.crop_size, args.crops_per_image_pseudo, scale_range)


def gen_val():
    for image, label in data_val:
        image = image.astype(np.float32) / 255.0
        yield image, label


samples_per_epoch = len(data_train) * len(data_pseudo)
steps_per_epoch = int(np.ceil(samples_per_epoch / args.batch_size))

ds_train_labeled = tf.data.Dataset.from_generator(
    gen_train_labeled,
    output_types=(tf.float32, tf.int32),
    output_shapes=((None, None, 3), (None, None)),
)
ds_train_labeled = ds_train_labeled.shuffle(samples_per_epoch)
ds_train_labeled = ds_train_labeled.map(
    lambda im, gt: resize_inputs(im, gt, args.crop_size),
    num_parallel_calls=tf.data.AUTOTUNE,
)
ds_train_labeled = ds_train_labeled.map(
    lambda im, gt: (im, tf.one_hot(gt, 3)), num_parallel_calls=tf.data.AUTOTUNE
)
ds_train_labeled = ds_train_labeled.batch(args.batch_size_labeled)
ds_train_labeled = ds_train_labeled.repeat()


ds_train_pseudo = tf.data.Dataset.from_generator(
    gen_train_pseudo,
    output_types=(tf.float32, tf.int32),
    output_shapes=((None, None, 3), (None, None)),
)
ds_train_pseudo = ds_train_pseudo.shuffle(samples_per_epoch)
ds_train_pseudo = ds_train_pseudo.map(
    lambda im, gt: resize_inputs(im, gt, args.crop_size),
    num_parallel_calls=tf.data.AUTOTUNE,
)
ds_train_pseudo = ds_train_pseudo.map(
    lambda im, gt: (im, tf.one_hot(gt, 3)), num_parallel_calls=tf.data.AUTOTUNE
)
ds_train_pseudo = ds_train_pseudo.batch(args.batch_size_pseudo)
ds_train_pseudo = ds_train_pseudo.repeat()

ds_zip = tf.data.Dataset.zip((ds_train_labeled, ds_train_pseudo))
ds_train = ds_zip.map(lambda a, b: (tf.concat((a[0], b[0]), 0), tf.concat((a[1], b[1]), 0)),
                       num_parallel_calls=tf.data.AUTOTUNE)

ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


ds_val = tf.data.Dataset.from_generator(
    gen_val, output_types=(tf.float32, tf.int32))
ds_val = ds_val.map(
    lambda im, gt: (im, tf.one_hot(gt, 3)), num_parallel_calls=tf.data.AUTOTUNE
)
ds_val = ds_val.batch(1)
ds_val = ds_val.prefetch(tf.data.AUTOTUNE)
ds_val = ds_val.repeat()

# create run dir
os.makedirs(run_dir, exist_ok=True)
params_path = os.path.join(run_dir, 'params.json')
args_dict = vars(args)
with open(params_path, 'w') as fp:
    json.dump(args_dict, fp)


# create model
model = UNetBuilder(
    (None, None, 3),
    args.filters_start,
    args.depth,
    normalization="batch",
    normalize_all=False,
    batch_norm_momentum=args.bn_momentum,
).build()
directional_loss = PRPDirectionalPenalty(args.strel_size,
                                         args.strel_spread,
                                         args.strel_iterations)


def directional_loss_metric(y, y_pred, **kwargs):
    return directional_loss(y_pred)


crossentropy = CategoricalCrossentropy(from_logits=False)


def crossentropy_metric(y_true, y_pred, **kwargs):
    return crossentropy(y_true, y_pred)


loss_fn = CombinedLoss(
    CategoricalCrossentropy(from_logits=False),
    PRPDirectionalPenalty(args.strel_size,
                          args.strel_spread,
                          args.strel_iterations),
    args.increase_epochs,
    args.max_weight,
)


model.compile(
    optimizer=Adam(args.starting_lr),
    loss=loss_fn,
    metrics=[
        OneHotMeanIoU(3),
        crossentropy_metric,
        directional_loss_metric,
    ],
)

model.fit(
    ds_train_labeled,
    steps_per_epoch=steps_per_epoch,
    validation_data=ds_val,
    validation_steps=len(data_val),
    epochs=args.epochs,
    callbacks=[loss_fn.callback,
               CSVLogger(os.path.join(run_dir, 'training_history.csv'))],
    verbose=args.verbose,
    validation_freq=args.val_freq
)

model.save(os.path.join(run_dir, 'saved_model'))
