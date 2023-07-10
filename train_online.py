import argparse
import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler
from tensorflow.keras.losses import CategoricalCrossentropy

from utils.data_generation import get_online_dataset
from utils.data_loading import read_dataset
from utils.directional_relations import PRPDirectionalPenalty
from utils.jaccard_loss import OneHotMeanIoU, jaccard_loss_mean_wrapper
from utils.size_regularization import QuadraticPenaltyHeight
from utils.unet import SemiSupUNetBuilder
from utils.utils import crop_to_multiple_of
from labeled_images import LABELED_IMAGES, LABELED_IMAGES_VAL

SC = 0
LED = 2

parser = argparse.ArgumentParser()
# data dir arguments
parser.add_argument("--run_id", type=str, default='')
parser.add_argument("--data_root", type=str, default="~/weak_supervision_data")
parser.add_argument("--runs_dir", type=str, default="runs/no_pseudo_runs")

# training arguments
parser.add_argument("--num_images_labeled", type=int, default=3)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--starting_lr", type=float, default=1e-3)
parser.add_argument("--warmup_epochs", type=int, default=20)
parser.add_argument("--lr_decay_rate", type=float, default=0.955)
parser.add_argument("--min_lr", type=float, default=1e-7)
parser.add_argument("--val_freq", type=int, default=1)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--rotation_angle", type=float, default=np.pi/8.)
parser.add_argument("--labeled_loss_fn", choices=['iou', 'crossentropy'], default='crossentropy')

# data generator arguments
parser.add_argument("--min_scale", type=float, default=-1.0)
parser.add_argument("--max_scale", type=float, default=1.4)
parser.add_argument("--hue_jitter", type=float, default=0.)
parser.add_argument("--sat_jitter", type=float, default=0.)
parser.add_argument("--val_jitter", type=float, default=0.)
parser.add_argument("--noise_value", type=float, default=0.)
parser.add_argument("--color_transfer_probability", type=float, default=0.)

# architecture arguments
parser.add_argument("--filters_start", type=int, default=8)
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--bn_momentum", type=float, default=0.9)

# unlabeled loss function arguments
parser.add_argument("--max_weight", type=float, default=1.0)
parser.add_argument("--increase_epochs", type=int, default=10)
parser.add_argument("--strel_size", type=int, default=20)
parser.add_argument("--strel_iterations", type=int, default=1)

parser.add_argument("--hmax_sc", type=float, default=70.)
parser.add_argument("--hmax_led", type=float, default=120.)
parser.add_argument("--height_reg_weight", type=float, default=0.)

# verbose
parser.add_argument("--verbose", type=int, default=2)

args = parser.parse_args()

if args.min_scale < 0.0:
    scale_range = (1.0 / args.max_scale, args.max_scale)
else:
    scale_range = (args.min_scale, args.max_scale)

if args.run_id == '':
    weight_str = f'{args.max_weight:.5f}'.replace('.', 'p')
    run_name = f'weight{weight_str}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
else:
    run_name = args.run_id
run_dir = os.path.join(args.runs_dir, run_name)

# load data
data_train = read_dataset(args.data_root, "train", LABELED_IMAGES[:args.num_images_labeled])
data_unlabeled = read_dataset(args.data_root, "train")
# just to be sure...
data_unlabeled = [(x, np.zeros_like(y)) for x, y in data_unlabeled]

data_val = read_dataset(args.data_root, "val", LABELED_IMAGES_VAL)
data_val = [
    (crop_to_multiple_of(im, 2**args.depth),
     crop_to_multiple_of(gt, 2**args.depth)) for (im, gt) in data_val
]


steps_per_epoch = len(data_unlabeled)

# set params for the data iterator
params = {
    "min_scale": args.min_scale,
    "max_scale": args.max_scale,
    "rotation_angle": args.rotation_angle,
    "hue_jitter": args.hue_jitter,
    "sat_jitter": args.sat_jitter,
    "val_jitter": args.val_jitter,
    "color_transfer_probability": args.color_transfer_probability,
    "noise_value": args.noise_value
}

# load labeled dataset and take the right amount of batches per epoch
ds_train_labeled: tf.data.Dataset = get_online_dataset(data_train, params)
ds_train_labeled = ds_train_labeled.take(steps_per_epoch).repeat()

# load unlabeled dataset, remove labels and take the right amount of batches per epoch
ds_train_unlabeled: tf.data.Dataset = get_online_dataset(data_unlabeled, params)
ds_train_unlabeled = ds_train_unlabeled.map(lambda x, y: x, num_parallel_calls=tf.data.AUTOTUNE)
ds_train_unlabeled = ds_train_unlabeled.take(steps_per_epoch).repeat()

ds_zip = tf.data.Dataset.zip((ds_train_labeled, ds_train_unlabeled))
ds_train = ds_zip.prefetch(tf.data.AUTOTUNE)


def gen_val():
    for image, label in data_val:
        image = image.astype(np.float32) / 255.0
        yield image, label


ds_val = tf.data.Dataset.from_generator(
    gen_val, output_types=(tf.float32, tf.int32))
ds_val = ds_val.map(
    lambda im, gt: (im, tf.one_hot(gt, 3)), num_parallel_calls=tf.data.AUTOTUNE
)
ds_val = ds_val.batch(1)
ds_val = ds_val.prefetch(tf.data.AUTOTUNE)
ds_val = ds_val.repeat()

# create run dir
try:
    os.makedirs(run_dir, exist_ok=False)
except FileExistsError as e:
    print(e)
    run_dir_tmp = run_dir
    i = 1
    while os.path.exists(run_dir_tmp):
        run_dir_tmp = run_dir + '_' + str(i)
        i += 1
    run_dir = run_dir_tmp
params_path = os.path.join(run_dir, 'params.json')
args_dict = vars(args)
with open(params_path, 'w') as fp:
    json.dump(args_dict, fp)


# define regularization weight schedules
increase_ratio = args.max_weight / float(args.increase_epochs)


def alpha_schedule1(t):
    return tf.minimum(tf.cast(t, tf.float32) * increase_ratio, args.max_weight)


def alpha_schedule2(t):
    return tf.minimum(tf.cast(t, tf.float32) * increase_ratio, args.height_reg_weight)


def alpha_schedule3(t):
    return tf.minimum(tf.cast(t, tf.float32) * increase_ratio, args.height_reg_weight)


alpha_schedule = [alpha_schedule1, alpha_schedule2, alpha_schedule3]

# create model
model = SemiSupUNetBuilder(
    (None, None, 3),
    args.filters_start,
    args.depth,
    normalization="batch",
    normalize_all=False,
    batch_norm_momentum=args.bn_momentum,
    alpha=alpha_schedule
).build()
directional_loss = PRPDirectionalPenalty(args.strel_size,
                                         args.strel_iterations)
height_penalty_sc = QuadraticPenaltyHeight(SC, args.hmax_sc)
height_penalty_led = QuadraticPenaltyHeight(LED, args.hmax_led)


def directional_loss_metric(y, y_pred, **kwargs): return directional_loss(y_pred)
def height_sc_metric(y, y_pred, **kwargs): return height_penalty_sc(y_pred)
def height_led_metric(y, y_pred, **kwargs): return height_penalty_led(y_pred)


if args.labeled_loss_fn == 'iou':
    loss_labeled = jaccard_loss_mean_wrapper()
else:
    loss_labeled = CategoricalCrossentropy(from_logits=False)

loss_unlab1 = PRPDirectionalPenalty(args.strel_size,
                                    args.strel_iterations)
loss_unlab2 = QuadraticPenaltyHeight(SC, args.hmax_sc)
loss_unlab3 = QuadraticPenaltyHeight(LED, args.hmax_led)


model.compile(
    tf.keras.optimizers.experimental.Adam(
        args.starting_lr, weight_decay=args.weight_decay),
    loss_labeled,
    [loss_unlab1,
     loss_unlab2,
     loss_unlab3],
    metrics=[
        OneHotMeanIoU(3),
        directional_loss_metric,
        height_sc_metric,
        height_led_metric
    ],
)


def lr_schedule(epoch, lr):
    if epoch > args.warmup_epochs:
        return max(lr * args.lr_decay_rate, args.min_lr)
    return lr


model.fit(
    ds_train,
    steps_per_epoch=steps_per_epoch,
    validation_data=ds_val,
    validation_steps=len(data_val),
    epochs=args.epochs,
    callbacks=[CSVLogger(os.path.join(run_dir, 'training_history.csv')),
               LearningRateScheduler(lr_schedule)],
    verbose=args.verbose,
    validation_freq=args.val_freq
)

model.save(os.path.join(run_dir, 'saved_model'))
