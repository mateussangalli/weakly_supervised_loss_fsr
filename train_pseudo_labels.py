import argparse
from datetime import datetime
import json
import os

from keras.callbacks import CSVLogger, LearningRateScheduler
from keras.losses import CategoricalCrossentropy
import numpy as np
import tensorflow as tf

from utils.combined_loss import CombinedLoss
from utils.data_generation import get_tf_train_dataset
from utils.data_loading import read_dataset, read_dataset_pseudo
from utils.directional_relations import PRPDirectionalPenalty
from utils.jaccard_loss import OneHotMeanIoU
from utils.unet import UNetBuilder
from utils.utils import crop_to_multiple_of

parser = argparse.ArgumentParser()
# data dir arguments
parser.add_argument("pseudo_labels_dir", type=str)
parser.add_argument("--run_id", type=str, default='')
parser.add_argument("--data_root", type=str, default="../prp_loreal_data")
parser.add_argument("--runs_dir", type=str, default="pseudo_labels_runs")

# training arguments
# WARN: make sure that batch_size_labeled divides batch_size_pseudo
parser.add_argument("--num_images_labeled", type=int, default=3)
parser.add_argument("--batch_size_labeled", type=int, default=16)
parser.add_argument("--batch_size_pseudo", type=int, default=48)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--starting_lr", type=float, default=1e-4)
parser.add_argument("--val_freq", type=int, default=1)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--lr_decay_rate", type=float, default=0.03)
parser.add_argument("--rotation_angle", type=float, default=np.pi/8.)

# crop generator arguments
parser.add_argument("--crop_size", type=int, default=192)
# WARN: please choose a number of crops_per_image_pseudo such that
#  num_images_pseudo * crops_per_image_pseudo * batch_size_labeled is a multiple of
#  num_images_labeled * batch_size_pseudo
parser.add_argument("--crops_per_image_pseudo", type=int, default=3)
parser.add_argument("--min_scale", type=float, default=-1.0)
parser.add_argument("--max_scale", type=float, default=1.3)

# architecture arguments
parser.add_argument("--filters_start", type=int, default=8)
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--bn_momentum", type=float, default=0.85)

# loss function arguments
parser.add_argument("--max_weight", type=float, default=1.0)
parser.add_argument("--increase_epochs", type=int, default=10)
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

if args.run_id == '':
    weight_str = f'{args.max_weight:.4f}'.replace('.', 'p')
    run_name = f'weight{weight_str}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
else:
    run_name = args.run_id
run_dir = os.path.join(args.runs_dir, run_name)

# load data
train_images = os.listdir(os.path.join(args.data_root, "train", "images"))
if args.num_images_labeled > 0:
    train_images_labeled = [
        train_images[3],
        train_images[34],
        train_images[64],
        train_images[4],
        train_images[5],
    ][: args.num_images_labeled]
else:
    train_images_labeled = train_images


data_train = read_dataset(args.data_root, "train", train_images_labeled)
data_pseudo = read_dataset_pseudo(
    args.data_root, "train", args.pseudo_labels_dir)
data_val = read_dataset(args.data_root, "val")
data_val = [
    (crop_to_multiple_of(im, 2**args.depth),
     crop_to_multiple_of(gt, 2**args.depth)) for (im, gt) in data_val
]

# NOTE: we want the same number of labeled and pseudo-labeled data batches
# NOTE: the number of batches is ceil(len(data) * crops_per_image / batch_size)
# NOTE: Everything else is determined, so we must find crops_per_image_labeled
crops_per_image_labeled = len(data_pseudo)
crops_per_image_labeled *= args.crops_per_image_pseudo
crops_per_image_labeled *= args.batch_size_labeled
crops_per_image_labeled //= len(data_train) * args.batch_size_pseudo

samples_per_epoch_label = crops_per_image_labeled * len(data_train)
samples_per_epoch_pseudo = args.crops_per_image_pseudo * len(data_pseudo)
steps_per_epoch_labeled = int(
    np.ceil(samples_per_epoch_label / args.batch_size_labeled))
steps_per_epoch_pseudo = int(
    np.ceil(samples_per_epoch_pseudo / args.batch_size_pseudo))
# verify that everything is at it should be
if steps_per_epoch_pseudo != steps_per_epoch_labeled:
    msg1 = f'labeled: \n crops/image: {crops_per_image_labeled}, num images: {len(data_train)}, batch_size: {args.batch_size_labeled}'
    msg2 = f'pseudo: \n crops/image: {args.crops_per_image_pseudo}, num images: {len(data_pseudo)}, batch_size: {args.batch_size_pseudo}'
    raise ValueError(
        'number of steps is wrong! \n' + msg1 + '\n' + msg2
    )
steps_per_epoch = steps_per_epoch_pseudo

params_labeled = {
    "min_scale": args.min_scale,
    "max_scale": args.max_scale,
    "crop_size": args.crop_size,
    "rotation_angle": args.rotation_angle,
    "crops_per_image": crops_per_image_labeled,
    "batch_size": args.batch_size_labeled
}
params_pseudo = {
    "min_scale": args.min_scale,
    "max_scale": args.max_scale,
    "crop_size": args.crop_size,
    "rotation_angle": args.rotation_angle,
    "crops_per_image": args.crops_per_image_pseudo,
    "batch_size": args.batch_size_pseudo
}

ds_train_labeled = get_tf_train_dataset(data_train, params_labeled)
ds_train_pseudo = get_tf_train_dataset(data_pseudo, params_pseudo)
ds_zip = tf.data.Dataset.zip((ds_train_labeled, ds_train_pseudo))
ds_train = ds_zip.map(lambda a, b: (tf.concat((a[0], b[0]), 0), tf.concat((a[1], b[1]), 0)),
                      num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


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
    optimizer=tf.keras.optimizers.experimental.Adam(
        args.starting_lr, weight_decay=args.weight_decay),
    loss=loss_fn,
    metrics=[
        OneHotMeanIoU(3),
        crossentropy_metric,
        directional_loss_metric,
    ],
)


def schedule(epoch, lr):
    if epoch > 0:
        return lr * tf.exp(-args.lr_decay_rate)
    return lr


model.fit(
    ds_train,
    steps_per_epoch=steps_per_epoch,
    validation_data=ds_val,
    validation_steps=len(data_val),
    epochs=args.epochs,
    callbacks=[loss_fn.callback,
               CSVLogger(os.path.join(run_dir, 'training_history.csv')),
               LearningRateScheduler(schedule)],
    verbose=args.verbose,
    validation_freq=args.val_freq
)

model.save(os.path.join(run_dir, 'saved_model'))
