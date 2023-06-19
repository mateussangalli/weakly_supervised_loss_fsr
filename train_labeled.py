import os

import numpy as np
import tensorflow as tf
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam

from utils.combined_loss import combined_loss
from utils.data_augmentation import resize_inputs
from utils.data_generation import crop_generator
from utils.data_loading import read_dataset
from utils.directional_relations import PRPDirectionalPenalty
from utils.jaccard_loss import OneHotMeanIoU
from utils.unet import UNetBuilder
from utils.utils import crop_to_multiple_of

DATA_ROOT = "~/weak_supervision_data"
CROP_SIZE = 192
CROPS_PER_IMAGE = 8
SCALE_RANGE = (1.0 / 1.3, 1.3)
BATCH_SIZE = 16
EPOCHS = 100

LEARNING_RATE = 5e-5

FILTERS_START = 8
DEPTH = 4
BN_MOMENTUM = 0.85

MAX_WEIGHT = 1.0
INCREASE_EPOCHS = 50


# load data

train_images = os.listdir(os.path.join(DATA_ROOT, "train", "images"))
train_images = [train_images[3], train_images[4], train_images[5]]

data_train = read_dataset(DATA_ROOT, "train", train_images)
data_val = read_dataset(DATA_ROOT, "val")
data_val = [
    (crop_to_multiple_of(im, 32), crop_to_multiple_of(gt, 32)) for (im, gt) in data_val
]


def gen_train():
    return crop_generator(data_train, CROP_SIZE, CROPS_PER_IMAGE, SCALE_RANGE)


def gen_val():
    for image, label in data_val:
        image = image.astype(np.float32) / 255.0
        yield image, label


samples_per_epoch = len(data_train) * CROPS_PER_IMAGE
steps_per_epoch = int(np.ceil(samples_per_epoch / BATCH_SIZE))

ds_train = tf.data.Dataset.from_generator(
    gen_train,
    output_types=(tf.float32, tf.int32),
    output_shapes=((None, None, 3), (None, None)),
)
ds_train = ds_train.shuffle(samples_per_epoch)
ds_train = ds_train.map(
    lambda im, gt: resize_inputs(im, gt, CROP_SIZE),
    num_parallel_calls=tf.data.AUTOTUNE,
)
ds_train = ds_train.map(
    lambda im, gt: (im, tf.one_hot(gt, 3)), num_parallel_calls=tf.data.AUTOTUNE
)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.repeat()
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)


ds_val = tf.data.Dataset.from_generator(gen_val, output_types=(tf.float32, tf.int32))
ds_val = ds_val.map(
    lambda im, gt: (im, tf.one_hot(gt, 3)), num_parallel_calls=tf.data.AUTOTUNE
)
ds_val = ds_val.batch(1)
ds_val = ds_val.prefetch(tf.data.AUTOTUNE)
ds_val = ds_val.repeat()


# create model
model = UNetBuilder(
    (None, None, 3),
    FILTERS_START,
    DEPTH,
    normalization="batch",
    normalize_all=False,
    batch_norm_momentum=BN_MOMENTUM,
).build()
directional_loss = PRPDirectionalPenalty(3, 2, 5)


def directional_loss_metric(y, y_pred, **kwargs):
    return directional_loss(y_pred)


loss_fn, loss_callback = combined_loss(
    CategoricalCrossentropy(from_logits=False),
    PRPDirectionalPenalty(3, 2, 5),
    INCREASE_EPOCHS,
    MAX_WEIGHT,
)


model.compile(
    optimizer=Adam(LEARNING_RATE),
    loss=loss_fn,
    metrics=[
        OneHotMeanIoU(3),
        CategoricalCrossentropy(from_logits=False),
        directional_loss_metric,
    ],
)

model.fit(
    ds_train,
    steps_per_epoch=steps_per_epoch,
    # validation_data=ds_val,
    # validation_steps=len(data_val),
    epochs=EPOCHS,
    callbacks=[loss_callback],
)
