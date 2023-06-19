import os
import unittest

import numpy as np
import tensorflow as tf

from utils.data_augmentation import resize_inputs
from utils.data_generation import crop_generator
from utils.data_loading import read_dataset

DATA_ROOT = "~/weak_supervision_data"
CROP_SIZE = 192
CROPS_PER_IMAGE = 5
SCALE_RANGE = (1.0 / 1.3, 1.3)
BATCH_SIZE = 4
EPOCHS = 100

LEARNING_RATE = 5e-5

FILTERS_START = 8
DEPTH = 4
BN_MOMENTUM = 0.85

MAX_WEIGHT = 1.0
INCREASE_EPOCHS = 50

NUM_REPEAT = 2

class TestDataset(unittest.TestCase):
    def test_train_dataset(self):
        train_images = os.listdir(os.path.join(DATA_ROOT, "train", "images"))
        train_images = [train_images[3], train_images[4], train_images[5]]
        data_train = read_dataset(DATA_ROOT, "train", train_images)

        def gen_train():
            return crop_generator(data_train,
                                  CROP_SIZE,
                                  CROPS_PER_IMAGE,
                                  SCALE_RANGE)

        samples_per_epoch = len(data_train) * CROPS_PER_IMAGE
        steps_per_epoch = int(np.ceil(samples_per_epoch / BATCH_SIZE))

        ds_train = tf.data.Dataset.from_generator(
            gen_train,
            output_types=(tf.float32, tf.int32),
            output_shapes=((None, None, 3), (None, None)),
        )
        ds_train = ds_train.shuffle(steps_per_epoch)
        ds_train = ds_train.map(
            lambda im, gt: resize_inputs(im, gt, CROP_SIZE),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds_train = ds_train.map(
            lambda im, gt: (im, tf.one_hot(gt, 3)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        ds_train = ds_train.batch(BATCH_SIZE)
        ds_train = ds_train.repeat(NUM_REPEAT)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

        num_steps = 0
        for x, y in ds_train:
            self.assertEqual(x.shape[1], CROP_SIZE, "cropping incorrectly")
            self.assertEqual(y.shape[1], CROP_SIZE, "cropping incorrectly")
            self.assertEqual(x.shape[2], CROP_SIZE, "cropping incorrectly")
            self.assertEqual(y.shape[2], CROP_SIZE, "cropping incorrectly")

            self.assertEqual(x.shape[3], 3, "wrong number of channels")
            self.assertEqual(y.shape[3], 3, "wrong number of channels")
            num_steps += 1
        self.assertEqual(num_steps, steps_per_epoch * NUM_REPEAT, "wrong number of steps")

        num_steps = 0
        for x, y in ds_train:
            num_steps += 1
        self.assertEqual(num_steps, steps_per_epoch * NUM_REPEAT, "repeat not working")


if __name__ == "__main__":
    unittest.main()
