import numpy as np
import tensorflow as tf

from utils.directional_relations import PRPDirectionalPenalty
from utils.unet import SemiSupUNetBuilder


def alpha_schedule(step):
    return 0. * tf.cast(step, tf.float32)


def t_schedule(step):
    return 1. + tf.cast(step, tf.float32) * .1


penalty = PRPDirectionalPenalty(5, 1)
# loss_unlab = PRPDirectionalLogBarrier(
#     5, 1, t_schedule, t=1., barrier_threshold=.3
# )

inputs_l = np.zeros([10, 32, 32, 5])
outputs_l = np.zeros([10, 32, 32, 3], np.float32)
outputs_l[:, 10:15, :, 0] = 1.
outputs_l[:, 5:10, :, 2] = 1.
outputs_l[..., 1] = 1. - outputs_l[..., 0] - outputs_l[..., 2]
inputs_l[..., :3] = outputs_l - .5
outputs_l = tf.constant(outputs_l)
inputs_l = tf.constant(inputs_l)
inputs_u = tf.zeros([20, 32, 32, 5])
ds_l = tf.data.Dataset.from_tensor_slices((inputs_l, outputs_l)).batch(2)
ds_u = tf.data.Dataset.from_tensor_slices(inputs_u).batch(4)
ds = tf.data.Dataset.zip((ds_l, ds_u))


model = SemiSupUNetBuilder((32, 32, 5), 2, 3, output_channels=3).build()
model.compile(tf.keras.optimizers.SGD(1e-2),
              tf.keras.losses.CategoricalCrossentropy(),
              {'loss_unlab': penalty},
              {'loss_unlab': alpha_schedule},
              metrics=['mse'])

model.fit(ds, epochs=200, verbose=2)
