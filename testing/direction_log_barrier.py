import unittest
import numpy as np
import tensorflow as tf

from utils.unet import UnsupUNetBuilder
from utils.data_loading import read_label
from utils.directional_relations import PRPDirectionalLogBarrier

PRED_PATH = 'testing/image_name'


class TestDirectionLogBarrier(unittest.TestCase):
    def test_nan(self):
        pred = read_label(PRED_PATH, one_hot=True)
        pred += np.random.uniform(0., 0.1, pred.shape)
        pred = np.maximum(np.minimum(pred, 1.), 0.)
        def t_schedule(step): return 5.
        loss_fn = PRPDirectionalLogBarrier(20, 0, t_schedule, t=5.)
        loss_value = loss_fn(pred)
        loss_value = np.array(loss_value)
        is_nan = np.isnan(loss_value)
        is_nan = np.any(is_nan)
        self.assertFalse(is_nan)

    def test_fit(self):
        def alpha_schedule(step): return 1.
        def t_schedule(step): return 1. + tf.cast(step, tf.float32) / 10.
        model = UnsupUNetBuilder((32, 32, 5), 2, 3, output_channels=3).build()
        model.compile(tf.keras.optimizers.SGD(learning_rate=1e-2),
                      {'loss1': PRPDirectionalLogBarrier(20, 0, t_schedule, t=5.)},
                      {'loss1': alpha_schedule},
                      metrics=['mse'])

        inputs = tf.random.uniform([1, 32, 32, 5])

        ds = tf.data.Dataset.from_tensor_slices(inputs).batch(1)

        model.fit(ds, epochs=200, verbose=1)
        for weight in model.get_weights():
            self.assertFalse(np.any(np.isnan(weight)))


if __name__ == "__main__":
    unittest.main()
