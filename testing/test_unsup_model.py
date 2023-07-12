import shutil
import unittest

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import load_model

from utils.size_regularization import LogBarrierHeight, LogBarrierHeightRatio
from utils.unlabeled_training import UnsupModel
from utils.unet import UnsupUNetBuilder


class UpdatableLoss(tf.keras.regularizers.Regularizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.t = tf.Variable(0., trainable=False)

    def update(self, step):
        self.t.assign(tf.math.floor(tf.cast(step, tf.float32) / 5.))

    def __call__(self, y_pred):
        return tf.reduce_sum(y_pred) * 1e-10 + self.t


def get_and_compile_model():
    input_layer = Input((5,))
    out_layer = Dense(10)(input_layer)
    model = UnsupModel(input_layer, out_layer)
    def loss1(pred): return tf.reduce_max(tf.abs(pred))
    def loss2(pred): return tf.reduce_sum(tf.abs(pred))
    model.compile('adam',
                  {'loss1': loss1, 'loss2': loss2},
                  {'loss1': lambda _: 1., 'loss2': lambda _: 1.})
    return model


def get_dataset():
    data = tf.random.uniform([40, 5])
    ds = tf.data.Dataset.from_tensor_slices(data).batch(8)
    return ds


class TestUnlabeledTraining(unittest.TestCase):
    def test_model_creation(self):
        input_layer = Input((5,))
        out_layer = Dense(10)(input_layer)
        model = UnsupModel(input_layer, out_layer)
        inputs = tf.random.uniform([20, 5])
        out = model(inputs)
        self.assertEqual(out.shape[0], 20)
        self.assertEqual(out.shape[1], 10)

    def test_model_compiling(self):
        model = get_and_compile_model()
        inputs = tf.random.uniform([20, 5])
        out = model(inputs)
        self.assertEqual(out.shape[0], 20)
        self.assertEqual(out.shape[1], 10)

    def test_model_fitting(self):
        model = get_and_compile_model()
        ds = get_dataset()
        model.fit(ds, epochs=2, verbose=0)

    def test_model_io(self):
        model = get_and_compile_model()
        ds = get_dataset()

        model.fit(ds, epochs=20, verbose=0)
        weights_b4 = model.get_weights()

        model.save("test_model")
        custom_objects = {"UnsupModel": UnsupModel}
        model2 = load_model(
            "test_model", custom_objects=custom_objects, compile=False)
        shutil.rmtree("test_model", ignore_errors=False, onerror=None)

        weights_after = model2.get_weights()

        for w1, w2 in zip(weights_b4, weights_after):
            dist = np.sum(np.abs(w1 - w2))
            self.assertAlmostEqual(dist, 0.)

    def test_model_eval(self):
        model = get_and_compile_model()
        ds = get_dataset()

        inputs_l_val = tf.random.uniform([20, 5])
        outputs_val = tf.random.uniform([20, 10])

        ds_val = tf.data.Dataset.from_tensor_slices((inputs_l_val, outputs_val)).batch(4)

        model.fit(ds, epochs=2, validation_data=ds_val)
        model.evaluate(ds_val)

    def test_schedule(self):
        def alpha_schedule1(t):
            return tf.random.uniform([])

        def alpha_schedule2(t):
            return tf.cast(t // 5, tf.float32)

        def loss1(y_pred): return tf.reduce_sum(tf.abs(y_pred)) * 1e-10
        def loss2(y_pred): return tf.reduce_sum(tf.abs(y_pred)) * 1e-10 + 1.

        input_layer = Input((5,))
        out_layer = Dense(10)(input_layer)
        model = UnsupModel(input_layer, out_layer)
        model.compile('adam',
                      {'loss1': loss1, 'loss2': loss2},
                      {'loss1': alpha_schedule1, 'loss2': alpha_schedule2}
                      )

        ds = get_dataset()

        history = model.fit(ds, epochs=15, verbose=0)
        expected = np.arange(15).astype(np.float32)
        results = np.array(history.history['total_loss'])
        dist = np.sum(np.abs(expected - results))
        self.assertAlmostEqual(dist, 0.)

    def test_update(self):
        def alpha_schedule1(t):
            return 1.

        def alpha_schedule2(t):
            return 1.

        def loss1(y_pred): return tf.reduce_sum(tf.abs(y_pred)) * 1e-10

        loss2 = UpdatableLoss()

        input_layer = Input((5,))
        out_layer = Dense(10)(input_layer)
        model = UnsupModel(input_layer, out_layer)
        model.compile('adam',
                      {'loss1': loss1, 'loss2': loss2},
                      {'loss1': alpha_schedule1, 'loss2': alpha_schedule2}
                      )

        ds = get_dataset()

        history = model.fit(ds, epochs=15, verbose=0)
        expected = np.arange(15).astype(np.float32)
        results = np.array(history.history['total_loss'])
        dist = np.sum(np.abs(expected - results))
        self.assertAlmostEqual(dist, 0.)

    def test_unet(self):
        def alpha_schedule1(t):
            return tf.random.uniform([])

        def alpha_schedule2(t):
            return tf.cast(t // 5, tf.float32)

        model = UnsupUNetBuilder((32, 32, 5), 2, 3,
                                 output_channels=3).build()
        model.compile('adam',
                      {'loss1': UpdatableLoss(), 'loss2': UpdatableLoss()},
                      {'loss1': alpha_schedule1, 'loss2': alpha_schedule2},
                      metrics=['mse'])

        inputs = tf.random.uniform([40, 32, 32, 5])
        inputs_l_val = tf.random.uniform([20, 32, 32, 5])
        outputs_val = tf.random.uniform([20, 32, 32, 3])

        ds = tf.data.Dataset.from_tensor_slices(inputs).batch(8)
        ds_val = tf.data.Dataset.from_tensor_slices((inputs_l_val, outputs_val)).batch(4)

        model.fit(ds, epochs=2, validation_data=ds_val, verbose=0)
        model.evaluate(ds_val)

    def test_log_barrier_height(self):
        def alpha_schedule(step):
            return 1.

        def t_schedule(step):
            return 1. + tf.cast(step, tf.float32) / 10
        model = UnsupUNetBuilder((32, 32, 5), 2, 3, output_channels=3).build()
        model.compile(tf.keras.optimizers.SGD(learning_rate=1e-2),
                      {'loss1': LogBarrierHeight(0, 0.1, 1., t_schedule),
                       'loss2': LogBarrierHeight(1, 0.1, 3., t_schedule)},
                      {'loss1': alpha_schedule,
                       'loss2': alpha_schedule},
                      metrics=['mse'])

        inputs = tf.zeros([1, 32, 32, 5])
        inputs_l_val = tf.random.uniform([20, 32, 32, 5])
        outputs_val = tf.random.uniform([20, 32, 32, 3])

        ds = tf.data.Dataset.from_tensor_slices(inputs).batch(1)
        ds_val = tf.data.Dataset.from_tensor_slices((inputs_l_val, outputs_val)).batch(4)

        model.fit(ds, epochs=200, validation_data=ds_val, verbose=0)
        model.evaluate(ds_val)
        out = model(inputs[:1, ...])
        out = np.array(out)[0, ...]
        out_height = np.sum(out, 0)
        out_mean_height = np.mean(out_height, 0)
        print(f'{out_mean_height=}')
        self.assertGreaterEqual(out_mean_height[0], 0.1)
        self.assertLessEqual(out_mean_height[0], 1.1)
        self.assertLessEqual(out_mean_height[1], 3.1)

    def test_log_barrier_ratio(self):
        def alpha_schedule(step):
            return 1.

        def t_schedule(step):
            return 1. + tf.cast(step, tf.float32) / 10.
        model = UnsupUNetBuilder((32, 32, 5), 2, 3, output_channels=3).build()
        model.compile(tf.keras.optimizers.SGD(learning_rate=1e-2),
                      {'loss1': LogBarrierHeightRatio(1, 2, 3., 5., t_schedule)},
                      {'loss1': alpha_schedule},
                      metrics=['mse'])

        inputs = tf.zeros([1, 32, 32, 5])
        inputs_l_val = tf.random.uniform([20, 32, 32, 5])
        outputs_val = tf.random.uniform([20, 32, 32, 3])

        ds = tf.data.Dataset.from_tensor_slices(inputs).batch(1)
        ds_val = tf.data.Dataset.from_tensor_slices((inputs_l_val, outputs_val)).batch(4)

        model.fit(ds, epochs=200, validation_data=ds_val, verbose=0)
        model.evaluate(ds_val)
        out = model(inputs[:1, ...])
        out = np.array(out)[0, ...]
        out_height = np.sum(out, 0)
        out_ratio = out_height[:, 1] / out_height[:, 2]
        out_mean_ratio = np.mean(out_ratio)
        print(f'{out_mean_ratio=}')
        self.assertGreaterEqual(out_mean_ratio, 2.9)
        self.assertLessEqual(out_mean_ratio, 5.1)


if __name__ == "__main__":
    unittest.main()
