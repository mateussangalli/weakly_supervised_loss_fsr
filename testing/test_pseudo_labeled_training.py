import shutil
import unittest

import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import load_model

from utils.directional_relations import PRPDirectionalPenalty
from utils.unet import SemiSupPseudoUNetBuilder
from utils.unlabeled_training import SemiSupModelPseudo


class TestUnlabeledTraining(unittest.TestCase):
    def test_model_creation(self):
        input_layer = Input((5,))
        out_layer = Dense(10)(input_layer)
        model = SemiSupModelPseudo(input_layer, out_layer, alpha=1.)
        inputs = tf.random.uniform([20, 5])
        out = model(inputs)
        self.assertEqual(out.shape[0], 20)
        self.assertEqual(out.shape[1], 10)

    def test_model_compiling(self):
        input_layer = Input((5,))
        out_layer = Dense(10)(input_layer)
        model = SemiSupModelPseudo(input_layer, out_layer, alpha=1.)
        model.compile('adam', tf.keras.losses.MeanSquaredError(),
                      tf.keras.losses.MeanSquaredError())
        inputs = tf.random.uniform([20, 5])
        out = model(inputs)
        self.assertEqual(out.shape[0], 20)
        self.assertEqual(out.shape[1], 10)

    def test_model_fitting(self):
        input_layer = Input((5,))
        out_layer = Dense(10)(input_layer)
        model = SemiSupModelPseudo(input_layer, out_layer, alpha=1.)
        model.compile('adam', tf.keras.losses.MeanSquaredError(),
                      tf.keras.losses.MeanSquaredError())

        inputs_l = tf.random.uniform([20, 5])
        outputs_l = tf.random.uniform([20, 10])
        inputs_u = tf.random.uniform([40, 5])
        outputs_u = tf.random.uniform([40, 10])

        ds_l = tf.data.Dataset.from_tensor_slices(
            (inputs_l, outputs_l)).batch(4)
        ds_u = tf.data.Dataset.from_tensor_slices(
            (inputs_u, outputs_u)).batch(8)
        ds = tf.data.Dataset.zip((ds_l, ds_u))

        model.fit(ds, epochs=2, verbose=0)

    def test_model_io(self):
        input_layer = Input((5,))
        out_layer = Dense(10)(input_layer)
        model = SemiSupModelPseudo(input_layer, out_layer, alpha=1.)
        model.compile('adam', tf.keras.losses.MeanSquaredError(),
                      tf.keras.losses.MeanSquaredError())

        inputs_l = tf.random.uniform([20, 5])
        outputs_l = tf.random.uniform([20, 10])
        inputs_u = tf.random.uniform([40, 5])
        outputs_u = tf.random.uniform([40, 10])

        ds_l = tf.data.Dataset.from_tensor_slices(
            (inputs_l, outputs_l)).batch(4)
        ds_u = tf.data.Dataset.from_tensor_slices(
            (inputs_u, outputs_u)).batch(8)
        ds = tf.data.Dataset.zip((ds_l, ds_u))

        model.fit(ds, epochs=20, verbose=0)
        weights_b4 = model.get_weights()

        model.save("test_model")
        custom_objects = {"SemiSupModelPseudo": SemiSupModelPseudo}
        model2 = load_model(
            "test_model", custom_objects=custom_objects, compile=False)
        shutil.rmtree("test_model", ignore_errors=False, onerror=None)

        weights_after = model2.get_weights()

        for w1, w2 in zip(weights_b4, weights_after):
            dist = np.sum(np.abs(w1 - w2))
            self.assertAlmostEqual(dist, 0.)

    def test_model_eval(self):
        input_layer = Input((5,))
        out_layer = Dense(10)(input_layer)
        model = SemiSupModelPseudo(input_layer, out_layer, alpha=1.)
        model.compile('adam', tf.keras.losses.MeanSquaredError(
        ), tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.losses.MeanSquaredError()])

        inputs_l = tf.random.uniform([20, 5])
        outputs_l = tf.random.uniform([20, 10])
        inputs_u = tf.random.uniform([40, 5])
        outputs_u = tf.random.uniform([40, 10])
        inputs_l_val = tf.random.uniform([20, 5])
        outputs_l_val = tf.random.uniform([20, 10])

        ds_l = tf.data.Dataset.from_tensor_slices(
            (inputs_l, outputs_l)).batch(4)
        ds_u = tf.data.Dataset.from_tensor_slices(
            (inputs_u, outputs_u)).batch(8)
        ds = tf.data.Dataset.zip((ds_l, ds_u))
        ds_val = tf.data.Dataset.from_tensor_slices(
            (inputs_l_val, outputs_l_val)).batch(4)

        model.fit(ds, epochs=2, validation_data=ds_val)
        model.evaluate(ds_val)

    def test_schedule(self):
        def alpha_schedule(t):
            return tf.cast(t // 5, tf.float32)

        def loss1(y, y_pred): return tf.reduce_sum(tf.abs(y_pred + y)) * 1e-10
        def loss2(y, y_pred): return tf.reduce_sum(
            tf.abs(y_pred + y)) * 1e-10 + 1.

        input_layer = Input((5,))
        out_layer = Dense(10)(input_layer)
        model = SemiSupModelPseudo(input_layer, out_layer, alpha=alpha_schedule)
        model.compile('adam', loss1, loss2)

        inputs_l = tf.random.uniform([20, 5])
        outputs_l = tf.random.uniform([20, 10])
        inputs_u = tf.random.uniform([40, 5])
        outputs_u = tf.random.uniform([40, 10])

        ds_l = tf.data.Dataset.from_tensor_slices(
            (inputs_l, outputs_l)).batch(4)
        ds_u = tf.data.Dataset.from_tensor_slices(
            (inputs_u, outputs_u)).batch(8)
        ds = tf.data.Dataset.zip((ds_l, ds_u))

        history = model.fit(ds, epochs=15, verbose=0)
        expected = np.arange(15).astype(np.float32)
        results = np.array(history.history['total_loss'])
        dist = np.sum(np.abs(expected - results))
        self.assertAlmostEqual(dist, 0.)

    def test_unet(self):
        model = SemiSupPseudoUNetBuilder(
            (32, 32, 5), 2, 3, output_channels=3).build()

        def loss2(pred): return tf.reduce_sum(tf.abs(pred))
        model.compile('adam', tf.keras.losses.MeanSquaredError(),
                      tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.losses.MeanSquaredError()])

        inputs_l = tf.random.uniform([20, 32, 32, 5])
        outputs_l = tf.random.uniform([20, 32, 32, 3])
        inputs_u = tf.random.uniform([40, 32, 32, 5])
        outputs_u = tf.random.uniform([40, 32, 32, 3])
        inputs_l_val = tf.random.uniform([20, 32, 32, 5])
        outputs_l_val = tf.random.uniform([20, 32, 32, 3])

        ds_l = tf.data.Dataset.from_tensor_slices(
            (inputs_l, outputs_l)).batch(4)
        ds_u = tf.data.Dataset.from_tensor_slices(
            (inputs_u, outputs_u)).batch(8)
        ds = tf.data.Dataset.zip((ds_l, ds_u))
        ds_val = tf.data.Dataset.from_tensor_slices(
            (inputs_l_val, outputs_l_val)).batch(4)

        model.fit(ds, epochs=2, validation_data=ds_val, verbose=0)
        model.evaluate(ds_val)


if __name__ == "__main__":
    unittest.main()
