import unittest

import numpy as np
import tensorflow as tf

from utils.size_regularization import QuadraticPenaltyHeight, get_mean_height


class TestSizeRegularization(unittest.TestCase):
    def test_mean_height(self):
        im = np.zeros([1, 40, 40, 3], np.float32)
        im[:, 10:20, :, 1] = 1.
        im[..., 0] = 1. - im[..., 1]
        im = tf.constant(im)
        mean_height = get_mean_height(im, 1)
        mean_height = int(np.array(mean_height)[0])
        self.assertEqual(mean_height, 10)

    def test_quadratic_penalty_height(self):
        im = np.zeros([1, 40, 40, 3], np.float32)
        im[:, 10:20, :, 1] = 1.
        im[..., 0] = 1. - im[..., 1]
        im = tf.constant(im)
        a = QuadraticPenaltyHeight(1, 20.)(im)
        self.assertAlmostEqual(a, 0., "quadratic penalty should be zero here")
        b = QuadraticPenaltyHeight(1, 5.)(im)
        self.assertAlmostEqual(b, 25., "quadratic penalty should be close to 25 here")

    def test_penalty_height_batch(self):
        im = np.zeros([50, 40, 40, 3], np.float32)
        im[:, 10:20, :, 1] = 1.
        im[..., 0] = 1. - im[..., 1]
        im = tf.constant(im)
        a = QuadraticPenaltyHeight(1, 20.)(im)
        self.assertAlmostEqual(a, 0., "quadratic penalty should be zero here")
        b = QuadraticPenaltyHeight(1, 5.)(im)
        print(b)
        self.assertAlmostEqual(b, 25., "quadratic penalty should be close to 25 here")


if __name__ == "__main__":
    unittest.main()
