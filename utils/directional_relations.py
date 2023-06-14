import numpy as np
import tensorflow as tf
from keras.layers import Layer
from keras.losses import Loss
from keras import backend
from keras.utils import control_flow_util

EPS = 1e-5


def directional_kernel(direction, kernel_size):
    xx, yy = np.meshgrid(np.linspace(-1, 1, kernel_size[0]),
                         np.linspace(-1, 1, kernel_size[1]),
                         indexing='ij')
    if kernel_size[0] == 1:
        xx = np.zeros_like(xx)
    if kernel_size[1] == 1:
        yy = np.zeros_like(yy)
    directions_kernel = np.stack([xx, yy], -1)
    directions_kernel = directions_kernel / (np.linalg.norm(directions_kernel, 2, 2, keepdims=True) + EPS)
    kernel = np.sum(-directions_kernel * direction, -1, keepdims=True)
    kernel = np.maximum(kernel, 0.) - 1.
    return kernel


class DirectionalRelation:
    def __init__(self, distance, spread, iterations, direction):
        self.distance = distance
        self.spread = spread
        self.iterations = iterations
        self.kernel_size = (2 * spread + 1, 2 * distance + 1)

        kernel = directional_kernel(np.asarray(direction), self.kernel_size)
        self.kernel = tf.constant(kernel, tf.float32)

    def __call__(self, inputs):
        out = inputs
        for i in range(self.iterations):
            tmp = tf.nn.dilation2d(out,
                                   self.kernel,
                                   (1, 1, 1, 1),
                                   'SAME',
                                   'NHWC',
                                   (1, 1, 1, 1))
            out = tf.maximum(tmp, out)
        return out - inputs


class RightOf(DirectionalRelation):
    def __init__(self, distance, spread, iterations):
        super().__init__(distance, spread, iterations, (0, 1))


class LeftOf(DirectionalRelation):
    def __init__(self, distance, spread, iterations):
        super().__init__(distance, spread, iterations, (0, -1))


class Above(DirectionalRelation):
    def __init__(self, distance, spread, iterations):
        super().__init__(distance, spread, iterations, (-1, 0))


class Below(DirectionalRelation):
    def __init__(self, distance, spread, iterations):
        super().__init__(distance, spread, iterations, (1, 0))


class PRPDirectionalPenalty(Loss):
    def __init__(self,
                 distance,
                 spread,
                 iterations,
                 sc_class=0,
                 le_class=2,
                 bg_class=1,
                 **kwargs):
        self.distance = distance
        self.spread = spread
        self.iterations = iterations

        self.sc_class = sc_class
        self.le_class = le_class
        self.bg_class = bg_class

        self.above = Above(distance, spread, iterations)
        self.below = Below(distance, spread, iterations)

    def regularization_term(self, inputs):
        prob_bg = tf.expand_dims(inputs[..., self.bg_class], -1)
        prob_sc = tf.expand_dims(inputs[..., self.sc_class], -1)
        prob_le = tf.expand_dims(inputs[..., self.le_class], -1)

        above_bg = self.above(prob_bg)
        below_bg = self.below(prob_bg)
        above_sc = self.above(prob_sc)

        sc_above_bg = prob_sc * above_bg
        le_below_bg = prob_le * below_bg
        le_above_sc = prob_le * above_sc

        penalty = tf.reduce_mean(sc_above_bg) + \
            tf.reduce_mean(le_below_bg) + \
            tf.reduce_mean(le_above_sc)

        return penalty

    def __call__(self, pred):
        return self.regularization_term(pred)

    def get_config(self):
        config = super().get_config()
        config['distance'] = self.distance
        config['spread'] = self.spread
        config['iterations'] = self.iterations


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    im = np.zeros([31, 31], np.float32)
    im[15, 15] = 1.
    im = im[np.newaxis, :, :, np.newaxis]
    layer = Below(5, 5, 6)
    out = layer(im)[0, :, :, 0]
    plt.imshow(out)
    plt.show()
    plt.imshow(layer.kernel[:, :, 0])
    plt.show()
