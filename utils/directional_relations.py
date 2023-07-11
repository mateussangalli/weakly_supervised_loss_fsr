import numpy as np
import tensorflow as tf

EPS = 1e-4


def directional_kernel(direction, kernel_size):
    xx, yy = np.meshgrid(np.linspace(-1, 1, kernel_size[0]),
                         np.linspace(-1, 1, kernel_size[1]),
                         indexing='ij')
    if kernel_size[0] == 1:
        xx = np.zeros_like(xx)
    if kernel_size[1] == 1:
        yy = np.zeros_like(yy)
    directions_kernel = np.stack([xx, yy], -1)
    # directions_kernel = directions_kernel / (np.linalg.norm(directions_kernel, 2, 2, keepdims=True) + EPS)
    point_norm = np.linalg.norm(directions_kernel, 2, 2, keepdims=True)
    kernel = np.sum(-directions_kernel * direction, -1, keepdims=True)
    kernel = (kernel + EPS) / (point_norm + EPS)
    kernel = np.maximum(kernel, 0.)**2
    kernel = np.where(point_norm <= 1, kernel, 0.)
    return kernel


def fuzzy_product_dilation(image, kernel, kernel_size):
    """
    Args:
        image: 4-D tensor [batches, height, width, depth]
        kernel: 3-D tensor [kernel_height, kernel_width, depth]
    """
    shape = tf.shape(image)
    patches = tf.image.extract_patches(
        image,
        sizes=[1, kernel_size[0], kernel_size[1], 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding='SAME',
    )
    patches = tf.reshape(patches, [shape[0], shape[1], shape[2], -1, shape[3]])
    kernel = tf.reshape(kernel, [1, 1, 1, kernel_size[0] * kernel_size[1], -1])
    prod = patches * kernel
    return tf.reduce_max(prod, 3)


class DirectionalRelation:
    def __init__(self, distance, iterations, direction, dilation_type='product'):
        self.distance = distance
        self.iterations = iterations
        self.kernel_size = (2 * distance + 1, 2 * distance + 1)
        allowed_dilation_types = ('product', 'maxplus')
        if dilation_type not in allowed_dilation_types:
            raise ValueError(f"dilation_type must be one of: {allowed_dilation_types}")
        self.dilation_type = dilation_type

        kernel = directional_kernel(np.asarray(direction), self.kernel_size)
        self.kernel = tf.constant(kernel, tf.float32)

    def __call__(self, inputs):
        out = inputs
        for i in range(self.iterations):
            if self.dilation_type == 'product':
                tmp = fuzzy_product_dilation(out, self.kernel, self.kernel_size)
            else:
                tmp = tf.nn.dilation2d(out,
                                       self.kernel - 1.,
                                       (1, 1, 1, 1),
                                       'SAME',
                                       'NHWC',
                                       (1, 1, 1, 1))
            out = tf.maximum(tmp, out)
        return out - inputs


class RightOf(DirectionalRelation):
    def __init__(self, distance, iterations, dilation_type='product'):
        super().__init__(distance, iterations, (0, 1), dilation_type)


class LeftOf(DirectionalRelation):
    def __init__(self, distance, iterations, dilation_type='product'):
        super().__init__(distance, iterations, (0, -1), dilation_type)


class Above(DirectionalRelation):
    def __init__(self, distance, iterations, dilation_type='product'):
        super().__init__(distance, iterations, (-1, 0), dilation_type)


class Below(DirectionalRelation):
    def __init__(self, distance, iterations, dilation_type='product'):
        super().__init__(distance, iterations, (1, 0), dilation_type)


class StraightBelow:
    def __init__(self, distance, iterations):
        self.distance = distance
        self.iterations = iterations
        self.kernel_size = (2 * distance + 1, 1)

        kernel = np.zeros(self.kernel_size, np.float32)
        kernel[:distance+1, 0] = 1.
        kernel = kernel[:, :, np.newaxis]
        self.kernel = tf.constant(kernel, tf.float32)

    def __call__(self, inputs):
        out = inputs
        for i in range(self.iterations):
            out = tf.nn.dilation2d(out,
                                   self.kernel - 1.,
                                   (1, 1, 1, 1),
                                   'SAME',
                                   'NHWC',
                                   (1, 1, 1, 1))
        return out - inputs

class StraightAbove:
    def __init__(self, distance, iterations):
        self.distance = distance
        self.iterations = iterations
        self.kernel_size = (2 * distance + 1, 1)

        kernel = np.zeros(self.kernel_size, np.float32)
        kernel[:distance+1, 0] = 1.
        kernel = kernel[::-1, :]
        kernel = kernel[:, :, np.newaxis]
        self.kernel = tf.constant(kernel, tf.float32)

    def __call__(self, inputs):
        out = inputs
        for i in range(self.iterations):
            out = tf.nn.dilation2d(out,
                                   self.kernel - 1.,
                                   (1, 1, 1, 1),
                                   'SAME',
                                   'NHWC',
                                   (1, 1, 1, 1))
        return out - inputs

def product_tnorm(a, b):
    return a * b


def lukasiewicz_tnorm(a, b):
    return tf.maximum(a + b - 1., 0.)

class PRPDirectionalPenalty(tf.keras.regularizers.Regularizer):
    def __init__(self,
                 distance,
                 iterations,
                 sc_class=0,
                 le_class=2,
                 bg_class=1,
                 dilation_type='maxplus',
                 tnorm='product',
                 reduction_type='mean',
                 return_map=False,
                 sym_bg=False,
                 **kwargs):
        self.distance = distance
        self.iterations = iterations

        self.sc_class = sc_class
        self.le_class = le_class
        self.bg_class = bg_class

        self.dilation_type = dilation_type
        if dilation_type == 'flat_line':
            self.above = StraightAbove(distance, iterations)
            self.below = StraightBelow(distance, iterations)
        else:
            self.above = Above(distance, iterations, dilation_type=dilation_type)
            self.below = Below(distance, iterations, dilation_type=dilation_type)

        if isinstance(tnorm, str):
            if tnorm == 'product':
                self.tnorm = product_tnorm
            elif tnorm == 'lukasiewicz':
                self.tnorm = product_tnorm
            else:
                raise ValueError('tnorm not recognized')

        self.return_map = return_map

        self.reduction_type = reduction_type

    def regularization_term(self, inputs):
        prob_bg = tf.expand_dims(inputs[..., self.bg_class], -1)
        prob_sc = tf.expand_dims(inputs[..., self.sc_class], -1)
        prob_le = tf.expand_dims(inputs[..., self.le_class], -1)

        above_sc = self.above(prob_sc)
        below_sc = self.below(prob_sc)
        below_le = self.below(prob_le)
        above_le = self.above(prob_le)
        above_bg = self.above(prob_bg)
        below_bg = self.below(prob_bg)

        sc_below_le = self.tnorm(prob_sc, below_le)
        le_above_sc = self.tnorm(prob_le, above_sc)

        bg_below_sc = self.tnorm(prob_bg, below_sc)
        bg_above_le = self.tnorm(prob_bg, above_le)

        map = le_above_sc + sc_below_le + bg_above_le + bg_below_sc
        if self.sym_bg:
            sc_above_bg = self.tnorm(prob_sc, above_bg)
            le_below_bg = self.tnorm(prob_le, below_bg)
            map += sc_above_bg + le_below_bg

        if self.return_map:
            return map

        if self.reduction_type == 'squared_mean':
            penalty = tf.reduce_mean(map**2)
        elif self.reduction_type == 'mean':
            penalty = tf.reduce_mean(map)
        else:
            raise ValueError("unrecognized reduction argument")

        return penalty

    def __call__(self, pred):
        return self.regularization_term(pred)

    def get_config(self):
        config = super().get_config()
        config['distance'] = self.distance
        config['iterations'] = self.iterations
        config['sc_class'] = self.sc_class
        config['le_class'] = self.le_class
        config['bg_class'] = self.bg_class
        config['dilation_type'] = self.dilation_type
        config['tnorm'] = self.tnorm
        config['return_map'] = self.return_map


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from skimage.io import imsave

    r = 30
    size = r * 2 + 1
    sqr_side = 10
    sqr_r = sqr_side // 2
    im = np.zeros([size, size], np.float32)
    im[r-sqr_r:r+sqr_r, r-sqr_r:r+sqr_r] = 1.
    im = im[np.newaxis, :, :, np.newaxis]
    # layer = Below(10, 2, dilation_type='maxplus')
    # layer = StraightAbove(10, 40)
    layer = LeftOf(30, 1, dilation_type='maxplus')
    out = layer(im) + im
    out = out[0, :, :, 0]
    plt.subplot(121)
    plt.imshow(out)
    plt.subplot(122)
    plt.imshow(layer.kernel[::-1, ::-1, 0])
    plt.show()
    #
    # im = np.array(im)[0, :, :, 0]
    # im = (255. * im).astype(np.uint8)
    # out = np.array(out)
    # out = (255. * out).astype(np.uint8)
    # kernel = np.array(layer.kernel[::-1, ::-1, 0])
    # kernel = (255. * kernel).astype(np.uint8)
    #
    # imsave('square.png', im)
    # imsave('left_of_square.png', out)
    # imsave('kernel_left_of.png', kernel)
