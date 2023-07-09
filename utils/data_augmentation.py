import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_io as tfio
from skimage.color import rgb2lab
from tqdm import tqdm

# mask_debug = np.ones([512, 512, 3], np.float32)
# mask_debug[:100, :100, :] = 0
# mask_debug = tf.constant(mask_debug)


class AugmentSometimes:
    """
    Given a data augmentation function, define a new augmentation that applies it to data with a certain probability
    each time.
    -------
    Attributes:
        prob: Probability of applying given transformation.
        func: Transformation that will be applied. Inputs and outputs should be of the same type.
    """

    def __init__(self, prob, func):
        self.prob = prob
        self.func = func

    def __call__(self, image, label):
        r = tf.random.uniform((), 0, 1, dtype=tf.float32) < self.prob
        return tf.cond(r, lambda: self.func(image, label), lambda:
                       (image, label))


@tf.function
def horizontal_flip(image, label):
    image_out = tf.image.flip_left_right(image)
    label_out = tf.image.flip_left_right(label)
    return image_out, label_out


@tf.function
def random_horizontal_flip(image, label):
    r = tf.random.uniform((), 0, 2, dtype=tf.int32)
    return tf.cond(r == 1, lambda: horizontal_flip(image, label), lambda:
                   (image, label))


class ColorTransferForeground:
    """
    Based on Yang XIAO code.
    Perform a random color transfer by shifting the mean and
        variance of the foreground pixels to a new mean and variance.
    Attributes
    ----------
        background_label:
            Integer corresponding to the "background" class in the ground truths.
            Pixels with this label are ignored during mean and variance computations.
        means:
            tf.Tensor of shape [N, 3] specifying the channel-wise means that can be used to transfer the color of the image.
            Can be None if you want to fit the means using the fit_new_dataset method.
            Has to be in the CIELab color space in the range [0,100] x [-128, 127] x [-128, 127].
        variances:
            Similar to means but for channel-wise variances.
        use_variance:
            Boolean. If True, the __call__ method will use the variance to normalize the colors of the image being
            augmented.

    Methods
    -------
        _compute_sum_and_weight:
            computes the sum of foreground input pixels and the number of foreground pixels
        fit_new_dataset:
            Computes the mean and variance of a dataset of images.
            Inputs should be RGB and means and variances are stored in CIELab.
        color_transfer_lab:
            Perform the color transfer for an image / ground truth pair in the CIELab space.
        color_transfer_rgb:
            Perform the color transfer for an image / ground truth pair in the RGB space.
        __call__:
            Alias for color_transfer_rgb
    """

    def __init__(self,
                 background_label=1,
                 means=None,
                 variances=None,
                 use_variance=False,
                 verbose=1,
                 color_space='rgb'):
        self.background_label = background_label
        self.use_variance = use_variance
        self.means = means
        self.variances = variances
        self.verbose = verbose
        self.color_space = color_space

        if means is not None:
            self.num_means = self.means.shape[0]
        else:
            self.num_means = 0

    def _mean_and_variance(self, image, label):
        mask = label != self.background_label
        mean = np.mean(image[mask, :].reshape(-1, 3), 0)
        variance = np.var(image[mask, :].reshape(-1, 3), 0)
        return mean, variance

    def fit_new_dataset(self, images, labels):
        means = list()
        variances = list()
        if self.verbose == 1:
            iterator = tqdm(zip(images, labels), total=len(images))
            print(
                'Computing means and variances for new dataset for color augmentation...'
            )
        else:
            iterator = zip(images, labels)
        for image, label in iterator:
            if self.color_space == 'lab':
                image = rgb2lab(image)
            mean, variance = self._mean_and_variance(np.asarray(image),
                                                     np.asarray(label))
            means.append(mean)
            variances.append(variance)
        num_means = len(means)

        means = np.stack(means, 0)
        variances = np.stack(variances, 0)
        means = tf.constant(means, dtype=tf.float32)
        variances = tf.constant(variances, dtype=tf.float32)
        if self.means is None:
            self.means = means
            self.variances = variances
        else:
            self.means = tf.concat([self.means, means], 0)
            self.variances = tf.concat([self.variances, variances], 0)
        self.num_means += num_means

    @tf.function
    def color_transfer_lab(self, im, gt):
        im = tfio.experimental.color.rgb_to_lab(im)

        im_new, gt = self.color_transfer(im, gt)

        im_new = tfio.experimental.color.lab_to_rgb(im_new)
        return im_new, gt

    @tf.function
    def color_transfer(self, im, gt):
        """
        Based on Yang XIAO code
        Perform a random color transfer by shifting the mean of the foreground pixels to a new mean
        Assumes image and means are in the CIELab color space
        Args:
            im: 3-dimensional tensor, last dimension is n_channels
            gt: 2-dimensional integer tensor

        Returns:
            im_new: tensor with the same shape and type as im

        """
        indice = tf.cast(tf.random.uniform((), 0, self.means.shape[0]),
                         tf.int32)
        # indice = int(tf.minimum(indice, self.num_means))
        mean_new = self.means[indice, :]
        mean_new = tf.reshape(mean_new, [1, 1, 3])

        mask = tf.expand_dims(gt != self.background_label, 2)
        mask_float = tf.where(mask, 1., 0.)
        mask_size = tf.reduce_sum(mask_float)
        mean_ori = tf.reduce_sum(mask_float * im, [0, 1],
                                 keepdims=True) / mask_size
        im_new = im - mean_ori * mask_float
        if self.use_variance:
            variance_new = self.variances[indice, :]
            variance_new = tf.reshape(variance_new, [1, 1, 3])
            variance_ori = tf.math.reduce_variance(im, [0, 1], keepdims=True)
            im_new = mask_float * im_new / (tf.sqrt(variance_ori) +
                                            1e-5) + (1. - mask_float) * im_new
            im_new = mask_float * im_new * tf.sqrt(variance_new) + (
                1. - mask_float) * im_new
        im_new = im_new + mean_new * mask_float
        return im_new, gt

    def __call__(self, image, label):
        if self.color_space == 'lab':
            return self.color_transfer_lab(image, label)
        return self.color_transfer(image, label)


def smoothstep(value, tmin, tmax):
    return tf.minimum(
        tf.maximum(
            (value - tmin) / (tmax - tmin),
            0.),
        1.)


class ColorTransfer:
    def __init__(self,
                 means,
                 probability=.5,
                 white_threshold_min=.96,
                 white_threshold_max=.99,
                 black_threshold=0.01):
        self.means = means
        self.probability = probability
        self.white_threshold_min = white_threshold_min
        self.white_threshold_max = white_threshold_max
        self.black_threshold = black_threshold

    def apply_augmentation(self, image: tf.Tensor):
        # mask out the white pixels an subtract the old mean
        # mask = tf.cast(tf.reduce_min(image, keepdims=True) < self.white_threshold, image.dtype)
        mask = smoothstep(tf.reduce_min(image, -1, keepdims=True),
                          self.white_threshold_min,
                          self.white_threshold_max)
        mask = 1. - mask
        mask = tf.where(tf.reduce_max(image, -1, keepdims=True) > self.black_threshold,
                        mask, 0.)
        # return tf.tile(mask, [1, 1, 3])
        im_mean = tf.reduce_sum(
            mask * image, [0, 1], keepdims=True) / (tf.reduce_sum(mask) + 1e-10)

        # select one of the mean tensors
        indice = tf.cast(tf.random.uniform(
            (), 0, self.means.shape[0]), tf.int32)
        new_mean = self.means[indice, :]
        new_mean = tf.reshape(new_mean, [1, 1, 3])
        mean_change = new_mean - im_mean

        return mask * (image + mean_change) + (1. - mask) * image

    def __call__(self, image: tf.Tensor):
        r = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)
        return tf.cond(r < self.probability,
                       lambda: self.apply_augmentation(image),
                       lambda: image)


@tf.function
def resize_inputs(im, gt, size):
    """
    Resizes an image and ground truth pair to squares of specified size
    Args:
        im: 3D tf.Tensor
        gt: 2D tf.Tensor with the dimensions matching the first two dimensions of im
        size: Desired size. Integer.

    Returns:
       im_resized: tf.Tensor with shape [size, size, im.shape[2]]
       gt_resized: tf.Tensor with shape [size, size]
    """
    im_resized = tf.image.resize(im, [size, size], method='bilinear')
    gt_resized = tf.image.resize(tf.expand_dims(gt, 2), [size, size],
                                 method='nearest')[:, :, 0]
    return im_resized, gt_resized


@tf.function
def log_uniform(shape, min_value, max_value, dtype=tf.float32):
    out = tf.random.uniform(shape,
                            tf.math.log(min_value),
                            tf.math.log(max_value),
                            dtype=dtype)
    out = tf.math.exp(out)
    return out


def resize_crop_pad_aug(scale_range, output_size):
    output_size = tf.constant(output_size, dtype=tf.int32)

    def resize_crop_pad(image, label):
        resize_factor = log_uniform([], scale_range[0], scale_range[1])

        # Compute the new size after resizing
        new_size = tf.cast(
            tf.round(tf.multiply(tf.cast(output_size, tf.float32),
                                 resize_factor)),
            tf.int32)

        # Resize the image and label
        image = tf.image.resize(image, new_size, method=tf.image.ResizeMethod.BILINEAR)
        label = tf.image.resize(label, new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Pad the image and label if necessary
        pad_h = tf.maximum(0, output_size[0] - new_size[0])
        pad_w = tf.maximum(0, output_size[1] - new_size[1])
        padded_h = tf.maximum(output_size[0], new_size[0])
        padded_w = tf.maximum(output_size[1], new_size[1])
        image = tf.image.pad_to_bounding_box(image, pad_h // 2, pad_w // 2, padded_h, padded_w)
        label = tf.image.pad_to_bounding_box(label, pad_h // 2, pad_w // 2, padded_h, padded_w)

        # Randomly select crop position
        crop_x = tf.random.uniform([], maxval=tf.maximum(1, new_size[1] - output_size[1]), dtype=tf.int32)
        crop_y = tf.random.uniform([], maxval=tf.maximum(1, new_size[0] - output_size[0]), dtype=tf.int32)

        # Crop the image and label
        image = tf.image.crop_to_bounding_box(image, crop_y, crop_x, output_size[0], output_size[1])
        label = tf.image.crop_to_bounding_box(label, crop_y, crop_x, output_size[0], output_size[1])


        return image, label
    return resize_crop_pad


class Cropper:

    def __init__(self,
                 crop_size,
                 crops_per_image,
                 max_scale=1.,
                 background_label=1):
        self.crop_size = crop_size
        self.max_scale = max_scale
        self.min_scale = 1 / max_scale
        self.crops_per_image = crops_per_image
        self.bounding_box_border = crop_size // 2
        self.background_label = background_label

    def foreground_bounding_box(self, label):
        shape = tf.shape(label)
        foreground = 1. - label[:, :, self.background_label]
        foreground_horizontal = tf.reduce_max(foreground, 0)
        foreground_vertical = tf.reduce_max(foreground, 1)
        top = tf.argmax(foreground_vertical, output_type=tf.int32)
        bottom = tf.argmax(foreground_vertical[::-1], output_type=tf.int32)
        left = tf.argmax(foreground_horizontal, output_type=tf.int32)
        right = tf.argmax(foreground_horizontal[::-1], output_type=tf.int32)
        top = tf.maximum(top - self.bounding_box_border, 0)
        left = tf.maximum(left - self.bounding_box_border, 0)
        bottom = tf.maximum(bottom - self.bounding_box_border, 0)
        right = tf.maximum(right - self.bounding_box_border, 0)
        bottom = shape[0] - bottom
        right = shape[1] - right
        return top, bottom, left, right

    def __call__(self, image, label):
        shape = tf.shape(image)
        t, b, l, r = self.foreground_bounding_box(label)
        t = tf.cast(t, tf.float32) / tf.cast(shape[0], tf.float32)
        b = tf.cast(b, tf.float32) / tf.cast(shape[0], tf.float32)
        l = tf.cast(l, tf.float32) / tf.cast(shape[1], tf.float32)
        r = tf.cast(r, tf.float32) / tf.cast(shape[1], tf.float32)

        scale = log_uniform((self.crops_per_image, ),
                            self.min_scale,
                            self.max_scale,
                            dtype=tf.float32)
        crop_size_x = self.crop_size / tf.cast(shape[1], tf.float32)
        crop_size_y = self.crop_size / tf.cast(shape[0], tf.float32)
        delta_x = scale * crop_size_x
        delta_y = scale * crop_size_y
        b = b - delta_y
        r = r - delta_x

        x1 = tf.random.uniform((self.crops_per_image, ), 0., 1., tf.float32)
        x1 = l + x1 * (r - l)
        x2 = x1 + delta_x
        y1 = tf.random.uniform((self.crops_per_image, ), 0., 1., tf.float32)
        y1 = t + y1 * (b - t)
        y2 = y1 + delta_y
        # y1 = t * tf.ones((self.crops_per_image, ), tf.float32)
        # y2 = b * tf.ones((self.crops_per_image, ), tf.float32)
        # x1 = l * tf.ones((self.crops_per_image, ), tf.float32)
        # x2 = r * tf.ones((self.crops_per_image, ), tf.float32)
        boxes = tf.stack([y1, x1, y2, x2], 1)
        images_cropped = tf.image.crop_and_resize(
            tf.expand_dims(image, 0),
            boxes,
            tf.zeros((self.crops_per_image, ),
                     dtype=tf.int32), [self.crop_size, self.crop_size],
            method='bilinear')
        labels_cropped = tf.image.crop_and_resize(
            tf.expand_dims(label, 0),
            boxes,
            tf.zeros((self.crops_per_image, ),
                     dtype=tf.int32), [self.crop_size, self.crop_size],
            method='nearest')
        return images_cropped, labels_cropped


class RandomRotation:
    def __init__(self, angle):
        self.max_angle = angle
        self.min_angle = -angle

    def __call__(self, image, gt):
        angle = tf.random.uniform((1,), self.min_angle, self.max_angle)
        im_out = tfa.image.rotate(image,
                                  angle,
                                  interpolation='bilinear',
                                  fill_mode='constant')
        gt_out = tfa.image.rotate(gt,
                                  angle,
                                  interpolation='nearest',
                                  fill_mode='constant',
                                  fill_value=1)
        return im_out, gt_out


def crop_to_multiple_of_preproc(image, label, k=32):
    shape = tf.shape(image)
    crop1 = shape[0] % k
    crop2 = shape[1] % k
    image_out = image[:shape[0] - crop1, :shape[1] - crop2, :]
    label_out = label[:shape[0] - crop1, :shape[1] - crop2, :]
    return image_out, label_out


class ColorJitter:
    def __init__(self, hue=0.1, saturation=0.1, brightness=0.1):
        self.hue = hue
        self.saturation = saturation
        self.brightness = brightness
        self.scaling = tf.constant([hue, saturation, brightness], tf.float32)
        self.scaling = tf.reshape(self.scaling, (1, 1, 3))

    def __call__(self, im):
        im = tf.image.rgb_to_hsv(im)
        offset = tf.random.uniform((1, 1, 3), -1., 1.) * self.scaling
        im = im + offset
        return tf.image.hsv_to_rgb(im)
