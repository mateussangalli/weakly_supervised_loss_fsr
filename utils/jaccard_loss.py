import tensorflow as tf
from keras.metrics import MeanIoU


@tf.function
def jaccard2_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(y_true, (tf.shape(y_true)[0], -1))
    # print(y_true.shape)
    y_pred_f = tf.reshape(y_pred, (tf.shape(y_pred)[0], -1))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = (tf.reduce_sum(y_true_f * y_true_f) + tf.reduce_sum(y_pred_f * y_pred_f) -
             intersection)
    return (intersection + smooth) / (union + smooth)


@tf.function
def jaccard2_loss(y_true, y_pred, smooth=1.):
    try:
        return 1 - jaccard2_coef(y_true, y_pred, smooth)
    except Exception:
        print(y_true.shape)
        print(y_pred.shape)


class Jaccard2Loss(tf.keras.losses.Loss):
    def __init__(self, smooth=1., **kwargs):
        super().__init__(**kwargs)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        return 1. - jaccard2_coef(y_true, y_pred, self.smooth)


@tf.function
def jaccard_loss_mean(y_true, y_pred, smooth=1):
    """
      Mean Jaccard Loss per classes.
      Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
              = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

      The jaccard distance loss is useful for unbalanced datasets. This has been
      shifted so it converges on 0 and is smoothed to avoid exploding or
      disapearing gradient.

      Ref: https://en.wikipedia.org/wiki/Jaccard_index
      """
    intersection = y_true * y_pred
    union = y_true * y_true + y_pred * y_pred - intersection
    union = tf.reduce_sum(union, axis=[1, 2])
    intersection = tf.reduce_sum(intersection, axis=[1, 2])
    jac = (intersection + smooth) / (union + smooth)
    jac = tf.reduce_mean(jac, axis=1)
    return 1 - jac


class OneHotMeanIoU(MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.ensure_shape(y_true, [None, None, None, self.num_classes])
        y_pred = tf.ensure_shape(y_pred, [None, None, None, self.num_classes])
        return super().update_state(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1), sample_weight)
