import tensorflow as tf


def get_mean_height(pred, class_num=0):
    heights = tf.reduce_sum(pred[:, :, :, class_num], 1)
    mean = tf.reduce_mean(heights, 1)
    return mean


class QuadraticPenaltyHeight(tf.keras.regularizers.Regularizer):
    def __init__(self, class_num, vmax, **kwargs):
        super().__init__(**kwargs)
        self.class_num = class_num
        self.vmax = vmax

    def get_config(self):
        config = super().get_config()
        config['class_num'] = self.class_num
        config['vmax'] = self.vmax

    def __call__(self, proba):
        mean_height = get_mean_height(proba, class_num=self.class_num)
        penalty = mean_height - self.vmax
        penalty = tf.where(penalty > 0., penalty, 0.)
        penalty = tf.pow(penalty, 2.)
        return tf.reduce_mean(penalty)
