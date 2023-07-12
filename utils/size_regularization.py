import tensorflow as tf

EPS = 1e-5

def get_mean_height(pred, class_num=0):
    heights = tf.reduce_sum(pred[:, :, :, class_num], 1)
    mean = tf.reduce_mean(heights, 1)
    return mean

def get_height(pred, class_num=0):
    return tf.reduce_sum(pred[:, :, :, class_num], 1)



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


def log_barrier(z, t):
    return tf.where(
        z < -(tf.math.reciprocal(tf.pow(t, 2))),
        -tf.math.reciprocal(t) * tf.math.log(-z),
        t * z - tf.math.reciprocal(t) * (tf.math.log(tf.math.reciprocal(tf.pow(t, 2))) - 1.)
    )


def log_barrier_both_sides(z, t, vmin, vmax):
    loss_right = log_barrier(z - vmax, t)
    loss_left = log_barrier(vmin - z, t)
    return loss_right + loss_left


class LogBarrierHeight(tf.keras.regularizers.Regularizer):
    def __init__(self, class_num, vmin, vmax, t_schedule, t=1., reduce=True, **kwargs):
        super().__init__(**kwargs)
        self.class_num = class_num
        self.vmin = vmin
        self.vmax = vmax
        self.t = t
        self.t_schedule = t_schedule
        self.reduce = reduce

    def update(self, step):
        step = tf.cast(step, tf.float32)
        self.t = self.t_schedule(step)

    def get_config(self):
        config = super().get_config()
        config['class_num'] = self.class_num
        config['vmax'] = self.vmax
        config['vmin'] = self.vmin
        config['t'] = self.t
        config['reduce'] = self.reduce

    def __call__(self, proba):
        height = get_height(proba, class_num=self.class_num)
        loss_value = log_barrier_both_sides(height, self.t, self.vmin, self.vmax)
        if self.reduce:
            return tf.reduce_mean(loss_value)
        return loss_value


class LogBarrierHeightRatio(tf.keras.regularizers.Regularizer):
    def __init__(self, class_enum, class_denom, vmin, vmax, t_schedule, t=1., reduce=True, **kwargs):
        super().__init__(**kwargs)
        self.class_enum = class_enum
        self.class_denom = class_denom
        self.vmin = vmin
        self.vmax = vmax
        self.t = t
        self.t_schedule = t_schedule
        self.reduce = reduce

    def update(self, step):
        step = tf.cast(step, tf.float32)
        self.t = self.t_schedule(step)

    def get_config(self):
        config = super().get_config()
        config['class_enum'] = self.class_enum
        config['class_denom'] = self.class_denom
        config['vmax'] = self.vmax
        config['vmin'] = self.vmin
        config['t'] = self.t
        config['reduce'] = self.reduce

    def __call__(self, proba):
        height_enum = get_height(proba, class_num=self.class_enum)
        height_denom = get_height(proba, class_num=self.class_denom)
        ratio = (height_enum + EPS) / (height_denom + EPS)
        loss_value = log_barrier_both_sides(ratio, self.t, self.vmin, self.vmax)
        if self.reduce:
            return tf.reduce_mean(loss_value)
        return loss_value


if __name__ == "__main__":
    # visualize log barrier
    import numpy as np
    import matplotlib.pyplot as plt
    N = 300
    im = np.zeros([1, N, N, 3], np.float32)
    for i in range(N):
        im[:, :i, i, :] = 1.
    plt.imshow(im[0, ...])
    plt.show()
    vmin = 30.
    vmax = float(N - 30)

    t_values = [1., 3., 10., 20.]
    for t in t_values:
        loss = LogBarrierHeight(0, vmin, vmax, t, reduce=False)(im)
        loss = np.array(loss)[0, :]
        plt.plot(loss, label=f'{t=}')
    plt.legend()
    plt.show()
