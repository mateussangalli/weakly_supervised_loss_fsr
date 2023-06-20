import tensorflow as tf
from keras.callbacks import LambdaCallback
from keras.losses import Loss


def combined_loss(loss_fn, reg_fn, num_epochs, max_weight=5.):
    weight = tf.Variable((0.,), shape=(1,), trainable=False)

    # Create a callback to update the epoch value
    def weight_function(epoch, logs):
        w = min((epoch + 1) / num_epochs, 1.)
        w = max_weight * w
        weight.assign((w,))

    epoch_callback = LambdaCallback(on_epoch_begin=weight_function)

    def loss_function(y_true, y_pred):
        return loss_fn(y_true, y_pred) + weight * reg_fn(y_pred)

    return loss_function, epoch_callback


class CombinedLoss(Loss):
    def __init__(self, loss_fn, reg_fn, num_epochs, max_weight=5., **kwargs):
        super().__init__(**kwargs)
        self.weight = tf.Variable((0.,), shape=(1,), trainable=False)
        self.loss_fn = loss_fn
        self.reg_fn = reg_fn
        self.num_epochs = num_epochs
        self.max_weight = max_weight

        # Create a callback to update the epoch value
        def weight_function(epoch, logs):
            w = min((epoch + 1) / num_epochs, 1.)
            w = max_weight * w
            self.weight.assign((w,))

        self.callback = LambdaCallback(on_epoch_begin=weight_function)

    def __call__(self, y_true, y_pred, sample_weight=None):
        return self.loss_fn(y_true, y_pred) + self.weight * self.reg_fn(y_pred)
