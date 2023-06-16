import tensorflow as tf
from keras.callbacks import LambdaCallback


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
