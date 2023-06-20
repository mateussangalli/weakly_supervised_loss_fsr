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


def combined_loss_multiple(loss_fn, reg_fns, num_epochs=[50,50], max_weight=[5.,5.]):
  W=[]
  for i in range(len(reg_fns)):
    W.append(tf.Variable((0.,), shape=(1,), trainable=False))
  # Create a callback to update the epoch value
  def weight_function(epoch, logs):
    for i in range(len(max_weight)):
      w = min((epoch + 1) / num_epochs[i], 1.)
      w = max_weight[i] * w
      W[i].assign((w,))

  epoch_callback=[]
  for i in range(len(reg_fns)):
    epoch_callback.append(LambdaCallback(on_epoch_begin=weight_function))


  def loss_function(y_true, y_pred):
    loss=loss_fn(y_true, y_pred)
    for i,reg_fn in enumerate(reg_fns):
      loss=loss + W[i]*reg_fn(y_pred)
    return loss
    
  return loss_function, epoch_callback
