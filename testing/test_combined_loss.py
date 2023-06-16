import numpy as np
import keras
import tensorflow as tf

from utils.combined_loss import combined_loss


if __name__ == "__main__":
    x_train = np.random.rand(5, 10, 10, 3)
    y_train = np.eye(3)[np.random.randint(0, 3, [5, 10, 10])]
    # Number of epochs you plan to train
    num_epochs = 10


    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(3, 3))

    def loss1(y, y_pred):
        return (tf.zeros_like(y_pred)+1e-5) * y_pred

    def loss2(y_pred):
        return tf.ones_like(y_pred) * y_pred

    loss_fn, callback = combined_loss(loss1, loss2, num_epochs)

    # Example usage during model.fit
    model.compile(optimizer="adam", loss=loss_fn)

    model.fit(x_train, y_train, epochs=num_epochs, callbacks=[callback])
