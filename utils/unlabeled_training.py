import tensorflow as tf

# TODO: almost everything
class UnlabeledTrain:
    def __init__(self, model, loss_fn, reg_fn, optimizer, weight_max=1., weight_epochs=50):
        self.model = model
        self.loss_fn = loss_fn
        self.reg_fn = reg_fn
        self.optimizer = optimizer
        self.weight = 0.

    def train_step(self, data):
        x1, y1, x2 = data
        # Run forward pass.
        with tf.GradientTape() as tape:
            y1_pred = self.model(x1, training=True)
            y2_pred = self.model(x2, training=True)
            loss1 = self.loss_fn(y1, y1_pred) + \
                self.reg_fn(y1_pred) * self.weight
            loss2 = self.reg_fn(y2_pred) * self.weight
            loss = loss1 + loss2
        # Run backwards pass.
        self.optimizer.minimize(
            loss, self.model.trainable_variables, tape=tape)
