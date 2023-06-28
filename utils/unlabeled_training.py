import tensorflow as tf


class SemiSupModel(tf.keras.Model):
    def __init__(self, *args, alpha=0., **kwargs):
        super(SemiSupModel, self).__init__(*args, **kwargs)
        if isinstance(alpha, float):
            self.alpha_schedule = lambda _: alpha
        else:
            self.alpha_schedule = alpha


    def compile(self, optimizer, loss_1, loss_2, **kwargs):
        # WARN: if you load this model you probably have to call this function again
        # WARN: also when you cal evaluate it will not use any of these losses
        super(SemiSupModel, self).compile(**kwargs)
        self.optimizer = tf.keras.optimizers.get(optimizer)
        self.loss_1 = tf.keras.losses.get(loss_1)
        self.loss_2 = tf.keras.losses.get(loss_2)

    def train_step(self, data):
        (x_l, y), x_u = data

        with tf.GradientTape() as tape:
            # Forward pass on labeled data
            y_pred = self(x_l, training=True)
            loss_1_value = self.loss_1(y, y_pred)

            # Forward pass on unlabeled data
            y_u_pred = self(x_u, training=True)
            loss_2_value = self.loss_2(y_u_pred)

            # Compute total loss with weight alpha(epoch)
            alpha = self.alpha_schedule(self.optimizer.iterations)
            total_loss = loss_1_value + alpha * loss_2_value

        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict of metrics for monitoring
        results = {m.name: m.result() for m in self.metrics}
        results['loss_1'] = loss_1_value
        results['loss_2'] = loss_2_value
        results['total_loss'] = total_loss

        return results
