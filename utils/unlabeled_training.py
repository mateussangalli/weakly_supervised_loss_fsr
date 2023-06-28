import tensorflow as tf


class SemiSupModel(tf.keras.Model):
    def __init__(self, *args, alpha_schedule, **kwargs):
        super(SemiSupModel, self).__init__(*args, **kwargs)
        self.alpha_schedule = alpha_schedule

    def compile(self, optimizer, loss_1, loss_2, **kwargs):
        super(SemiSupModel, self).compile(**kwargs)
        self.optimizer = optimizer
        self.loss_1 = loss_1
        self.loss_2 = loss_2

    def train_step(self, data):
        x_l, y, x_u = data

        with tf.GradientTape() as tape:
            # Forward pass on labeled data
            y_pred = self(x_l, training=True)
            loss_1_value = self.loss_1(y, y_pred)

            # Forward pass on unlabeled data
            y_u_pred = self(x_u, training=True)
            loss_2_value = self.loss_2(x_u, y_u_pred)

            # Compute total loss with weight alpha(epoch)
            alpha = self.alpha_schedule(
                tf.keras.backend.get_value(self.optimizer.iterations))
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
