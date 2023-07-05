import tensorflow as tf
import collections


class SemiSupModel(tf.keras.Model):
    """
    Subclass of the tf.keras.Model class that incorporates unlabeled data into the training loop.
    The loss function of the unlabeled data takes a single argument (y_pred).
        Compiling the loss function and optimizer are not supported.
        sample_weights are not supported
    """
    def __init__(self, *args, alpha=0., num_unlabeled_losses=1, **kwargs):
        super(SemiSupModel, self).__init__(*args, **kwargs)
        if isinstance(alpha, float):
            self.alpha_schedule = [lambda _: alpha] * num_unlabeled_losses
        else:
            if isinstance(alpha, collections.Iterable):
                self.alpha_schedule = alpha
            else:
                self.alpha_schedule = [alpha] * num_unlabeled_losses

    def compile(self, optimizer, loss_labeled, loss_unlabeled, **kwargs):
        # WARN: if you load this model you probably have to call this function again
        # WARN: also when you cal evaluate it will not use any of these losses
        super(SemiSupModel, self).compile(**kwargs)
        self.optimizer = tf.keras.optimizers.get(optimizer)
        self.loss_labeled = tf.keras.losses.get(loss_labeled)
        if isinstance(loss_unlabeled, collections.Iterable):
            self.loss_unlabeled = [tf.keras.regularizers.get(loss_fn) for loss_fn in loss_unlabeled]
        else:
            self.loss_unlabeled = [tf.keras.regularizers.get(loss_unlabeled)]

    def train_step(self, data):
        (x_l, y), x_u = data

        with tf.GradientTape() as tape:
            # Forward pass on labeled data
            y_pred = self(x_l, training=True)
            loss_lab_value = self.loss_labeled(y, y_pred)

            # Forward pass on unlabeled data
            y_u_pred = self(x_u, training=True)
            loss_unlab_value = 0.
            for alpha_schd, loss_fn in zip(self.alpha_schedule, self.loss_unlabeled):
                alpha = alpha_schd(self.optimizer.iterations)
                loss_unlab_value += alpha * loss_fn(y_u_pred)

            # Compute total loss with weight alpha(epoch)
            total_loss = loss_lab_value + loss_unlab_value

        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict of metrics for monitoring
        results = {m.name: m.result() for m in self.metrics}
        results['loss_labeled'] = loss_lab_value
        results['loss_unlabeled'] = loss_unlab_value
        results['total_loss'] = total_loss

        return results


class SemiSupModelPseudo(tf.keras.Model):
    """
    Subclass of the tf.keras.Model class that incorporates unlabeled data into the training loop.
    Specically made to be used with pseudo-labeling approaches.
    The loss function of the unlabeled data takes a two arguments (y_u, y_u_pred).
        Compiling the loss function and optimizer are not supported.
        sample_weights are not supported
    """
    def __init__(self, *args, alpha=0., num_unlabeled_losses=1, **kwargs):
        super(SemiSupModelPseudo, self).__init__(*args, **kwargs)
        if isinstance(alpha, float):
            self.alpha_schedule = [lambda _: alpha] * num_unlabeled_losses
        else:
            if isinstance(alpha, collections.Iterable):
                self.alpha_schedule = alpha
            else:
                self.alpha_schedule = [alpha] * num_unlabeled_losses

    def compile(self, optimizer, loss_labeled, loss_unlabeled, **kwargs):
        # WARN: if you load this model you probably have to call this function again
        # WARN: also when you cal evaluate it will not use any of these losses
        super(SemiSupModelPseudo, self).compile(**kwargs)
        self.optimizer = tf.keras.optimizers.get(optimizer)
        self.loss_labeled = tf.keras.losses.get(loss_labeled)
        if isinstance(loss_unlabeled, collections.Iterable):
            self.loss_unlabeled = [tf.keras.regularizers.get(loss_fn) for loss_fn in loss_unlabeled]
        else:
            self.loss_unlabeled = [tf.keras.regularizers.get(loss_unlabeled)]

    def train_step(self, data):
        (x_l, y), (x_u, y_u) = data

        with tf.GradientTape() as tape:
            # Forward pass on labeled data
            y_pred = self(x_l, training=True)
            loss_lab_value = self.loss_labeled(y, y_pred)

            # Forward pass on unlabeled data
            y_u_pred = self(x_u, training=True)
            loss_unlab_value = 0.
            for alpha_schd, loss_fn in zip(self.alpha_schedule, self.loss_unlabeled):
                alpha = alpha_schd(self.optimizer.iterations)
                loss_unlab_value += alpha * loss_fn(y_u, y_u_pred)

            # Compute total loss with weight alpha(epoch)
            total_loss = loss_lab_value + loss_unlab_value

        # Compute gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict of metrics for monitoring
        results = {m.name: m.result() for m in self.metrics}
        results['loss_labeled'] = loss_lab_value
        results['loss_unlabeled'] = loss_unlab_value
        results['total_loss'] = total_loss

        return results
