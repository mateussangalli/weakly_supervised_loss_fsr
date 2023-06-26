from keras.models import Model
from keras.callbacks import Callback


class MeanTeacher(Callback):
    def __init__(self, alpha, teacher: Model):
        if isinstance(alpha, float):
            self.alpha = lambda epoch: alpha

        if callable(alpha):
            self.alpha = alpha

        self.teacher = teacher

    def on_epoch_end(self, epoch, logs):
        student_weights = self.model.get_weights()
        teacher_weights = self.teacher.get_weights()

        alpha = self.alpha(epoch)

        new_weights = list()
        for ws, wt in zip(student_weights, teacher_weights):
            w_new = alpha * wt + (1. - alpha) * ws
            new_weights.append(w_new)

        self.teacher.set_weights(new_weights)
