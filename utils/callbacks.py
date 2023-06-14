import random

import numpy as np
import pandas as pd
from keras.callbacks import Callback
from keras.utils import io_utils

import wandb


class AdditionalValidationSets(Callback):
    def __init__(self, validation_sets, path, verbose=0):
        """
        :param validation_sets:
        a list of 2-tuples (dataset, name) where data set if a tf.data.Dataset object and name is a string
        a list of 3-tuples (dataset, steps, name) where
            data set if a tf.data.Dataset object,
            steps is an integer
            and name is a string
        :param verbose:
        verbosity mode, 1 or 0
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [2, 3]:
                raise ValueError()
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.path = path
        self.results = {}

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        if self.verbose == 1:
            io_utils.print_msg('additional metrics:')
        for validation_set in self.validation_sets:
            if len(validation_set) == 2:
                dataset, validation_set_name = validation_set

                results = self.model.evaluate(dataset,
                                              verbose=self.verbose)
            else:
                dataset, steps, validation_set_name = validation_set

                results = self.model.evaluate(dataset,
                                              steps=steps,
                                              verbose=self.verbose)

            if len(self.model.metrics) > 1:
                for metric, result in zip(self.model.metrics, results):
                    valuename = validation_set_name + '_' + metric.name
                    self.history.setdefault(valuename, []).append(result)
                    if self.verbose == 1:
                        io_utils.print_msg(f'{valuename}: {result}')
            else:
                valuename = validation_set_name + '_loss'
                self.history.setdefault(valuename, []).append(results)
                if self.verbose == 1:
                    io_utils.print_msg(f'{valuename}: {results}')

    def on_train_end(self, logs=None):
        df = pd.DataFrame(self.history)
        df.to_csv(self.path)


class WandBImageLogger(Callback):
    def __init__(self, image, frequency=1, key=None):
        super().__init__()
        self.image = image
        self.frequency = frequency
        self.key = key
        wandb.log({'image': wandb.Image(image)})

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.frequency != 0:
            return
        if self.key is not None:
            prediction = self.model(self.image[np.newaxis, ...])[self.key]
        else:
            prediction = self.model(self.image[np.newaxis, ...])
        prediction = np.array(prediction)[0, ...]
        wandb.log({'prediction': wandb.Image(prediction)})
