import numpy as np
from matplotlib import pyplot as plt

from keras import models, layers, losses, datasets, utils, optimizers, callbacks
from keras import backend as K


class LearningRateSchedulerPerBatch(callbacks.callbacks.Callback):

    def __init__(self, schedule, verbose=0):
        super(LearningRateSchedulerPerBatch, self).__init__()
        self.schedule = schedule
        self.verbose = verbose
        self.loss_history = []
        self.lr_history = []

    def on_train_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        try:  # new API
            lr = self.schedule(batch, lr)
        except TypeError:  # old API for backward compatibility
            lr = self.schedule(batch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler setting learning '
                  'rate to %s.' % (batch + 1, lr))
        self.lr_history.append(lr)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        self.loss_history.append(logs.get('loss'))


def find_lr(model, x, y, lr_max, lr_min, batch_size):
    K.set_value(model.optimizer.lr, lr_min)
    n = len(x)
    n_batches = int(n/batch_size)
    decay = (lr_min/lr_max) ** (1/n_batches)
    schedule = lambda i_batch: lr_min / (decay ** i_batch)
    scheduler = LearningRateSchedulerPerBatch(schedule, verbose=1)
    model.fit(x, y,
              batch_size=batch_size,
              epochs=1,
              callbacks=[scheduler],
              shuffle=True)

    plt.plot(scheduler.loss_history, np.log10(np.array(scheduler.lr_history)))
    plt.show()

