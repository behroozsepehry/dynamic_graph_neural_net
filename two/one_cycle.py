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


def exponential_smoothing(series, smoothing_beta=0.9):
    smooth_series = series.copy()
    for i in range(1, len(series)):
        smooth_series[i] = (1-smoothing_beta) * smooth_series[i] + smoothing_beta * smooth_series[i-1]
    return smooth_series


def find_truncate_ind_after_min(arr, ratio):
    i = np.argmin(arr)
    arr_min = np.min(arr)
    while arr[i] < ratio * arr_min:
        i += 1
    return i


class CycleLrSchedule(object):
    def __init__(self, n_epochs, lr_max, lr_min):
        self.n_epochs = n_epochs
        self.lr_min = lr_min
        self.lr_max = lr_max

    def __call__(self, epoch):
        epoch_mid = self.n_epochs//2
        if epoch < epoch_mid:
            lr = self.lr_min * (1. - epoch/epoch_mid) + self.lr_max * (epoch/epoch_mid)
        else:
            lr = self.lr_min * (epoch/epoch_mid - 1.) + self.lr_max * (2. - epoch/epoch_mid)
        return lr


def plot_lr_finding(history_log_lr_truncated, history_loss_smooth_truncated):
    plt.plot(history_log_lr_truncated, history_loss_smooth_truncated)
    plt.xlabel('learning rate (log10 scale)')
    plt.ylabel('loss')
    plt.show()


def plot_train_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def find_lr(model, x, y, lr_max, lr_min, lr_scale, batch_size, smoothing_beta, verbose=0):
    K.set_value(model.optimizer.lr, lr_min)
    n = len(x)
    n_batches = int(n/batch_size)
    decay = (lr_min/lr_max) ** (1/n_batches)
    schedule = lambda i_batch: lr_min / (decay ** i_batch)
    scheduler = LearningRateSchedulerPerBatch(schedule, verbose=verbose)
    model.fit(x, y,
              batch_size=batch_size,
              epochs=1,
              callbacks=[scheduler],
              shuffle=True)

    history_log_lr = np.log10(np.array(scheduler.lr_history))
    history_loss_smooth = exponential_smoothing(scheduler.loss_history, smoothing_beta)
    i = find_truncate_ind_after_min(history_loss_smooth, 1.5)
    history_log_lr_truncated = history_log_lr[:i]
    history_loss_smooth_truncated = history_loss_smooth[:i]

    lr_best = scheduler.lr_history[np.argmin(history_loss_smooth)]
    lr_picked = lr_best * lr_scale
    if verbose:
        print('Searched for learning rate between %s and %s, picked %s' % (lr_max, lr_min, lr_picked))
        plot_lr_finding(history_log_lr_truncated, history_loss_smooth_truncated)
    return lr_picked

