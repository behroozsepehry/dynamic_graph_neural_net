import copy
import numpy as np

import tensorflow
from keras import models, layers, losses, datasets, utils, optimizers, callbacks
from keras import backend as K

from two import one_cycle


def get_model(optimizer):
    """Lenet 5
    code based on based on https://engmrk.com/lenet-5-a-classic-cnn-architecture/
    We do not implement the symmetry breaking of the original paper of LeCun
    """
    model = models.Sequential([
        layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1),
                      activation='tanh', input_shape=(32, 32, 3), padding='valid'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        layers.Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'),
        layers.Flatten(),
        layers.Dense(84, activation='tanh'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def get_optimizer():
    optimizer = optimizers.SGD()
    return optimizer


def get_dataset():
    """CIFAR 10 dataset
    Code based on https://keras.io/examples/cifar10_cnn/
    """
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    # Convert class vectors to binary class matrices.
    y_train = utils.to_categorical(y_train, num_classes)
    y_test = utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return (x_train, y_train), (x_test, y_test)


def main():
    # Set hyperparameters
    train_data_truncate_ratio = 0.3  # truncate training data for faster experiments
    batch_size = 32
    epochs = 10
    validation_split = 0.3
    lr_max = 1.  # upper bound on search interval for learning rate
    lr_min = 1e-08  # lower bound on search interval for learning rate
    lr_scale_finding = 1.0  # How to scale the best learning rate found, which could be large
    lr_scale_training = 0.1  # Ratio of smallest learning rate used in training to the found learning rate
    smoothing_beta = 0.98  # Smoothing parameter in learning rate finding
    verbose = 1

    # Set seeds
    np.random.seed(0)
    tensorflow.set_random_seed(0)

    # Get model
    optimizer = get_optimizer()
    model = get_model(optimizer)
    model_untrained1 = copy.deepcopy(model)
    model_untrained2 = copy.deepcopy(model)
    model_untrained3 = copy.deepcopy(model)

    # Get data
    (x_train, y_train), (x_test, y_test) = get_dataset()
    truncate_ind = int(train_data_truncate_ratio * len(x_train))
    x_train = x_train[:truncate_ind]
    y_train = y_train[:truncate_ind]

    # Find learning rate
    lr = one_cycle.find_lr(model, x_train, y_train,
                      lr_max=lr_max, lr_min=lr_min, lr_scale=lr_scale_finding,
                      batch_size=batch_size, smoothing_beta=smoothing_beta, verbose=verbose)

    # Train model using one cycle policy on learning rate schedule
    K.set_value(model_untrained1.optimizer.lr, lr)
    schedule_one_cycle = one_cycle.CycleLrSchedule(n_epochs=epochs, lr_max=lr, lr_min=lr*lr_scale_training)
    scheduler_one_cycle = callbacks.callbacks.LearningRateScheduler(schedule_one_cycle, verbose=verbose)
    history_one_cycle = model_untrained1.fit(x_train, y_train,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             validation_split=validation_split,
                                             callbacks=[scheduler_one_cycle],
                                             shuffle=True)

    # Train model using constant learning rate
    K.set_value(model_untrained2.optimizer.lr, lr)
    history_const_lr_large = model_untrained2.fit(x_train, y_train,
                                                  batch_size=batch_size,
                                                  epochs=epochs,
                                                  validation_split=validation_split,
                                                  shuffle=True)

    # Train model using constant learning rate
    K.set_value(model_untrained3.optimizer.lr, lr*lr_scale_training)
    history_const_lr_small = model_untrained3.fit(x_train, y_train,
                                                  batch_size=batch_size,
                                                  epochs=epochs,
                                                  validation_split=validation_split,
                                                  shuffle=True)
    # Plot training history for different learning rate schedules
    one_cycle.plot_train_histories({'One Cycle': history_one_cycle,
                                    'Constant Large': history_const_lr_large,
                                    'Constant Small': history_const_lr_small})


if __name__ == '__main__':
    main()