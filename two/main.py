import copy
from matplotlib import pyplot as plt

from keras import models, layers, losses, datasets, utils, optimizers, callbacks
from keras import backend as K

from two import one_cycle


def get_model(optimizer):
    # Lenet 5 based on https://engmrk.com/lenet-5-a-classic-cnn-architecture/
    # We do not implement the symmetry breaking of the original paper of LeCun
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
    optimizer = optimizers.SGD(learning_rate=0.1)
    return optimizer


def get_dataset():
    # Code based on https://keras.io/examples/cifar10_cnn/
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
    batch_size = 32
    epochs = 10
    validation_split = 0.1
    lr_max = 1.
    lr_min = 1e-08
    lr_scale = 0.1
    smoothing_beta = 0.98
    verbose = 1

    optimizer = get_optimizer()
    model = get_model(optimizer)
    model_untrained1 = copy.deepcopy(model)
    model_untrained2 = copy.deepcopy(model)

    (x_train, y_train), (x_test, y_test) = get_dataset()
    lr = one_cycle.find_lr(model, x_train, y_train,
                      lr_max=lr_max, lr_min=lr_min, lr_scale=lr_scale,
                      batch_size=batch_size, smoothing_beta=smoothing_beta, verbose=verbose)

    K.set_value(model_untrained1.optimizer.lr, lr)
    schedule_one_cycle = one_cycle.CycleLrSchedule(n_epochs=epochs, lr_max=lr, lr_min=lr*lr_scale)
    scheduler_one_cycle = callbacks.callbacks.LearningRateScheduler(schedule_one_cycle, verbose=verbose)
    history_one_cycle = model_untrained1.fit(x_train, y_train,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             validation_split=validation_split,
                                             callbacks=[scheduler_one_cycle],
                                             shuffle=True)
    one_cycle.plot_train_history(history_one_cycle)

    K.set_value(model_untrained1.optimizer.lr, lr)
    history_const_lr = model_untrained2.fit(x_train, y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         validation_split=validation_split,
                         shuffle=True)
    one_cycle.plot_train_history(history_const_lr)



if __name__ == '__main__':
    main()