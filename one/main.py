import numpy as np
import time
from matplotlib import pyplot as plt

from one.deeplearning.modules.sequential import Sequential
from one.deeplearning.modules.linear import Linear, Relu, Dropout
from one.deeplearning.optim.sgd import Sgd
from one.deeplearning.utils.metrics import Accuracy
from one.deeplearning.modules.loss import BinaryCrossEntropySigmoid
from one.deeplearning.utils.dataloader import ClassificationCsvDataLoader
from one.deeplearning.modules.base import Tensor


def compute_metric(model, metric_func, dataloader_iterator):
    n_batches = 0
    metric = 0.
    for x, y in dataloader_iterator:
        n_batches += 1
        x = Tensor(x)
        y_hat = model(x)
        metric = metric + metric_func(y_hat.data > 0, y)
    metric /= n_batches
    return metric


def compute_confusion_matrix(model, dataloader_iterator):
    confusion_matrix_samples = [[None for _ in range(2)] for _ in range(2)]
    confusion_matrix_counts = np.zeros((2, 2))
    for x, y in dataloader_iterator:
        x = Tensor(x)
        y_hat = model(x)
        y_hat = y_hat.data > 0
        y_hat = y_hat.astype(int)
        for i in range(len(y_hat)):
            confusion_matrix_samples[y[i, 0]][y_hat[i, 0]] = x.data[i]
            confusion_matrix_counts[y[i, 0]][y_hat[i, 0]] += 1
    return confusion_matrix_samples, confusion_matrix_counts


def show_confusion_matrix_samples(confusion_matrix_samples, image_shape):
    fig, ax = plt.subplots(nrows=2, ncols=2)

    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            if confusion_matrix_samples[i][j] is not None:
                col.imshow(confusion_matrix_samples[i][j].reshape(image_shape))
                col.axis('off')

    fig.suptitle('Confusion Matrix Image Samples')
    plt.show()


def train_cross_validate(model, optimizer, dataloader, loss_func, metric_func, n_epochs):
    t0 = time.time()
    for epoch in range(n_epochs):
        epoch_loss_train = 0.
        epoch_metric_train = 0.
        n_batches = 0
        for x, y in dataloader(validation=False):
            n_batches += 1
            x = Tensor(x)
            model.zero_grad()
            y_hat = model(x)
            loss = loss_func(y_hat, y)
            loss.backward()
            optimizer.step()
            epoch_loss_train = epoch_loss_train + loss.data[0, 0]
            epoch_metric_train = epoch_metric_train + metric_func(y_hat.data > 0, y)

        epoch_loss_train /= n_batches
        epoch_metric_train /= n_batches
        epoch_metric_val = compute_metric(model, metric_func, dataloader(validation=True))
        print('epoch: %s, train_loss: %.3f, train_metric: %.3f, val_metric: %.3f'
              % (epoch, epoch_loss_train, epoch_metric_train, epoch_metric_val))

    print('Finished training in %.3f seconds' % (time.time()-t0))


def main():
    np.random.seed(0)
    dataloader_train = ClassificationCsvDataLoader('files/dataset/mnist_train.csv', ['5'],
                                                   batch_size=32, shuffle=True, val_ratio=0.1,
                                                   filter_y=[2, 7],
                                                   map_y=lambda y: 1 if y == 7 else 0,
                                                   map_x=(lambda x: (x / 255.))
                                                   )
    dataloader_test = ClassificationCsvDataLoader('files/dataset/mnist_test.csv', ['7'],
                                                  batch_size=32,
                                                  filter_y=[2, 7],
                                                  map_y=lambda y: 1 if y == 7 else 0,
                                                  map_x=(lambda x: (x / 255.))
                                                  )
    dim_in = dataloader_train.x.shape[1]
    model = Sequential(
        [
            Linear(dim_in, 100),
            Dropout(0.2),
            Relu(),
            Linear(100, 1)
        ]
    )
    loss = BinaryCrossEntropySigmoid()
    metric = Accuracy(mode='avg')
    optimizer = Sgd(model.all_params(), lr=0.0001)
    train_cross_validate(model, optimizer, dataloader_train, loss, metric, 10)

    test_metric = compute_metric(model, metric, dataloader_test())
    print('Test metric: %.3f' % test_metric)
    test_confusion_matrix_samples, confusion_matrix_counts = compute_confusion_matrix(model, dataloader_test())
    confusion_matrix_ratio = confusion_matrix_counts/np.sum(confusion_matrix_counts)
    print('Test data confusion matrix:\n%s' % confusion_matrix_ratio)
    show_confusion_matrix_samples(test_confusion_matrix_samples, (28, 28))


if __name__ == '__main__':
    main()