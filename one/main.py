from one.deeplearning.modules.sequential import Sequential
from one.deeplearning.modules.linear import Linear, Relu, Dropout
from one.deeplearning.optim.sgd import Sgd
from one.deeplearning.utils.metrics import Accuracy
from one.deeplearning.modules.loss import BinaryCrossEntropySigmoid
from one.deeplearning.utils.dataloader import ClassificationCsvDataLoader
from one.deeplearning.modules.base import Tensor

def train_cross_validate(model, optimizer, dataloader, loss_func, metric_func, n_epochs):
    for epoch in range(n_epochs):
        epoch_loss = 0.
        epoch_metric = 0.
        n_batches = 0
        for x, y in dataloader():
            x = Tensor(x)
            n_batches += 1
            model.zero_grad()
            y_hat = model(x)
            loss = loss_func(y_hat, y)
            loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss.data
            epoch_metric = epoch_metric + metric_func(y_hat.data > 0, y)

        epoch_loss /= n_batches
        epoch_metric /= n_batches
        print('epoch: %s, loss: %s, metric: %s' % (epoch, epoch_loss, epoch_metric))


def main():
    # dataloader_train = ClassificationCsvDataLoader('files/dataset/mnist_train.csv', ['5'],
    #                                                batch_size=32, shuffle=True, val_ratio=0.1,
    #                                                filter_y=[2, 7],
    #                                                map_y=lambda y: 1 if y == 7 else 0,
    #                                                map_x=(lambda x: (x / 255.))
    #                                                )
    dataloader_test = ClassificationCsvDataLoader('files/dataset/mnist_test.csv', ['7'],
                                                  batch_size=32,
                                                  filter_y=[2, 7],
                                                  map_y=lambda y: 1 if y == 7 else 0,
                                                  map_x=(lambda x: (x / 255.))
                                                  )
    dataloader_train = dataloader_test
    dim_in = dataloader_train.x.shape[1]
    model = Sequential(
        [
            Linear(dim_in, 1),
        ]
    )
    loss = BinaryCrossEntropySigmoid()
    metric = Accuracy(mode='avg')
    optimizer = Sgd(model.all_params(), lr=0.0001)
    train_cross_validate(model, optimizer, dataloader_train, loss, metric, 100)

if __name__ == '__main__':
    main()