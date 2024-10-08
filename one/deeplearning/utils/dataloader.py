from typing import List, Dict
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Dataloader(ABC):
    """Base class for dataloaders"""
    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Should return an iterator for data"""
        raise NotImplementedError


class NumpyDataIterator(object):
    """Iterator on numpy array dataset"""
    def __init__(self, x: np.ndarray, y:np.ndarray,
                 batch_size=1, shuffle=False):
        self.x = x
        self.y = y
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        self.inds = np.arange(len(self.x))
        if self.shuffle:
            # In the beginning of iteration, shuffle the indicies
            np.random.shuffle(self.inds)
        self.i = 0
        return self

    def __next__(self):
        if self.i < len(self.x):
            inds = self.inds[self.i:min(self.i+self.batch_size, len(self.x))]
            x = self.x[inds]
            y = self.y[inds]
            self.i += self.batch_size
            return x, y
        else:
            raise StopIteration


class ClassificationCsvDataLoader(Dataloader):
    """Data loader for csv datasets for classification"""
    def __init__(self, path: str,
                 y_labels: List['str'],
                 batch_size=1,
                 shuffle=False,
                 val_ratio=0.,
                 filter_y: List = None,
                 map_y=None,
                 map_x=None):
        """
        :param path: path to the csv file
        :param y_labels: label of the csv file corresponding to the output feature
        :param shuffle: whether to shuffle the data each time iterating over it
        :param val_ratio: ratio of data dedicated to validation set, first val_ratio % of data will become validation data
        :param filter_y: only consider rows that have target value in filter_y
        :param map_y: function applied to all targets
        :param map_x: function applied to all input features
        """
        df = pd.read_csv(path)
        self.shuffle = shuffle
        x_labels = [label for label in df.columns if label not in y_labels]
        self.x = df.loc[:, x_labels].values.astype(np.float)
        self.y = df.loc[:, y_labels].values.astype(np.int)
        n_total = len(self.x)
        if filter_y:
            inds_to_keep = [i for i in range(n_total) if self.y[i, 0] in filter_y]
            self.x = self.x[inds_to_keep]
            self.y = self.y[inds_to_keep]
        if map_y:
            for i in range(len(self.y)):
                self.y[i] = map_y(self.y[i])
        if map_x:
            for i in range(len(self.x)):
                self.x[i] = map_x(self.x[i])

        n_kept = len(self.x)
        n_val = int(n_kept*val_ratio)
        self.x_val = self.x[:n_val]
        self.x = self.x[n_val:]
        self.y_val = self.y[:n_val]
        self.y = self.y[n_val:]
        self.data_iterator = NumpyDataIterator(self.x, self.y, batch_size, shuffle)
        self.data_iterator_val = NumpyDataIterator(self.x_val, self.y_val, batch_size, shuffle)

    def __call__(self, validation=False) -> NumpyDataIterator:
        """
        :param validation: whether to return validation data or not
        :return: iterator on data
        """
        if validation:
            return self.data_iterator_val
        else:
            return self.data_iterator


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    train = False
    if not train:
        cdl = ClassificationCsvDataLoader('../../files/dataset/mnist_test.csv', ['7'],
                                          batch_size=2, shuffle=True, val_ratio=0.1, filter_y=[2, 7],
                                          map_y=lambda y: 1 if y == 7 else 0,
                                          map_x=(lambda x: (x / 255.)))
        for x, y in cdl():
            print(y)
            plt.imshow(x[-1].reshape(28, 28))
            plt.show()
    else:
        cdl = ClassificationCsvDataLoader('../../files/dataset/mnist_train.csv', ['5'],
                                          batch_size=32, shuffle=True, val_ratio=0.1,
                                          filter_y=[2, 7],
                                          map_y=lambda y: 1 if y == 7 else 0,
                                          map_x=(lambda x: (x / 255.)))

        for x, y in cdl():
            print(y)
            plt.imshow(x[-1].reshape(28, 28))
            plt.show()


