from typing import List
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
    def __init__(self, x, y, batch_size=1, shuffle=False):
        self.x = x
        self.y = y
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        self.inds = np.arange(len(self.x))
        if self.shuffle:
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


class CsvDataLoader(Dataloader):

    def __init__(self, path: str,
                 y_labels: List['str'],
                 batch_size=1,
                 shuffle=False,
                 val_ratio=0.,
                 filter_y: List=None):
        """
        :param path: path to the csv file
        :param y_labels: the labels of the csv file corresponding to the target values
        :param shuffle: whether to shuffle the data each time iterating over it
        :param val_ratio: ratio of data dedicated to validation set, first val_ratio% of data will become validation data
        :param filter_y: only consider rows that have target value in filter_y
        """
        df = pd.read_csv(path)
        self.shuffle = shuffle
        x_labels = [label for label in df.columns if label not in y_labels]
        self.x = df.loc[:, x_labels].values
        self.y = df.loc[:, y_labels].values
        n_total = len(self.x)
        if filter_y:
            inds_to_keep = [i for i in range(n_total) if self.y[i, 0] in filter_y]
            self.x = self.x[inds_to_keep]
            self.y = self.y[inds_to_keep]
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
        :return:
        """
        if validation:
            return self.data_iterator_val
        else:
            return self.data_iterator


if __name__ == '__main__':
    cdl = CsvDataLoader('../../files/dataset/mnist_test.csv', ['7'], batch_size=2, shuffle=True, val_ratio=0.1, filter_y=[1,2,3])
    for x, y in cdl():
        print(y)