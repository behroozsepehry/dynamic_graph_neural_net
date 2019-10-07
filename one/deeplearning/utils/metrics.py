from abc import ABC, abstractmethod
import numpy as np


class Metric(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Accuracy(Metric):
    """Count the number of elements equal in two numpy arrays"""
    def __init__(self, mode: str = 'avg'):
        if mode not in ['sum', 'avg']:
            raise ValueError('mode must be sum or avg')
        self.mode = mode

    def __call__(self, y_hat: np.ndarray, y: np.ndarray):
        total_size = np.prod(y.shape[0])
        val = np.sum(y == y_hat)
        if self.mode == 'avg':
            val /= total_size
        return val
