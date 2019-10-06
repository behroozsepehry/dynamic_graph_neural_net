from typing import List, Dict
from abc import ABC, abstractmethod
import numpy as np


class Parameter(object):
    """Parameter of modules storing both the values and gradients"""
    def __init__(self, vals: np.array):
        self.vals = vals
        self.grad = 0

    def zero_grad(self):
        self.grad = 0


class Tensor(object):
    """Extend numpy arrays for dynamic computational graphs"""
    def __init__(self, data):
        self.data = data
        # List of previous Tensors generating this tensor for backward propagation
        self.previous = []
        # Keeping a state for backward propagation
        self.state = {}
        # The module that generated this tensor
        self.module = None


class Module(ABC):
    """Base module for all computational objects"""
    def __init__(self):
        # Dictionary of parameters
        self.params = {}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __call_backward__(self, *args, **kwargs):
        return self.backward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward propagation"""
        raise NotImplementedError

    @abstractmethod
    def backward(self, *args, **kwargs):
        """Backward propagation"""
        raise NotImplementedError


class Linear(Module):
    """Linear layer"""
    def __init__(self, dim_in: int, dim_out: int):
        """ Kaiming initialization

        :param dim_in:
        :param dim_out:
        """
        # Initialize weight
        self.params['w'] = Parameter(np.random.randn(dim_in, dim_out) * ((2/dim_in)**0.5))
        # initialize bias
        self.params['b'] = Parameter(np.zeros(dim_out))

    def forward(self, x):
        """
        :param x: Tensor input
        :return:
        """
        pass
