from abc import ABC, abstractmethod
import numpy as np


class Parameter(object):
    """Parameter of modules storing both the values and gradients"""
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = 0

    def zero_grad(self):
        self.grad = 0


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

    @abstractmethod
    def all_params(self):
        """Return all parameters contained in the module for optimization"""
        raise NotImplementedError


class Tensor(object):
    """Extend numpy arrays for dynamic computational graphs
    Usually first dimension is for batches
    """
    def __init__(self, data: np.ndarray, module: Module = None, **kwargs):
        self.data = data
        # List of previous Tensors generating this tensor for backward propagation
        self.previous = kwargs.get('previous', [])
        # Keeping a state for backward propagation
        self.state = kwargs.get('state', {})
        # The module that generated this tensor
        self.module = module
