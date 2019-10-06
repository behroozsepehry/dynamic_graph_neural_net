from abc import ABC, abstractmethod
import numpy as np


class Parameter(object):
    """Parameter of modules storing both the values and gradients"""
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = 0.

    def zero_grad(self):
        self.grad = 0.


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
    def backward(self, output, grad_output: np.ndarray):
        """Backward propagation

        :param output:
        :param grad_output:
        :return: (List) For every input should return a grad_input, in the same order
        """
        raise NotImplementedError

    @abstractmethod
    def all_params(self):
        """Return all parameters contained in the module for optimization"""
        raise NotImplementedError

    @abstractmethod
    def name(self):
        """A unique and constant name for the module"""
        raise NotImplementedError

    def zero_grad(self):
        def zero_grad_param(p):
            if type(p) is Parameter:
                p.zero_grad()
            else:
                for pp in p.values():
                    zero_grad_param(pp)

        params = self.all_params()
        zero_grad_param(params)


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

    def backward(self, grad: np.ndarray = None):
        """Backpropagation algorithm"""

        # If last layer, use identity gradient
        if grad is None:
            grad = np.ones(self.data.shape)

        # For previous layers, backpropagate the gradient
        if self.module:
            grads_previous = self.module.backward(self, grad)
            for i, g in enumerate(grads_previous):
                self.previous[i].backward(g)
