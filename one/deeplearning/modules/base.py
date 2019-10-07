from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict


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

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward propagation"""
        raise NotImplementedError

    @abstractmethod
    def backward(self, output, grad_output: np.ndarray) -> List:
        """Backward propagation

        :param output: output Tensor previously created by this module
        :param grad_output: gradient of output
        :return: For every input must return a grad_input, in the same order
        """
        raise NotImplementedError

    @abstractmethod
    def all_params(self) -> Dict:
        """Return all parameters contained in the module for optimization"""
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        """A unique and constant name for the module"""
        raise NotImplementedError

    def zero_grad(self):
        """Set gradients to 0, to reset the result of previous backpropagations"""
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
        First dimension is for batches in all modeules we create
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

                
class Optimizer(ABC):
    """Base class for optimizers"""
    @abstractmethod
    def step(self):
        raise NotImplementedError
