from typing import List
import numpy as np

from one.deeplearning.modules import base, linear, loss


class Sequential(base.Module):
    """Sequential module"""

    # Unique id
    id = 0

    def __init__(self, layers: List[base.Module]):
        """
        :param layers: list of modules
        """
        super(Sequential, self).__init__()
        self.id = Sequential.id
        Sequential.id += 1
        self.layers = layers

    def forward(self, input: base.Tensor):
        y = input
        for layer in self.layers:
            y = layer(y)
        return y

    def backward(self, output, grad_output: np.ndarray):
        """No need to define backward
        as all the modules in the sequence should already have backward"""
        pass

    def all_params(self):
        params = {layer.name(): layer.all_params() for layer in self.layers}
        return params

    def name(self):
        return 'Sequential'+str(self.id)


if __name__ == '__main__':
    model = Sequential(
        [
            linear.Linear(3, 2),
            linear.Relu(),
            linear.Linear(2, 1),
        ]
    )

    input = base.Tensor(np.random.rand(2, 3))
    target = np.array([[1.], [0.]])
    output = model(input)
    bce_layer = loss.BinaryCrossEntropySigmoid()
    bce_tensor = bce_layer(output, target)
    bce_tensor.backward()
    print('output', output)
    print('all_params', model.all_params())
    model.zero_grad()

