import numpy as np

import one.deeplearning.modules.base as base


class Linear(base.Module):
    """Linear layer"""

    # A unique id that is incremented each time a linear layer is created
    id = 0

    def __init__(self, dim_in: int, dim_out: int):
        """ Kaiming initialization

        :param dim_in:
        :param dim_out:
        """
        super(Linear, self).__init__()
        self.id = Linear.id
        Linear.id += 1
        # Initialize weight
        self.params['w'] = base.Parameter(np.random.randn(dim_in, dim_out) * ((2 / dim_in) ** 0.5))
        # initialize bias
        self.params['b'] = base.Parameter(np.zeros(dim_out))

    def forward(self, input: base.Tensor) -> base.Tensor:
        """Compute w*input+b"""
        output_data = np.einsum('ij,jk->ik', input.data, self.params['w'].data) + self.params['b'].data
        output = base.Tensor(data=output_data, module=self, state={'input_data': input.data}, previous=[input])
        return output

    def backward(self, output: base.Tensor, grad_output: np.ndarray):
        input_data = output.state['input_data']
        self.params['w'].grad = self.params['w'].grad + np.einsum('ij,ik->jk', input_data, grad_output)
        self.params['b'].grad = self.params['b'].grad + np.einsum('ij->j', grad_output)
        grad_input = np.einsum('ik,jk->ij', grad_output, self.params['w'].data)
        return [grad_input]

    def all_params(self):
        return self.params

    def name(self):
        return 'Linear_' + str(self.id)


class Relu(base.Module):
    """RELU activation layer"""

    # A unique id that is incremented each time a Relu layer is created
    id = 0

    def __init__(self):
        super(Relu, self).__init__()
        self.id = Relu.id
        Relu.id += 1

    def forward(self, input: base.Tensor):
        positive = input.data > 0
        output_data = positive * input.data
        output = base.Tensor(data=output_data, module=self, state={'positive': positive}, previous=[input])
        return output

    def backward(self, output: base.Tensor, grad_output: np.ndarray):
        positive = output.state['positive']
        grad_input = positive * grad_output
        return [grad_input]

    def all_params(self):
        return self.params

    def name(self):
        return 'RELU_' + str(self.id)


class Dropout(base.Module):
    """Dropout layer"""

    # A unique id that is incremented each time a Relu layer is created
    id = 0

    def __init__(self, rate):
        super(Dropout, self).__init__()
        self.id = Dropout.id
        Dropout.id += 1
        self.rate = rate

    def forward(self, input: base.Tensor):
        mask = np.random.choice([False, True], size=input.data.shape, p=[self.rate, 1-self.rate])
        output_data = mask * input.data
        output = base.Tensor(data=output_data, module=self, state={'mask': mask}, previous=[input])
        return output

    def backward(self, output: base.Tensor, grad_output: np.ndarray):
        mask = output.state['mask']
        grad_input = mask * grad_output
        return [grad_input]

    def all_params(self):
        return self.params

    def name(self):
        return 'Dropout_' + str(self.id)


if __name__ == '__main__':
    input = base.Tensor(np.random.rand(2, 3))
    l1 = Linear(3, 4)
    d1 = Dropout(0.5)
    r1 = Relu()
    output = l1(input)
    output = d1(output)
    output = r1(output)
    print(output.data)
    output.backward()