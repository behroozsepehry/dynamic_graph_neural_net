import numpy as np
import one.deeplearning.modules.base as base


class Linear(base.Module):
    """Linear layer"""
    def __init__(self, dim_in: int, dim_out: int):
        """ Kaiming initialization

        :param dim_in:
        :param dim_out:
        """
        super(Linear, self).__init__()
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
        self.params['w'].grad = np.einsum('ij,ik->jk', input_data, grad_output)
        self.params['b'].grad = np.einsum('ij->i', grad_output)
        grad_input = np.einsum('ik,jk->ij', grad_output, self.params['w'].data)
        return grad_input


class Relu(base.Module):
    