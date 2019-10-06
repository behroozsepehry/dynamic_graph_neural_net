import numpy as np
import one.deeplearning.modules.base as base


class BinaryCrossEntropySigmoid(base.Module):
    """Applies sigmoid, then finds binary cross entropy"""

    # A unique id that is incremented each time a BinaryCrossEntropySigmoid layer is created
    id = 0

    def __init__(self):
        super(BinaryCrossEntropySigmoid, self).__init__()
        self.id = BinaryCrossEntropySigmoid.id
        BinaryCrossEntropySigmoid.id += 1

    def forward(self, input: base.Tensor, target: np.ndarray):
        # Log sum exp trick to calculate log(exp(x)+exp(0))
        input_data_positive = input.data * (input.data > 0)
        log_sum = input_data_positive + np.log(np.exp(input.data-input_data_positive) + np.exp(-input_data_positive))
        output_data_batch = -(target * (input.data - log_sum) + (1-target) * (-log_sum))
        output_data = np.mean(output_data_batch, axis=0, keepdims=True)
        output = base.Tensor(data=output_data, module=self,
                             state={'log_sum': log_sum, 'target': target}, previous=[input])
        return output

    def backward(self, output: base.Tensor, grad_output: np.ndarray):
        target = output.state['target']
        log_sum = output.state['log_sum']
        batch_size = target.shape[0]
        sigmoids = 1. - (1. / np.exp(log_sum))
        grad_input = (1-target) * sigmoids - target * (1-sigmoids)
        grad_input = grad_input / batch_size
        return [grad_input]

    def all_params(self):
        return self.params

    def name(self):
        return 'BinaryCrossEntropySigmoid_' + str(self.id)


if __name__ == '__main__':
    from one.deeplearning.modules import linear as lin
    l1 = lin.Linear(3, 2)
    l2 = lin.Linear(2, 1)
    ent = BinaryCrossEntropySigmoid()

    input = base.Tensor(np.random.rand(2, 3))
    target = np.array([[1.], [0.]])
    output = base.Tensor(np.array([[1.], [-1.]]))
    ce = ent(output, target)
    print('loss', ce.data)
    print('expected loss', 0.31)

    output1 = l2(l1(input))
    output2 = ent(output1, target)
    print(output2.data)
    output2.backward()