from one.deeplearning.optim import base as obase
from one.deeplearning.modules import base as mbase


class Sgd(obase.Optimizer):
    def __init__(self, params: dict, lr: float):
        self.params = params
        self.lr = lr

    def step(self):
        def step_param(p):
            if type(p) is mbase.Parameter:
                p.data -= self.lr * p.grad
            else:
                for pp in p.values():
                    step_param(pp)

        step_param(self.params)
