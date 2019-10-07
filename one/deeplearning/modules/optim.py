from one.deeplearning.modules import base


class Sgd(base.Optimizer):
    """Pure stochastic gradient descent algorithm"""

    def __init__(self, params: dict, lr: float):
        self.params = params
        self.lr = lr

    def step(self):
        """Do one step of sgd"""
        def step_param(p):
            if type(p) is base.Parameter:
                p.data -= self.lr * p.grad
            else:
                for pp in p.values():
                    step_param(pp)

        step_param(self.params)
