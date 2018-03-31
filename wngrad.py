import torch
from torch.optim import Optimizer


class WNGrad(Optimizer):
    def __init__(self, params, lr=0.56):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Learning rate adjustment
                    state['bj'] = 1.0

                state['step'] += 1
                state['bj'] += \
                    (group['lr']**2)/(state['bj'])*(grad.norm(1))**2

                p.data.sub_(group['lr'] / state['bj'] * grad)
        return loss
