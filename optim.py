"""
Phil Wang's PyTorch Lion implementation: https://github.com/lucidrains/lion-pytorch
"""

from typing import Callable, Optional

import torch
from torch.optim.optimizer import Optimizer


def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    # stepweight decay
    p.data.mul_(1 - lr * wd)

    # weight update
    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1).sign_()
    p.add_(update, alpha=-lr)

    # decay the momentum running average coefficient
    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)


class Lion(Optimizer):
    def __init__(
        self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, fused=False
    ):
        assert lr > 0.0
        assert all(0.0 <= beta <= 1.0 for beta in betas)

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)

        super().__init__(params, defaults)

        self.update_fn = update_fn

        if fused:
            # not implemented
            from lion_pytorch.triton import update_fn as triton_update_fn

            self.update_fn = triton_update_fn

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: p.grad is not None, group["params"]):
                lr, wd, beta1, beta2, state = (
                    group["lr"],
                    group["weight_decay"],
                    *group["betas"],
                    self.state[p],
                )

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                self.update_fn(p, p.grad, exp_avg, lr, wd, beta1, beta2)

        return loss
