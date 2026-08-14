import torch
import torch.distributed as dist


def make_sharded_optimizer(params, optimizer_cls, **kwargs):
    return _DistributedOptimizer(params, optimizer_cls, **kwargs)


class _DistributedOptimizer:
    def __init__(self, params, optimizer_cls, **kwargs):
        self._optimizer = optimizer_cls(params, **kwargs)

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    @property
    def state(self):
        return self._optimizer.state

    def zero_grad(self, set_to_none=True):
        return self._optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            for group in self.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        dist.all_reduce(parameter.grad)
                        parameter.grad.div_(world_size)
        return self._optimizer.step(closure=closure)

def step(self, closure=None):
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad)
                    parameter.grad.div_(world_size)

    return self._optimizer.step(closure=closure)
