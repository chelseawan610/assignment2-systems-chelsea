import torch
import torch.distributed as dist


class CPUFSDP(torch.nn.Module):
    def __init__(self, module, compute_dtype=None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        if dist.is_initialized():
            with torch.no_grad():
                for parameter in self.module.parameters():
                    dist.broadcast(parameter, src=0)
        for parameter in self.module.parameters():
            if parameter.requires_grad and dist.is_initialized():
                parameter.register_hook(self._average_gradient)
        if compute_dtype is not None:
            from cs336_basics.model import Embedding, Linear
            for submodule in self.module.modules():
                if not isinstance(submodule, (Linear, Embedding)):
                    continue
                submodule.register_forward_pre_hook(self._make_forward_pre_hook(compute_dtype))
                submodule.register_forward_hook(self._restore_forward_hook)
                if isinstance(submodule, Linear):
                    submodule.register_full_backward_pre_hook(self._make_backward_pre_hook(compute_dtype))
                submodule.weight.register_post_accumulate_grad_hook(
                    self._make_gradient_restore_hook(submodule, isinstance(submodule, Linear))
                )

    @staticmethod
    def _make_forward_pre_hook(dtype):
        def hook(module, inputs):
            module._master_weight = module.weight.data
            module.weight.data = module.weight.data.to(dtype)
        return hook

    @staticmethod
    def _restore_forward_hook(module, inputs, output):
        module.weight.data = module._master_weight
        del module._master_weight
        module.weight.grad = None

    @staticmethod
    def _make_backward_pre_hook(dtype):
        def hook(module, grad_output):
            module._master_weight_backward = module.weight.data
            module.weight.data = module.weight.data.to(dtype)
            module.weight.grad = None
        return hook

    @staticmethod
    def _make_gradient_restore_hook(module, is_linear):
        def hook(parameter):
            if is_linear and hasattr(module, "_master_weight_backward"):
                module.weight.data = module._master_weight_backward
                del module._master_weight_backward
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.to(torch.float32)
        return hook

    def _average_gradient(self, gradient):
        dist.all_reduce(gradient)
        gradient.div_(self.world_size)
        return gradient

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def gather_full_params(model):
    return {name: parameter.detach().clone() for name, parameter in model.module.named_parameters()}


def finish_backward(model, optimizer):
    return None
