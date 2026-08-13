import torch
import torch.distributed as dist


class DistributedDataParallel(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(parameter, src=0)

        for parameter in self.module.parameters():
            if parameter.requires_grad:
                parameter.register_hook(self._sync_gradient)

    def _sync_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(gradient)
        gradient /= self.world_size
        return gradient

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def finish_gradient_synchronization(
    ddp_model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
) -> None:
    return None
