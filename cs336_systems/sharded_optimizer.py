import torch
import torch.distributed as dist


def make_sharded_optimizer(params, optimizer_cls, **kwargs):
    """Compatibility baseline; true state partitioning is GPU/distributed work."""
    return optimizer_cls(params, **kwargs)
