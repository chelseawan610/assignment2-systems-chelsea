import  torch

from torch.utils.checkpoint import checkpoint

def run_blocks(blocks, x):
    for block in blocks:
        x = block(x)
    return x

def run_blocks_checkpointed(blocks, x):
    def checkpoint_fn(tensor):
        return run_blocks(blocks, tensor)
    return checkpoint(checkpoint_fn, x, use_reentrant=False)


