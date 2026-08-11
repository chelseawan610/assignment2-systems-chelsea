import argparse
import timeit
import statistics

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
    "10B": {
        "d_model": 4608,
        "d_ff": 12288,
        "num_layers": 50,
        "num_heads": 36,
    },
}
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        choices=["forward", "forward_backward", "full_step"],
                        default="forward",
                        help="Select the benchmark scope")
    parser.add_argument("--warmup-steps",
                        type=int,
                        default=5
                        )
    parser.add_argument("--measurement-steps",
                        type=int,
                        default=10
                        )
    parser.add_argument("--model-size",
                        type=str,
                        choices=["small", "medium", "large", "xl", "10B"],
                        default="small",
                        help="Select the model size")
    parser.add_argument("--context-length",
                        type=int,
                        default=512,
                        help="input the context length")
    args = parser.parse_args()
    if args.warmup_steps < 0:
        parser.error("warmup-steps must be non-negative")
    if args.measurement_steps <= 0:
        parser.error("measurement-steps must be positive")
    if args.context_length <= 0:
        parser.error("context-length must be positive")
    return args


def get_model_config(args):
    model_config = dict(MODEL_CONFIGS[args.model_size])
    model_config["vocab_size"] = 10000
    model_config["context_length"] = args.context_length
    model_config["rope_theta"] = 10000.0
    return model_config


if __name__ == "__main__":
    args = parse_args()
    model_config = get_model_config(args)

# create model
    model = BasicsTransformerLM(**model_config)
    print("Model created successfully.")
    print("Number of parameters:", model.get_num_params())

# create batch
    batch_size = 4
    batch = torch.randint(
        low=0,
        high=model_config["vocab_size"],
        size=(batch_size, model_config["context_length"]),
        dtype=torch.long,
    )

# create targets
    targets = torch.randint(
        low = 0,
        high = model_config["vocab_size"],
        size = batch.shape,
        dtype = torch.long
    )
#create optimizer
    optimizer = AdamW(model.parameters())


    def forward_step():
        with torch.no_grad():
            _ = model(batch)

    def forward_backward_step():
        optimizer.zero_grad(set_to_none=True)

        logits = model(batch)
        loss = cross_entropy(logits, targets)

        loss.backward()

    def full_step():
        optimizer.zero_grad(set_to_none=True)

        logits = model(batch)
        loss = cross_entropy(logits, targets)

        loss.backward()
        optimizer.step()

    if args.mode == "forward":
        step_fn = forward_step
    elif args.mode == "forward_backward":
        step_fn = forward_backward_step
    elif args.mode == "full_step":
        step_fn = full_step

    for _ in range(args.warmup_steps):
        step_fn()
    print("Warm-up complete.")

    times = timeit.repeat(
        step_fn,
        repeat=args.measurement_steps,
        number=1)
    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0.0

    print("Mean time in milliseconds:", mean_time * 1000)
    print("Std time in milliseconds:", std_time * 1000)
    print("Measured times in seconds:", times)