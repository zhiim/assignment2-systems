import argparse

import numpy as np
import pandas as pd
import yaml

from cs336_systems.benchmark import benchmark

parser = argparse.ArgumentParser(
    prog="A script to perform end-to-end benchmarking of forward (and backward)"
    " pass."
)

parser.add_argument(
    "-c", "--config", required=True, type=str, help="config file"
)

args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.load(f, Loader=yaml.Loader)
params = config["params"]

df = pd.DataFrame.from_dict(params, orient="index")
df = df.assign(forward_time="")
df = df.assign(backward_time="")

context_length = config["context_length"]
n_warm_up = config["n_warm_up"]
n_steps = config["n_steps"]
forward_only = config["forward_only"]

for size, param_dict in params.items():
    forward_time, backward_time = benchmark(
        context_length=context_length,
        d_model=param_dict["d_model"],
        d_ff=param_dict["d_ff"],
        num_layers=param_dict["num_layers"],
        num_heads=param_dict["num_heads"],
        n_warm_up=n_warm_up,
        n_steps=n_steps,
        forward_only=forward_only,
    )
    forward_time = np.array(forward_time)
    backward_time = np.array(backward_time)
    df.at[size, "forward_time"] = (
        f"{forward_time.mean():.4f} ± {forward_time.std():.4f}"
    )
    df.at[size, "backward_time"] = (
        f"{backward_time.mean():.4f} ± {backward_time.std():.4f}"
    )

df.to_markdown("benchmark.md")
