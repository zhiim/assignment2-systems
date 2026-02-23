from timeit import default_timer as timer

import torch
import torch.cuda.nvtx as nvtx
from cs336_basics.modules import Transformer

VOCAB_SIZE = 10000
BATCH_SIZE = 4


def benchmark(
    context_length: int = 128,
    d_model: int = 768,
    d_ff: int = 3072,
    num_layers: int = 12,
    num_heads: int = 12,
    n_warm_up: int = 5,
    n_steps: int = 10,
    forward_only: bool = True,
):
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # generate input data
    data = torch.randint(
        low=0, high=VOCAB_SIZE, size=(BATCH_SIZE, context_length), device=device
    )

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        device=device,
    )

    # warm up
    model.eval()
    with torch.no_grad():
        for i in range(n_warm_up):
            with nvtx.range(f"Warm-up {i + 1}/{n_warm_up}"):
                model(data)
    # 此处等待所有 warm-up 完成，不需要每轮 warm-up 之后都等待
    # 会破坏流水线导致 warm-up 性能下降
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # forward pass
    forward_times = []
    with torch.no_grad():
        for i in range(n_steps):
            start = timer()
            with nvtx.range(f"Forward pass {i + 1}/{n_steps}"):
                model(data)
                # 此处等待每轮 forward pass 完成，确保时间测量准确
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            end = timer()
            forward_times.append(end - start)

    if forward_only:
        return forward_times

    # backward pass
    model.train()
    backward_times = []

    for i in range(n_steps):
        output = model(data)
        loss = output.mean()
        # ensure forward pass is complete before starting backward pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = timer()
        with nvtx.range(f"Backward pass {i + 1}/{n_steps}"):
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        end = timer()
        backward_times.append(end - start)

    return (forward_times, backward_times)
