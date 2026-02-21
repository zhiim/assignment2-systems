from timeit import default_timer as timer

import torch
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
        for _ in range(n_warm_up):
            model(data)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # forward pass
    forward_times = []
    with torch.no_grad():
        for _ in range(n_steps):
            start = timer()
            model(data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = timer()
            forward_times.append(end - start)

    if forward_only:
        return forward_times

    # backward pass
    model.trian()
    backward_times = []

    for _ in range(n_steps):
        output = model(data)
        loss = output.mean()
        start = timer()
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = timer()
        backward_times.append(end - start)

    return (forward_times, backward_times)
