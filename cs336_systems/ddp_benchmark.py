import os
import time
import multiprocessing as pymp
from pathlib import Path

import modal
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd


app = modal.App("ddp")

NUM_WARM_UPS = 5
NUM_STEPS = 10

GPU_LIST = [2, 4, 6]

# float32 = 4 bytes, so numel = bytes / 4
DATA_SIZES = [
    ("1MB", 1_000_000 // 4),
    ("10MB", 10_000_000 // 4),
    ("100MB", 100_000_000 // 4),
    ("1GB", 1_000_000_000 // 4),
]


def build_image(*, include_tests: bool = False) -> modal.Image:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .run_commands(
            "apt-get update && apt-get install -y wget",
            "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
            "dpkg -i cuda-keyring_1.1-1_all.deb",
            "apt-get update",
        )
        .apt_install("libcap2-bin", "libdw1", "cuda-nsight-systems-13-2")
        .add_local_dir("cs336-basics", remote_path="/.uv/cs336-basics", copy=True)
        .uv_sync()
        .add_local_python_source("cs336_systems")
    )

    if include_tests:
        image = image.add_local_dir("tests", remote_path="/root/tests")

    return image


image = build_image()


def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )


def ddp_timing_worker(
    rank: int,
    world_size: int,
    label: str,
    numel: int,
    out,
):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    try:
        x = torch.randn(numel, device=device, dtype=torch.float32)

        # Warmup
        for _ in range(NUM_WARM_UPS):
            dist.all_reduce(x, op=dist.ReduceOp.SUM)

        torch.cuda.synchronize(device)
        dist.barrier()
        torch.cuda.synchronize(device)

        # Timed section
        start = time.perf_counter()

        for _ in range(NUM_STEPS):
            dist.all_reduce(x, op=dist.ReduceOp.SUM)

        torch.cuda.synchronize(device)
        elapsed_s = time.perf_counter() - start

        # Distributed runtime is bottlenecked by the slowest rank.
        elapsed_tensor = torch.tensor([elapsed_s], device=device)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        max_elapsed_s = float(elapsed_tensor.item())

        if rank == 0:
            size_bytes = numel * 4
            mean_ms = 1000.0 * max_elapsed_s / NUM_STEPS

            out.append(
                {
                    "world_size": world_size,
                    "size": label,
                    "numel": numel,
                    "bytes": size_bytes,
                    "mean_ms": mean_ms,
                }
            )

    finally:
        dist.destroy_process_group()


def run_one_benchmark(world_size: int, label: str, numel: int) -> dict:
    with pymp.Manager() as manager:
        out = manager.list()

        mp.spawn(
            fn=ddp_timing_worker,
            args=(world_size, label, numel, out),
            nprocs=world_size,
            join=True,
        )
        return dict(out[0])


@app.function(gpu="B200:6", image=image, timeout=60 * 20)
def run_all_benchmarks():
    results = []

    for world_size in GPU_LIST:
        for label, numel in DATA_SIZES:
            print(f"Running world_size={world_size}, size={label}")
            result = run_one_benchmark(world_size, label, numel)
            print(result)
            results.append(result)

    return results

def save_tables(results):
    df = pd.DataFrame(results)

    size_order = [label for label, _ in DATA_SIZES]
    df["size"] = pd.Categorical(df["size"], categories=size_order, ordered=True)
    df = df.sort_values(["size", "world_size"])

    latency_tex_path = Path("ddp_allreduce_latency.tex")

    latency = df.pivot(index="size", columns="world_size", values="mean_ms")
    latency = latency.reindex(size_order)
    latency.columns = [f"{c} GPUs" for c in latency.columns]
    latency.index.name = "Tensor size"

    latency_tex = latency.to_latex(
        float_format=lambda x: f"{x:.3f}",
        caption="All Reduce Latency Benchmarking",
        label="tab:all_reduce_latency",
        escape=False,
    )

    latency_tex_path.write_text(latency_tex)

    return latency_tex_path, latency_tex


@app.local_entrypoint()
def modal_main():
    results = run_all_benchmarks.remote()

    latency_tex_path, latency_tex = save_tables(results)

    with open("ddp_latency_benchmarking.tex", "w") as f:
        f.write(latency_tex)