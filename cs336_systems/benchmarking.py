import timeit
from pathlib import Path

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

from contextlib import nullcontext
import torch
import numpy as np
import modal
import subprocess
import argparse
import torch.cuda.nvtx as nvtx
import os

BATCH_SIZE = 4

VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 128
ROPE_THETA = 10_000.0

small_hyperparam = {
    "vocab_size": VOCAB_SIZE,
    "context_length": CONTEXT_LENGTH,
    "d_model": 768,
    "num_layers": 12,
    "num_heads": 12,
    "d_ff": 3072,
    "rope_theta": ROPE_THETA,
}

medium_hyperparam = {
    "vocab_size": VOCAB_SIZE,
    "context_length": CONTEXT_LENGTH,
    "d_model": 1024,
    "num_layers": 24,
    "num_heads": 16,
    "d_ff": 4096,
    "rope_theta": ROPE_THETA,
}

large_hyperparam = {
    "vocab_size": VOCAB_SIZE,
    "context_length": CONTEXT_LENGTH,
    "d_model": 1280,
    "num_layers": 36,
    "num_heads": 20,
    "d_ff": 5120,
    "rope_theta": ROPE_THETA,
}

xl_hyperparam = {
    "vocab_size": VOCAB_SIZE,
    "context_length": CONTEXT_LENGTH,
    "d_model": 2560,
    "num_layers": 32,
    "num_heads": 32,
    "d_ff": 10240,
    "rope_theta": ROPE_THETA,
}

ten_b_hyperparam = {
    "vocab_size": VOCAB_SIZE,
    "context_length": CONTEXT_LENGTH,
    "d_model": 4608,
    "num_layers": 50,
    "num_heads": 36,
    "d_ff": 12288,
    "rope_theta": ROPE_THETA,
}

app = modal.App(name="gpu-benchmarking")
# vol = modal.Volume.from_name(f"xl-forward-back-optimizer_{CONTEXT_LENGTH}", create_if_missing=True)

# vol = modal.Volume.from_name(f"xl_memory_profile_{CONTEXT_LENGTH}_mixed_precision1", create_if_missing=True)

vol = modal.Volume.from_name(f"xl_forward_backward_nsys1", create_if_missing=True)


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

@app.function(image=image, gpu="B200", volumes={"/profiles": vol})
def benchmarking_script(num_warmups: int, num_steps: int, hyperparams: dict, mode: str, mixed_precision: bool, memory_measurement: bool):
    """
    Initialize a BasicsTransformerLM, create random data, and benchmark:
    - forward
    - forward + backward
    - forward + backward + optimizer
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize model
    torch.manual_seed(0)
    model = BasicsTransformerLM(hyperparams["vocab_size"], 
                                hyperparams["context_length"], 
                                hyperparams["d_model"],
                                hyperparams["num_layers"],
                                hyperparams["num_heads"],
                                hyperparams["d_ff"],
                                hyperparams["rope_theta"])
    model = model.to(device)

    # generate random batch of data
    data = torch.randint(0, hyperparams["vocab_size"], (BATCH_SIZE, hyperparams["context_length"]), device=device)
    targets = torch.randint(0, hyperparams["vocab_size"], (BATCH_SIZE, hyperparams["context_length"]), device=device)

    times = [] 
    if mode == "forward":
        with torch.no_grad(): # forward only, no need to store gradient mappings
            cast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if mixed_precision else nullcontext() # optionally used mixed precision 
            with cast:
                for i in range(num_warmups): # warm up
                    model(data)
                torch.cuda.synchronize() # synch cuda before starting real timer
                if memory_measurement: # measures memory 
                    torch.cuda.memory._record_memory_history(max_entries=1000000)
                for i in range(num_steps): 
                    start = timeit.default_timer()
                    model(data)
                    torch.cuda.synchronize()
                    end = timeit.default_timer()
                    times.append(end - start)
                times = np.array(times)
                if memory_measurement:
                    torch.cuda.memory._dump_snapshot(f"/profiles/memory_snapshot_{mode}_mixed_precision:{mixed_precision}_{CONTEXT_LENGTH}.pickle")
                    torch.cuda.memory._record_memory_history(enabled=None)
                    vol.commit()
                return np.mean(times), np.std(times)
    elif mode == "forward-back":
            # forward-back warmup
        for i in range(num_warmups):
            model.zero_grad(set_to_none=True)
            cast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if mixed_precision else nullcontext()
            with cast:
                output = model(data)    
                loss = cross_entropy(output, targets)
            loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        for i in range(num_steps):
            start = timeit.default_timer()
            cast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if mixed_precision else nullcontext()
            with cast:
                output = model(data)
                loss = cross_entropy(output, targets)
            loss.backward()
            torch.cuda.synchronize()
            end = timeit.default_timer()
            times.append(end - start)
            model.zero_grad(set_to_none=True)
        times = np.array(times)
        return np.mean(times), np.std(times)
    else: # forward backward and optimizer 
# optimizer warmup
        optimizer = AdamW(model.parameters())
        for i in range(num_warmups):
            optimizer.zero_grad(set_to_none=True)
            cast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if mixed_precision else nullcontext() # optionally used mixed precision 
            with cast:
                output = model(data)
                loss = cross_entropy(output, targets)
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
# optimizer measured
        nvtx.range_push("forward-back-optimizer")
        if memory_measurement:
            torch.cuda.memory._record_memory_history(max_entries=1000000)
        for i in range(num_steps):
            start = timeit.default_timer()
            optimizer.zero_grad(set_to_none=True)
            cast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if mixed_precision else nullcontext() # optionally used mixed precision 
            with cast:
                output = model(data)
                loss = cross_entropy(output, targets)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            end = timeit.default_timer()
            times.append(end - start)
        if memory_measurement:
            torch.cuda.memory._dump_snapshot("/profiles/memory_snapshot_forwardbackoptimizer_mixed_precision.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)
            vol.commit()
        nvtx.range_pop()
        times = np.array(times)
        return np.mean(times), np.std(times)

def generate_latex_table(setup, result) -> None:
    import pandas as pd

    formatted_result = []
    for item in result:
        if isinstance(item, tuple):
            mean, std = item
            formatted_result.append(f"{mean:.4f} ± {std:.4f}")
        elif isinstance(item, Exception):
            formatted_result.append("OOM")
        else:
            formatted_result.append(str(item))

    df = pd.DataFrame(setup, columns=["Size", "Mode"])
    df["Result"] = formatted_result

    table = df.pivot(index="Size", columns="Mode", values="Result").reset_index()

    table["Size"] = pd.Categorical(
        table["Size"],
        categories=["small", "med", "large", "xl", "10b"],
        ordered=True,
    )
    table = table.sort_values("Size")

    latex = table.to_latex(
        index=False,
        caption="Benchmark results across model sizes.",
        label="tab:model-benchmark-results",
    )

    Path("benchmarking_mixed_precision.tex").write_text(latex)

@app.function(
    image=image,
    gpu="B200",
    volumes={"/profiles": vol},
    timeout=60 * 30,
)
def profile_on_modal(mode: str):
    out = f"/profiles/benchmark_xl_{mode}_{CONTEXT_LENGTH}"

    cmd = [
        "nsys", "profile",
        "--cuda-memory-usage=true",
        "--trace=cuda,cudnn,cublas,osrt,nvtx",
        "--pytorch=functions-trace,autograd-shapes-nvtx",
        # "--capture-range=nvtx",
        # "--nvtx-capture=measure",
        # "--cudabacktrace=all",
        # "--python-backtrace=cuda",
        "--force-overwrite=true",
        "-o", out,
        "python", __file__,
        "--worker",
        "--mode", mode,
        "--num-warmups", "5",
        "--num-steps", "10",
    ]

    env = os.environ.copy()
    # env["PYTORCH_ALLOC_CONF"] = "backend:cudaMallocAsync"

    # subprocess.run(cmd, env=env, check=True)
    subprocess.run(cmd, check=True)

    vol.commit()
    return f"{out}.nsys-rep"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=["forward", "forward-back", "forward-back-optimizer"])
    parser.add_argument("--num-warmups", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=1)
    args = parser.parse_args()

    if args.worker:
        hyperparams = xl_hyperparam

        mean, std = benchmarking_script.local(
            args.num_warmups,
            args.num_steps,
            hyperparams,
            args.mode,
            False,
            False,
        )

        print(f"{args.mode}: {mean:.4f} ± {std:.4f}")

@app.local_entrypoint()
def modal_main() -> None:
    # used in nsys question
    report_path = profile_on_modal.remote("forward-back")
    print(report_path)

    # used in benchmarking question

    eval_params = []
    setup = []
    
    # for model_type, hyperparam in [("small", small_hyperparam), ("med", medium_hyperparam), ("large", large_hyperparam), ("xl", xl_hyperparam), ("10b", ten_b_hyperparam)]:
    #     for mode in ["forward", "forward-back"]:
    #         eval_params.append((5, 10, hyperparam, mode, True, True))
    #         setup.append((model_type, mode))

    # memory profile xl
    # for model_type, hyperparam in [("xl", xl_hyperparam)]:
    #     for mode in ["forward", "forward-back-optimizer"]:
    #         eval_params.append((5, 10, hyperparam, mode, True, True))
    #         setup.append((model_type, mode))

    # result = list(benchmarking_script.starmap(eval_params, return_exceptions=True))
    # latex = generate_latex_table(setup, result)
    # return latex