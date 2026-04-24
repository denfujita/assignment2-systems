import timeit
from pathlib import Path

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
import torch
import numpy as np
import modal


BATCH_SIZE = 4

app = modal.App(name="gpu-benchmarking")

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

@app.function(image=image, gpu="B200")
def benchmarking_script(num_warmups: int, num_steps: int, hyperparams: dict, mode: str):
    """
    Initialize a BasicsTransformerLM, create random data, and benchmark:
    - forward
    - forward + backward
    - forward + backward + optimizer
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)

    # initialize LM
    model = BasicsTransformerLM(
        hyperparams["vocab_size"],
        hyperparams["context_length"],
        hyperparams["d_model"],
        hyperparams["num_layers"],
        hyperparams["num_heads"],
        hyperparams["d_ff"],
        hyperparams["rope_theta"],
    ).to(device)

    # initialize train and target data
    data = torch.randint(
        0,
        hyperparams["vocab_size"],
        (BATCH_SIZE, hyperparams["context_length"]),
        device=device,
    )
    targets = torch.randint(
        0,
        hyperparams["vocab_size"],
        (BATCH_SIZE, hyperparams["context_length"]),
        device=device,
    )

    optimizer = AdamW(model.parameters()) if mode == "forward-back-optimizer" else None

    def run_step():
        if mode == "forward":
            with torch.no_grad():
                return model(data)

        output = model(data)
        loss = cross_entropy(output, targets)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        else:
            loss.backward()
            model.zero_grad(set_to_none=True)

        return loss

    # warm up
    for _ in range(num_warmups):
        run_step()

    torch.cuda.synchronize()

    # measure
    times = []
    for _ in range(num_steps):
        start = timeit.default_timer()
        run_step()
        torch.cuda.synchronize()
        end = timeit.default_timer()

        times.append(end - start)

    times = np.array(times)
    return np.mean(times), np.std(times)


def generate_latex_table(setup, result) -> str:
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

    Path("benchmarking_one_warmups_results.tex").write_text(latex)

@app.local_entrypoint()
def modal_main() -> None:

    VOCAB_SIZE = 10_000
    CONTEXT_LENGTH = 512
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

    eval_params = []
    setup = []
    
    for model_type, hyperparam in [("small", small_hyperparam), ("med", medium_hyperparam), ("large", large_hyperparam), ("xl", xl_hyperparam), ("10b", ten_b_hyperparam)]:
        for mode in ["forward", "forward-back", "forward-back-optimizer"]:
            eval_params.append((1, 10, hyperparam, mode))
            setup.append((model_type, mode))

    result = list(benchmarking_script.starmap(eval_params, return_exceptions=True))
    latex = generate_latex_table(setup, result)
    return latex