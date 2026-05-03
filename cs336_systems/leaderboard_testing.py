import modal
import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


class Config:
    ctx_len = 32768
    vocab_size = 151936
    d_model = 4096
    d_ff = 11008
    num_layers = 34
    num_heads = 32
    torch_dtype = torch.bfloat16
    is_causal = True
    batch_size = 2


cfg = Config()
app = modal.App("leaderboard-testing")


def build_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .run_commands(
            "apt-get update && apt-get install -y wget",
            "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
            "dpkg -i cuda-keyring_1.1-1_all.deb",
            "apt-get update",
        )
        .add_local_dir("cs336-basics", remote_path="/.uv/cs336-basics", copy=True)
        .uv_sync()
        .add_local_python_source("cs336_systems")
    )


image = build_image()


@app.function(image=image, gpu="B200:2", timeout=60 * 20)
def test_timing_forward_backward():
    import triton  # type: ignore[reportMissingImports]

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    labels, targets = torch.randint(
        high=cfg.vocab_size,
        size=(2, cfg.batch_size, cfg.ctx_len),
        device=device,
    )

    model = BasicsTransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.ctx_len,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
    )
    model = model.to(device=device, dtype=cfg.torch_dtype)
    optimizer = AdamW(model.parameters())

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        res = model(labels)
        loss = cross_entropy(res, targets).sum()
        loss.backward()
        optimizer.step()

    timing_results = triton.testing.do_bench(train_step, rep=30_000, warmup=10_000)
    torch.cuda.synchronize()
    print(timing_results)
    return timing_results


@app.local_entrypoint()
def main():
    result = test_timing_forward_backward.remote()
    print(result)
