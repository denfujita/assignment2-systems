import sys
import torch
import modal
import pandas as pd

BATCH_SIZE = 1
USE_CAUSAL = True
NUM_WARMUPS = 5
NUM_STEPS = 10

seq_len_params = [128 * 2 ** x for x in range(10)]
d_params = [16 * 2 ** x for x in range(4)]
dtype_params = [(torch.bfloat16, "bf16"), (torch.float32, "fp32")]

attention_inputs = []

for s in seq_len_params:
    for d in d_params:
        for dtype, dtype_name in dtype_params:
            attention_inputs.append((s, d, dtype, dtype_name))

app = modal.App(name="flash-benchmarking")

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

def _make_attn_inputs(seq_len: int, d: int, dtype, device=None, requires_grad: bool = True):
    torch.random.manual_seed(0)
    q = torch.randn(BATCH_SIZE, seq_len, d, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(BATCH_SIZE, seq_len, d, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(BATCH_SIZE, seq_len, d, device=device, dtype=dtype, requires_grad=requires_grad)
    return q, k, v

def _make_causal_mask(seq_len: int, device=None):
    if not USE_CAUSAL:
        return None

    i = torch.arange(seq_len, device=device)
    return i[:, None] >= i[None, :]

def benchmark(fn):
    import triton

    for _ in range(NUM_WARMUPS):
        fn()

    torch.cuda.synchronize()
    ms = triton.testing.do_bench(fn, warmup=0, rep=NUM_STEPS)
    torch.cuda.synchronize()

    return float(ms)

def safe_bench(name, make_fn):
    try:
        fn = make_fn()
        return benchmark(fn)
    except Exception as e:
        print(f"{name} failed: {e}")
        return None
    finally:
        torch.cuda.empty_cache()

@app.function(image=image, gpu="B200", timeout=60 * 60, max_containers=80)
def run(seq_len: int, d: int, dtype, dtype_name: str):
    sys.path.append("/root")

    from cs336_basics.model import scaled_dot_product_attention
    from cs336_systems.flash_attention import SpeedyAttention
    from cs336_systems.triton_attention import SpeedyTritonAttention

    device = "cuda"

    print(f"running seq_len={seq_len}, d={d}, dtype={dtype_name}")

    row = {"seq_len": seq_len, "d": d, "dtype": dtype_name}

    def make_regular_fwd():
        q, k, v = _make_attn_inputs(seq_len, d, dtype, device, requires_grad=False)
        mask = _make_causal_mask(seq_len, device)

        def regular_fwd():
            with torch.no_grad():
                scaled_dot_product_attention(q, k, v, mask)

        return regular_fwd

    row["regular_pytorch_fwd_ms"] = safe_bench("regular pytorch fwd", make_regular_fwd)

    def make_regular_bwd():
        q, k, v = _make_attn_inputs(seq_len, d, dtype, device, requires_grad=True)
        mask = _make_causal_mask(seq_len, device)
        out = scaled_dot_product_attention(q, k, v, mask)
        grad_output = torch.randn_like(out)

        def regular_bwd():
            q.grad = None
            k.grad = None
            v.grad = None
            out.backward(grad_output, retain_graph=True)

        return regular_bwd

    row["regular_pytorch_bwd_ms"] = safe_bench("regular pytorch bwd", make_regular_bwd)

    def make_regular_e2e():
        q, k, v = _make_attn_inputs(seq_len, d, dtype, device, requires_grad=True)
        mask = _make_causal_mask(seq_len, device)

        def regular_e2e():
            q.grad = None
            k.grad = None
            v.grad = None
            out = scaled_dot_product_attention(q, k, v, mask)
            out.backward(torch.ones_like(out))

        return regular_e2e

    row["regular_pytorch_e2e_ms"] = safe_bench("regular pytorch e2e", make_regular_e2e)

    def make_fa2_pytorch_fwd():
        q, k, v = _make_attn_inputs(seq_len, d, dtype, device, requires_grad=False)

        def fa2_pytorch_fwd():
            with torch.no_grad():
                SpeedyAttention.apply(q, k, v, USE_CAUSAL)

        return fa2_pytorch_fwd

    row["fa2_pytorch_fwd_ms"] = safe_bench("fa2 pytorch fwd", make_fa2_pytorch_fwd)

    def make_fa2_pytorch_bwd():
        q, k, v = _make_attn_inputs(seq_len, d, dtype, device, requires_grad=True)
        out = SpeedyAttention.apply(q, k, v, USE_CAUSAL)
        grad_output = torch.randn_like(out)

        def fa2_pytorch_bwd():
            q.grad = None
            k.grad = None
            v.grad = None
            out.backward(grad_output, retain_graph=True)

        return fa2_pytorch_bwd

    row["fa2_pytorch_bwd_ms"] = safe_bench("fa2 pytorch bwd", make_fa2_pytorch_bwd)

    def make_fa2_pytorch_e2e():
        q, k, v = _make_attn_inputs(seq_len, d, dtype, device, requires_grad=True)

        def fa2_pytorch_e2e():
            q.grad = None
            k.grad = None
            v.grad = None
            out = SpeedyAttention.apply(q, k, v, USE_CAUSAL)
            out.backward(torch.ones_like(out))

        return fa2_pytorch_e2e

    row["fa2_pytorch_e2e_ms"] = safe_bench("fa2 pytorch e2e", make_fa2_pytorch_e2e)

    def make_fa2_triton_fwd():
        q, k, v = _make_attn_inputs(seq_len, d, dtype, device, requires_grad=False)

        def fa2_triton_fwd():
            with torch.no_grad():
                SpeedyTritonAttention.apply(q, k, v, USE_CAUSAL)

        return fa2_triton_fwd

    row["fa2_triton_fwd_ms"] = safe_bench("fa2 triton fwd", make_fa2_triton_fwd)

    return row

@app.local_entrypoint()
def main():
    rows = list(run.starmap(attention_inputs, return_exceptions=True))

    results = []

    for inp, row in zip(attention_inputs, rows):
        if isinstance(row, Exception):
            seq_len, d, dtype, dtype_name = inp
            print(f"FAILED seq_len={seq_len}, d={d}, dtype={dtype_name}: {row}")

            row = {
                "seq_len": seq_len,
                "d": d,
                "dtype": dtype_name,
                "regular_pytorch_fwd_ms": None,
                "regular_pytorch_bwd_ms": None,
                "regular_pytorch_e2e_ms": None,
                "fa2_pytorch_fwd_ms": None,
                "fa2_pytorch_bwd_ms": None,
                "fa2_pytorch_e2e_ms": None,
                "fa2_triton_fwd_ms": None,
            }

        results.append(row)

    df = pd.DataFrame(results)

    latex_df = df.rename(columns={
        "seq_len": "Seq Len",
        "d": "D",
        "dtype": "Dtype",
        "regular_pytorch_fwd_ms": "Regular Fwd",
        "regular_pytorch_bwd_ms": "Regular Bwd",
        "regular_pytorch_e2e_ms": "Regular E2E",
        "fa2_pytorch_fwd_ms": "FA2 PyTorch Fwd",
        "fa2_pytorch_bwd_ms": "FA2 PyTorch Bwd",
        "fa2_pytorch_e2e_ms": "FA2 PyTorch E2E",
        "fa2_triton_fwd_ms": "FA2 Triton Fwd",
    })

    cols = ["Regular Fwd", "Regular Bwd", "Regular E2E", "FA2 PyTorch Fwd", "FA2 PyTorch Bwd", "FA2 PyTorch E2E", "FA2 Triton Fwd"]

    for col in cols:
        latex_df[col] = latex_df[col].map(lambda x: "--" if pd.isna(x) else f"{x:.3f}")

    latex = latex_df.to_latex(
        index=False,
        escape=False,
        column_format="rrlrrrrrrr",
        caption="FlashAttention-2 benchmark comparison",
        label="tab:flash_benchmark",
    )

    with open("flash_benchmark.tex", "w") as f:
        f.write(latex)