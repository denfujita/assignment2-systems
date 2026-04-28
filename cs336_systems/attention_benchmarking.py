from cs336_basics.model import scaled_dot_product_attention
import modal
import torch
import pandas as pd

BATCH = 8
NUM_WARMUPS = 10
NUM_STEPS = 100

d_model_list = [16, 32, 64, 128]
seq_len_list = [256, 1024, 4096, 8192, 16384]

attention_inputs = []

for d in d_model_list: 
    for s in seq_len_list:
        attention_inputs.append((d, s))


app = modal.App(name="attention-benchmarking")

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
def pytorch_attention(d_model: int, seq_len: int): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Q = torch.randn(BATCH, seq_len, d_model, device=device)
    K = torch.randn(BATCH, seq_len, d_model, device=device) 
    V = torch.randn(BATCH, seq_len, d_model, device=device)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)  

    # warm up
    attn = torch.compile(scaled_dot_product_attention)
    for i in range(NUM_WARMUPS):
        v = attn(Q, K, V)
    torch.cuda.synchronize()

    forward_oom = False
    back_oom = False

    forward_time = 0
    try:
        start.record()
        for i in range(NUM_STEPS):
            v = attn(Q, K, V)
            torch.cuda.synchronize()
        end.record()
        torch.cuda.synchronize()
        forward_time = start.elapsed_time(end)
    except torch.cuda.OutOfMemoryError as e:
        forward_oom = True
        torch.cuda.empty_cache()

    Q = torch.randn(BATCH, seq_len, d_model, requires_grad=True, device=device)
    K = torch.randn(BATCH, seq_len, d_model, requires_grad = True, device=device) 
    V = torch.randn(BATCH, seq_len, d_model, requires_grad=True, device=device)

    total_back_mem = 0
    try: 
        def attention_loss(Q, K, V):
            return scaled_dot_product_attention(Q, K, V).sum()
        compiled_loss = torch.compile(attention_loss)

        for i in range(NUM_WARMUPS):
            output = compiled_loss(Q, K, V)
            output.backward()

        Q.grad = K.grad = V.grad = None
        torch.cuda.synchronize()

        total_backward_ms = 0.0
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)

        for i in range(NUM_STEPS):
            loss = compiled_loss(Q, K, V)
            total_back_mem += torch.cuda.memory_allocated()
            torch.cuda.synchronize()
            bwd_start.record()
            loss.backward()
            bwd_end.record()
            torch.cuda.synchronize()
            total_backward_ms += bwd_start.elapsed_time(bwd_end)
        total_back_mem /= NUM_STEPS
    except torch.cuda.OutOfMemoryError as e:
        back_oom = True

    return (forward_time / NUM_STEPS, total_back_mem / 1024**2, total_backward_ms / NUM_STEPS, forward_oom, back_oom)

@app.local_entrypoint()
def modal_main():
    benchmark_results = pytorch_attention.starmap(attention_inputs, return_exceptions=True)
    rows = []
    for (d_model, seq_len), result in zip(attention_inputs, benchmark_results):
        if isinstance(result, Exception): 
            continue
        else: 
            forward_ms, memory_mib, backward_ms, forward_oom, back_oom = result
            rows.append({
                "d_model": d_model,
                "seq_len": seq_len,
                "forward_ms": "oom" if forward_oom else forward_ms,
                "memory_before_backward_MiB": "oom" if back_oom or forward_oom else memory_mib,
                "backward_ms": "oom" if back_oom else backward_ms,
            })

    df = pd.DataFrame(rows).sort_values(["d_model", "seq_len"])

    latex_str = df.to_latex(
        index=False,
        escape=False,
        float_format="%.3f",
        column_format="rrrrr",
    )

    # save locally
    with open("attention_benchmark_table.tex", "w") as f:
        f.write(latex_str)
        




