# CS336 Assignment 2 Systems TODO

Status: not started

Use this file to track progress through Assignment 2. The handout is `cs336_assignment2_systems.pdf`; implementation hooks are in `tests/adapters.py`.

## Ground Rules

- Keep implementation code in `cs336_systems/` unless there is a clear reason to add scripts elsewhere.
- Use `tests/adapters.py` to connect your implementations to the provided tests.
- Do not modify the provided tests as a substitute for passing them.
- Keep written answers and profiling results organized for `writeup.pdf`.
- Generate final `code.zip` with `./test_and_make_submission.sh`.

## 0. Setup

- [ ] Confirm `uv run python` can import `cs336_basics`.
- [ ] Decide whether to use staff `cs336-basics` or your Assignment 1 implementation.
- [ ] Skim `cs336-basics/cs336_basics/model.py` to understand the reference Transformer API.
- [ ] Skim `tests/adapters.py` to understand all required adapter functions.
- [ ] Create a place to store benchmark outputs, tables, screenshots, and notes.

## 1. Profiling And Benchmarking

### Benchmarking Script

- [ ] Write an end-to-end benchmark script.
- [ ] Support model hyperparameters from the handout table.
- [ ] Generate random batches.
- [ ] Support forward-only timing.
- [ ] Support forward + backward timing.
- [ ] Support full training-step timing with optimizer step.
- [ ] Add warmup-step control.
- [ ] Add measured-step control.
- [ ] Synchronize CUDA after each timed step.
- [ ] Record average and standard deviation.

### Required Runs

- [ ] Benchmark small model.
- [ ] Benchmark medium model.
- [ ] Benchmark large model.
- [ ] Benchmark xl model.
- [ ] Benchmark 10B model if resources allow.
- [ ] Repeat without warmup.
- [ ] Repeat with 1 or 2 warmup steps.
- [ ] Write 1-2 sentence timing summary.
- [ ] Write 2-3 sentence warmup explanation.

### Nsight Systems

- [ ] Add NVTX ranges to separate warmup, forward, backward, and optimizer step.
- [ ] Profile two model sizes.
- [ ] Profile three power-of-two context lengths larger than 128.
- [ ] Identify total forward-pass time.
- [ ] Compare Nsight timing with Python timing.
- [ ] Identify top cumulative CUDA kernel in forward.
- [ ] Count top-kernel invocations in one forward pass.
- [ ] Compare top kernel for forward-only vs forward+backward.
- [ ] Identify non-matmul kernels with meaningful runtime.
- [ ] Profile one full training step.
- [ ] Compare fraction of time in matmul for inference vs training.
- [ ] Compare softmax runtime vs attention matmul runtime.
- [ ] Save screenshots/tables for writeup.

## 2. Mixed Precision

- [ ] Run the accumulation experiment from the handout.
- [ ] Write 2-3 sentence accuracy comment.
- [ ] Answer ToyModel autocast dtype questions.
- [ ] Explain why layer norm is precision-sensitive.
- [ ] Explain FP16 vs BF16 implications for layer norm.
- [ ] Add optional BF16 autocast to benchmark script.
- [ ] Benchmark full precision vs BF16 for each model size.
- [ ] Write timing/commentary response.

## 3. Memory Profiling

- [ ] Add PyTorch CUDA memory snapshot option to benchmark script.
- [ ] Profile xl model with context length 128.
- [ ] Profile xl model with context length 2048.
- [ ] Capture active memory timeline for forward-only.
- [ ] Capture active memory timeline for full training step.
- [ ] Record peak memory for each context length and mode.
- [ ] Repeat peak memory measurements with mixed precision.
- [ ] Derive residual-stream activation tensor size for xl config.
- [ ] Inspect largest allocations in memory_viz.
- [ ] Use Nsight memory profiling to inspect saved tensors per TransformerBlock.
- [ ] List five largest contributing saved-tensor operations.
- [ ] Estimate gradient tensor memory for one TransformerBlock.
- [ ] Add screenshots and written responses to writeup notes.

## 4. Activation Checkpointing

- [ ] Understand PyTorch saved tensor hooks example.
- [ ] Explain memory-optimal checkpointing strategy asymptotically.
- [ ] Include a short non-solution code sketch in writeup if allowed by course policy.
- [ ] For xl model, test one-level checkpoint block sizes.
- [ ] Measure peak memory for chosen checkpointing strategy.
- [ ] Measure next smaller checkpoint block size.
- [ ] Measure next larger checkpoint block size.
- [ ] Write reasoning and measured peak memory.

## 5. Attention And FlashAttention

### PyTorch Attention Benchmark

- [ ] Write attention benchmark script.
- [ ] Use batch size 8.
- [ ] Remove multihead dimension.
- [ ] Sweep head dimension: 16, 32, 64, 128.
- [ ] Sweep sequence length: 256, 1024, 4096, 8192, 16384.
- [ ] Time 100 forward passes.
- [ ] Measure memory before backward.
- [ ] Time 100 backward passes.
- [ ] Record OOM cases.
- [ ] Account for memory usage in one OOM-scale configuration.
- [ ] Explain how saved backward memory scales with sequence length.

### Torch Compile

- [ ] Add compiled attention benchmark.
- [ ] Compare compiled vs uncompiled attention.
- [ ] Compile full Transformer in end-to-end benchmark.
- [ ] Compare vanilla vs compiled Transformer timing.

### FlashAttention-2 Implementation

- [ ] Implement pure PyTorch `torch.autograd.Function` FlashAttention-2 forward.
- [ ] Save exactly the tensors required by tests, including logsumexp tensor.
- [ ] Connect `get_flashattention_autograd_function_pytorch`.
- [ ] Run `uv run pytest -k test_flash_forward_pass_pytorch`.
- [ ] Implement Triton FlashAttention-2 forward kernel.
- [ ] Connect `get_flashattention_autograd_function_triton`.
- [ ] Run `uv run pytest -k test_flash_forward_pass_triton`.
- [ ] Add causal masking support to Triton forward.
- [ ] Save causal flag for backward.
- [ ] Implement backward using PyTorch and `torch.compile`.
- [ ] Run `uv run pytest -k test_flash_backward`.
- [ ] Benchmark FlashAttention vs regular PyTorch attention with `triton.testing.do_bench`.
- [ ] Sweep sequence length 128 through 65536 where possible.
- [ ] Sweep embedding dimension 16 through 128.
- [ ] Test BF16 and FP32.
- [ ] Report forward, backward, and end-to-end latencies.

## 6. Distributed Data Parallel

### Communication Benchmark

- [ ] Write single-node distributed all-reduce benchmark.
- [ ] Test tensor sizes: 1MB, 10MB, 100MB, 1GB.
- [ ] Test 2 GPUs/processes.
- [ ] Test 4 GPUs/processes if available.
- [ ] Test 6 GPUs/processes if available.
- [ ] Plot or tabulate results.
- [ ] Write 2-3 sentence commentary.

### Naive DDP

- [ ] Implement naive DDP script.
- [ ] Broadcast parameters from rank 0.
- [ ] All-reduce individual parameter gradients after backward.
- [ ] Verify toy model matches single-process training.
- [ ] Benchmark xl model on 1 node x 2 GPUs.
- [ ] Measure total training-step time.
- [ ] Measure gradient communication time.

### Improved DDP

- [ ] Implement flat-gradient communication benchmark.
- [ ] Compare flat all-reduce vs individual all-reduces.
- [ ] Implement overlapping DDP wrapper with post-accumulate grad hooks.
- [ ] Broadcast initial weights.
- [ ] Launch async per-parameter gradient all-reduces.
- [ ] Wait before optimizer step.
- [ ] Connect `get_ddp`.
- [ ] Connect `ddp_on_after_backward`.
- [ ] Run `uv run pytest tests/test_ddp.py`.
- [ ] Benchmark overlapping DDP.
- [ ] Capture Nsight screenshots showing communication overlap.

## 7. Optimizer State Sharding

- [ ] Implement sharded optimizer wrapper.
- [ ] Support arbitrary optimizer class.
- [ ] Shard parameters across ranks.
- [ ] Keep optimizer state only for local shard.
- [ ] Synchronize updated parameters after optimizer step.
- [ ] Support `add_param_group`.
- [ ] Connect `get_sharded_optimizer`.
- [ ] Run `uv run pytest tests/test_sharded_optimizer.py`.
- [ ] Repeat test several times for reliability.
- [ ] Profile memory with and without optimizer state sharding.
- [ ] Record memory after model initialization.
- [ ] Record memory before optimizer step.
- [ ] Record memory after optimizer step.
- [ ] Benchmark speed with and without optimizer state sharding.
- [ ] Compare implementation to ZeRO stage 1 in writeup.

## 8. Fully Sharded Data Parallel

- [ ] Implement FSDP wrapper.
- [ ] Wrap or hook `cs336_basics.model.Linear`.
- [ ] Wrap or hook `cs336_basics.model.Embedding`.
- [ ] Do not shard small layers such as norms unless justified.
- [ ] Shard weights across ranks.
- [ ] All-gather weights for forward.
- [ ] Free gathered weights after use.
- [ ] All-gather weights for backward.
- [ ] Reduce-scatter gradients.
- [ ] Keep master weights and optimizer updates in FP32.
- [ ] Support optional `compute_dtype` for communication and compute.
- [ ] Implement finish/wait method for gradient synchronization.
- [ ] Implement full-parameter gathering for validation.
- [ ] Connect `get_fsdp`.
- [ ] Connect `fsdp_on_after_backward`.
- [ ] Connect `fsdp_gather_full_params`.
- [ ] Run `uv run pytest tests/test_fsdp.py`.
- [ ] Repeat test several times for reliability.
- [ ] Estimate expected peak memory savings.
- [ ] Profile xl model on two GPUs.
- [ ] Check whether all-gather finishes before forward needs weights.
- [ ] Save Nsight screenshots for writeup.

## 9. Written Parallelism Analysis

- [ ] Answer alternate ring all-reduce runtime.
- [ ] Answer DP backward FLOPs.
- [ ] Answer DP backward communication time.
- [ ] Answer DP communication-bottleneck inequality.
- [ ] Answer FSDP forward/backward FLOPs.
- [ ] Answer FSDP forward/backward communication time.
- [ ] Answer FSDP bottleneck inequalities.
- [ ] Write TP backward-pass equations.
- [ ] Answer TP forward/backward FLOPs.
- [ ] Answer TP forward/backward communication time.
- [ ] Answer TP bottleneck inequalities.
- [ ] Answer 2D FSDP+TP forward FLOPs.
- [ ] Answer 2D FSDP+TP communication time with overlap.
- [ ] Answer optimal total-device inequality with overlap.
- [ ] Answer optimal total-device inequality without overlap.

## 10. Leaderboard

- [ ] Decide whether to attempt leaderboard.
- [ ] Benchmark baseline full training step for 8B config.
- [ ] Identify memory bottlenecks.
- [ ] Identify runtime bottlenecks.
- [ ] Optimize only after correctness tests pass.
- [ ] Record best wall-clock full training-step time.
- [ ] Submit result to leaderboard if attempting.

## Final Submission

- [ ] All required tests pass locally.
- [ ] Benchmark/profiling data copied into writeup notes.
- [ ] Screenshots included in `writeup.pdf`.
- [ ] Written answers typeset.
- [ ] Run `./test_and_make_submission.sh`.
- [ ] Confirm `code.zip` is created.
- [ ] Submit `writeup.pdf`.
- [ ] Submit `code.zip`.

