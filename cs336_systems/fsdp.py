from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class FSDP(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._sharded_params: dict[str, nn.Parameter] = {}
        self._sharded_param_modules: dict[str, nn.Module] = {}

        from cs336_basics.model import Embedding as CS336Embedding
        from cs336_basics.model import Linear as CS336Linear

        for name, child in self.module.named_modules():
            if isinstance(child, (CS336Linear, CS336Embedding, nn.Linear, nn.Embedding)):
                weight = child.weight
                full_shape = tuple(weight.shape)
                rows_per_rank = (full_shape[0] + self.world_size - 1) // self.world_size
                padded_rows = rows_per_rank * self.world_size

                padded = weight.data.new_zeros((padded_rows, *full_shape[1:]))
                padded[: full_shape[0]].copy_(weight.data)
                start = self.rank * rows_per_rank
                end = start + rows_per_rank

                child._fsdp_full_shape = full_shape
                child._fsdp_rows_per_rank = rows_per_rank
                child.weight.data = padded[start:end].clone()
                param_name = f"{name}.weight" if name else "weight"
                self._sharded_params[param_name] = child.weight
                self._sharded_param_modules[param_name] = child

                def forward_pre_hook(m, inputs):
                    m._fsdp_local_weight = m.weight.data
                    if self.world_size > 1 and dist.is_initialized():
                        gathered = [torch.zeros_like(m.weight.data) for _ in range(self.world_size)]
                        dist.all_gather(gathered, m.weight.data.contiguous())
                        full = torch.cat(gathered, dim=0)[: m._fsdp_full_shape[0]].reshape(m._fsdp_full_shape)
                        m.weight.data = full
                    if self.compute_dtype is not None:
                        m.weight.data = m.weight.data.to(self.compute_dtype)

                def forward_hook(m, inputs, output):
                    m.weight.data = m._fsdp_local_weight
                    del m._fsdp_local_weight
                    m.weight.grad = None

                def backward_pre_hook(m, grad_output):
                    m._fsdp_local_weight_bwd = m.weight.data
                    if self.world_size > 1 and dist.is_initialized():
                        gathered = [torch.zeros_like(m.weight.data) for _ in range(self.world_size)]
                        dist.all_gather(gathered, m.weight.data.contiguous())
                        full = torch.cat(gathered, dim=0)[: m._fsdp_full_shape[0]].reshape(m._fsdp_full_shape)
                        m.weight.data = full
                    if self.compute_dtype is not None:
                        m.weight.data = m.weight.data.to(self.compute_dtype)
                    m.weight.grad = None

                def grad_hook(param, m=child):
                    if hasattr(m, "_fsdp_local_weight_bwd"):
                        param.data = m._fsdp_local_weight_bwd
                        del m._fsdp_local_weight_bwd
                    if param.grad is None:
                        return

                    grad = param.grad.to(torch.float32)
                    if self.world_size > 1 and dist.is_initialized():
                        if tuple(grad.shape) == tuple(m._fsdp_full_shape):
                            padded = grad.new_zeros((m._fsdp_rows_per_rank * self.world_size, *m._fsdp_full_shape[1:]))
                            padded[: m._fsdp_full_shape[0]].copy_(grad)
                            dist.all_reduce(padded, op=dist.ReduceOp.SUM)
                            padded.div_(self.world_size)
                            start = self.rank * m._fsdp_rows_per_rank
                            end = start + m._fsdp_rows_per_rank
                            param.grad = padded[start:end].contiguous()
                        else:
                            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
                            param.grad = (grad / self.world_size).contiguous()
                    else:
                        param.grad = grad.contiguous()

                child.register_forward_pre_hook(forward_pre_hook)
                child.register_forward_hook(forward_hook)
                if isinstance(child, (CS336Linear, nn.Linear)):
                    child.register_full_backward_pre_hook(backward_pre_hook)
                child.weight.register_post_accumulate_grad_hook(grad_hook)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        if self.world_size == 1 or not dist.is_initialized():
            return

        for name, param in self.module.named_parameters():
            if param.grad is None or name in self._sharded_params:
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(self.world_size)

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        full_params = {}
        for name, param in self.module.named_parameters():
            if name not in self._sharded_params or self.world_size == 1 or not dist.is_initialized():
                full_params[name] = param.data.detach().clone()
                continue

            module = self._sharded_param_modules[name]
            gathered = [torch.zeros_like(param.data) for _ in range(self.world_size)]
            dist.all_gather(gathered, param.data.contiguous())
            full = torch.cat(gathered, dim=0)[: module._fsdp_full_shape[0]]
            full_params[name] = full.reshape(module._fsdp_full_shape).detach().clone()
        return full_params