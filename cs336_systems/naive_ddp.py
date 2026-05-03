from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self._pending_gradient_syncs = []
        self._handles = []
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

        for param in self.module.parameters():
            if dist.is_initialized():
                dist.broadcast(param.data, src=0)

            if param.requires_grad:
                self._handles.append(param.register_post_accumulate_grad_hook(self._start_gradient_sync))

    def _start_gradient_sync(self, param: torch.Tensor) -> None:
        if self._world_size == 1 or param.grad is None:
            return

        work = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self._pending_gradient_syncs.append((work, param.grad))

    def finish_gradient_synchronization(self) -> None:
        for work, grad in self._pending_gradient_syncs:
            work.wait()
            grad.div_(self._world_size)
        self._pending_gradient_syncs.clear()

    def sync_gradients(self) -> None:
        self.finish_gradient_synchronization()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
