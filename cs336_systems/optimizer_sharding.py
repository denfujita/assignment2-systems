from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class OptimizerSharding(torch.optim.Optimizer):

    def __init__(self, params, optimizer_cls: type[Optimizer], **kwargs: Any) -> None:
        """Initializes the sharded state optimizer. params is a collection of parameters to be optimized (or parameter
        groups, in case the user wants to use different hyperparameters, such as learning rates, for
        different parts of the model); these parameters will be sharded across all the ranks. The
        optimizer_cls parameter specifies the type of optimizer to be wrapped (e.g., optim.AdamW).
        Finally, any remaining keyword arguments are forwarded to the constructor of the
        optimizer_cls. Make sure to call the torch.optim.Optimizer super-class constructor in this
        method."""
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self._next_param_index = 0
        self._param_to_owner: dict[torch.nn.Parameter, int] = {}
        self._ordered_params: list[torch.nn.Parameter] = []
        self._local_param_groups: list[dict[str, Any]] = []
        self._optimizer: Optimizer | None = None
        self._finished_init = False

        super().__init__(params, defaults=kwargs)

        if self._local_param_groups:
            self._optimizer = self.optimizer_cls(self._local_param_groups, **self.optimizer_kwargs)
            self.state = self._optimizer.state
        self._finished_init = True

    def step(self, closure=None, **kwargs):
        """Calls the wrapped optimizer’s step() method with the
        provided closure and keyword arguments. After updating the parameters, synchronize with the
        other ranks."""
        loss = None
        if self._optimizer is not None:
            loss = self._optimizer.step(closure=closure, **kwargs)
        elif closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.world_size > 1 and dist.is_initialized():
            for param in self._ordered_params:
                dist.broadcast(param.data, src=self._param_to_owner[param])
        return loss

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """This method should add a parameter
        group to the sharded optimizer. This is called during construction of the sharded optimizer by
        the super-class constructor and may also be called during training (e.g., for gradually
        unfreezing layers in a model). As a result, this method should handle assigning the model’s 
        parameters among the ranks."""
        super().add_param_group(param_group)

        global_group = self.param_groups[-1]
        local_params = []
        local_param_names = []
        param_names = global_group.get("param_names")

        for index, param in enumerate(global_group["params"]):
            if param not in self._param_to_owner:
                self._param_to_owner[param] = self._next_param_index % self.world_size
                self._ordered_params.append(param)
                self._next_param_index += 1

            owner = self._param_to_owner[param]
            if owner == self.rank:
                local_params.append(param)
                if param_names is not None:
                    local_param_names.append(param_names[index])

        if not local_params:
            return

        local_group = {key: value for key, value in global_group.items() if key not in {"params", "param_names"}}
        local_group["params"] = local_params
        if param_names is not None:
            local_group["param_names"] = local_param_names

        if self._optimizer is None:
            self._local_param_groups.append(local_group)
            if self._finished_init:
                self._optimizer = self.optimizer_cls(self._local_param_groups, **self.optimizer_kwargs)
                self.state = self._optimizer.state
        else:
            self._optimizer.add_param_group(local_group)
            self.state = self._optimizer.state

    @property
    def local_optimizer(self) -> Optimizer | None:
        return self._optimizer