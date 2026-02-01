from __future__ import annotations

from typing import Callable, Optional

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError, UnsupportedError
from botorch.utils.transforms import normalize
from torch import Tensor


def get_hypervolume_scalarization(weights: Tensor, Y: Tensor) -> Callable[[Tensor, Optional[Tensor]], Tensor]:
    r"""
    """
    if weights.shape != Y.shape[-1:]:
        raise BotorchTensorDimensionError(
            "weights must be an `m`-dim tensor where Y is `... x m`."
            f"Got shapes {weights.shape} and {Y.shape}."
        )
    elif Y.ndim > 2:
        raise NotImplementedError("Batched Y is not currently supported.")

    def hypervolume_obj(Y: Tensor, X: Optional[Tensor] = None) -> Tensor:
        Y_w =  Y / weights
        Y_w_plus = Y_w.clamp_min(0.0)
        obj_val = (Y_w_plus.min(dim=-1)) ** Y.shape[-1]
        return obj_val

    # A boolean mask indicating if minimizing an objective
    minimize = weights < 0
    if minimize.any():
        raise UnsupportedError(
            "Negative weights are not supported."
        )

    if Y.shape[-2] == 1:
        # If there is only one observation, set the bounds to be
        # [min(Y_m), min(Y_m) + 1] for each objective m. This ensures we do not
        # divide by zero
        Y_bounds = torch.cat([Y, Y + 1], dim=0)
    else:
        # Set the bounds to be [min(Y_m), max(Y_m)], for each objective m
        Y_bounds = torch.stack([Y.min(dim=-2).values, Y.max(dim=-2).values])

    def obj(Y: Tensor, X: Optional[Tensor] = None) -> Tensor:
        # scale to [0,1]
        Y_normalized = normalize(Y, bounds=Y_bounds)
        # If minimizing an objective, convert Y_normalized values to [-1,0],
        # such that min(w*y) makes sense, we want all w*y's to be positive
        Y_normalized[..., minimize] = Y_normalized[..., minimize] - 1
        # multiply the scalarization by -1, so that the scalarization should
        # be maximized
        return hypervolume_obj(Y=Y_normalized)

    return obj