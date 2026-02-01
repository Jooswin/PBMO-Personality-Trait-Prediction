from __future__ import annotations

from math import ceil
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.model import Model
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor

from src.utils.get_preferential_gp_sample import get_preferential_gp_rff_sample


def random_search_optimizer(
    model: GenericDeterministicModel,
    bounds: Tensor,
    num_points: int,
    maximize: bool,
    pop_size: int = 1024,
    max_tries: int = 10,
) -> Tuple[Tensor, Tensor]:
    r"""Optimize a function via random search.

    Args:
        model: The model.
        bounds: A `2 x d`-dim Tensor containing the input bounds.
        num_points: The number of optimal points to be outputted.
        maximize: If true, we consider a maximization problem.
        pop_size: The number of function evaluations per try.
        max_tries: The maximum number of tries.

    Returns:
        A two-element tuple containing

        - A `num_points x d`-dim Tensor containing the collection of optimal inputs.
        - A `num_points x M`-dim Tensor containing the collection of optimal
            objectives.
    """
    tkwargs = {"dtype": bounds.dtype, "device": bounds.device}
    weight = 1.0 if maximize else -1.0
    optimal_inputs = torch.tensor([], **tkwargs)
    optimal_outputs = torch.tensor([], **tkwargs)
    num_tries = 0
    ratio = 2
    while ratio > 1 and num_tries < max_tries:
        X = draw_sobol_samples(bounds=bounds, n=pop_size, q=1).squeeze(-2)
        Y = model.posterior(X).mean
        X_aug = torch.cat([optimal_inputs, X], dim=0)
        Y_aug = torch.cat([optimal_outputs, Y], dim=0)
        pareto_mask = is_non_dominated(weight * Y_aug)
        optimal_inputs = X_aug[pareto_mask]
        optimal_outputs = Y_aug[pareto_mask]
        num_found = len(optimal_inputs)
        ratio = ceil(num_points / num_found)
        num_tries = num_tries + 1
    # If maximum number of retries exceeded throw out a runtime error.
    return optimal_inputs, optimal_outputs
    

def sample_optimal_points(
    model: Model,
    bounds: Tensor,
    num_samples: int,
    num_points: int,
    optimizer: Callable[
        [GenericDeterministicModel, Tensor, int, bool, Any], Tuple[Tensor, Tensor]
    ] = random_search_optimizer,
    num_rff_features: int = 512,
    maximize: bool = True,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Compute a collection of optimal inputs and outputs from samples of a Gaussian
    Process (GP).

    Steps:
    (1) The samples are generated using random Fourier features (RFFs).
    (2) The samples are optimized sequentially using an optimizer.

    TODO: We can generalize the GP sampling step to accommodate for other sampling
        strategies rather than restricting to RFFs e.g. decoupled sampling.

    TODO: Currently this defaults to random search optimization, might want to
        explore some other alternatives.

    Args:
        model: The model. This does not support models which include fantasy
            observations.
        bounds: A `2 x d`-dim Tensor containing the input bounds.
        num_samples: The number of GP samples.
        num_points: The number of optimal points to be outputted.
        optimizer: A callable that solves the deterministic optimization problem.
        num_rff_features: The number of random Fourier features.
        maximize: If true, we consider a maximization problem.
        optimizer_kwargs: The additional arguments for the optimizer.

    Returns:
        A two-element tuple containing

        - A `num_samples x num_points x d`-dim Tensor containing the collection of
            optimal inputs.
        - A `num_samples x num_points x M`-dim Tensor containing the collection of
            optimal objectives.
    """
    tkwargs = {"dtype": bounds.dtype, "device": bounds.device}
    M = model.num_outputs
    d = bounds.shape[-1]
    if M == 1:
        if num_points > 1:
            raise UnsupportedError(
                "For single-objective optimization `num_points` should be 1."
            )
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    pareto_sets = []
    pareto_fronts = []
    min_num_points = np.inf
    for i in range(num_samples):
        sample_i = get_preferential_gp_rff_sample(
            model=model, n_samples=1
        )
        ps_i, pf_i = optimizer(
            model=sample_i,
            bounds=bounds,
            num_points=num_points,
            maximize=maximize,
            **optimizer_kwargs,
        )
        min_num_points = min(min_num_points, len(ps_i))
        pareto_sets.append(ps_i.clone().unsqueeze(0))
        pareto_fronts.append(pf_i.clone().unsqueeze(0))
    pareto_sets = torch.cat([pareto_set[:, :min_num_points, :] for pareto_set in pareto_sets])
    pareto_fronts = torch.cat([pareto_front[:, :min_num_points, :] for pareto_front in pareto_fronts])
    return pareto_sets, pareto_fronts
