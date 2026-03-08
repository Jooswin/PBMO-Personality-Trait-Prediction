#!/usr/bin/env python3

from __future__ import annotations

import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.monte_carlo import qSimpleRegret
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex

from src.acquisition_functions.get_hypervolume_scalarization import get_hypervolume_scalarization
from src.acquisition_functions.scalarized_posterior_mean import ScalarizedPosteriorMean
from src.utils.get_preferential_gp_sample import get_preferential_gp_rff_sample
from src.utils.utils import optimize_acqf_and_get_suggested_query


# generates a (scalarized) dueling thompson sampling query
def gen_dueling_thompson_sampling_query(
    model,
    num_alternatives,
    bounds,
    num_restarts,
    raw_samples,
    scalarize=True,
    fix_scalarization=True,
    scalarization="chebyshev",
):
    query = []
    # this scalarizes a multi-output sample (required by SDTS)
    if scalarize:
        mean_train_inputs = model.posterior(model.train_inputs[0][0]).mean.detach()
        if fix_scalarization:
            if scalarization == "chebyshev":
                weights = sample_simplex(mean_train_inputs.shape[-1]).squeeze()
                scalarized_obj = GenericMCObjective(
                    get_chebyshev_scalarization(weights=weights, Y=mean_train_inputs)
                )
            elif scalarization == "hypervolume":
                weights = torch.normal(mean=0.0, std=1.0, shape=mean_train_inputs.shape[-1])
                weights = torch.abs(weights / torch.norm(weights, dim=-1))
                scalarized_obj = GenericMCObjective(
                    get_hypervolume_scalarization(weights=weights, Y=mean_train_inputs)
                )
    # generate each alternative in the query sequentially (this could be parallelized)
    for _ in range(num_alternatives):
        model_rff_sample = get_preferential_gp_rff_sample(model=model, n_samples=1)
        if scalarize:
            if not fix_scalarization:
                if scalarization == "chebyshev":
                    weights = sample_simplex(mean_train_inputs.shape[-1]).squeeze()
                    scalarized_obj = GenericMCObjective(
                        get_chebyshev_scalarization(weights=weights, Y=mean_train_inputs)
                    )
                elif scalarization == "hypervolume":
                    weights = torch.normal(mean=0.0, std=1.0, shape=mean_train_inputs.shape[-1])
                    weights = torch.abs(weights / torch.norm(weights, dim=-1))
                    scalarized_obj = GenericMCObjective(
                        get_hypervolume_scalarization(weights=weights, Y=mean_train_inputs)
                    )
            acquisition_function = ScalarizedPosteriorMean(
                model=model_rff_sample, objective=scalarized_obj
            )
        else:
            acquisition_function = PosteriorMean(model=model_rff_sample)

        new_x = optimize_acqf_and_get_suggested_query(
            acq_func=acquisition_function,
            bounds=bounds,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            batch_size=1,  # batching is not supported by RFFs-based sample constructor
            batch_limit=1,
            init_batch_limit=1,
        )
        query.append(new_x.clone())

    query = torch.cat(query, dim=-2)
    query = query.unsqueeze(0)
    return query
