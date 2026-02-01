#!/usr/bin/env python3

from typing import Callable, Dict, Optional

import numpy as np
import os
import sys
import time
import torch
from botorch.acquisition import PosteriorMean, qSimpleRegret
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.model import Model
from botorch.sampling.samplers import SobolQMCNormalSampler
from torch import Tensor

from src.fit_model import fit_model
from src.utils import generate_random_queries, compute_posterior_mean_maximizer
from src.mcpbo_trial import get_new_suggested_query


# runs a single iteration for a given problem
def run_one_iteration(
    queries: Tensor,
    responses: Tensor,
    input_dim: int,
    algo: str,
    trial: int,
    iteration: int,
) -> Tensor:
    model = fit_model(
        queries,
        responses,
        model_id=model_id,
        algo=algo,
    )

    if iteration == 1:
        if not os.path.exists(results_folder + "posterior_mean_maximizers/"):
            os.makedirs(results_folder + "posterior_mean_maximizers/")
        if not os.path.exists(results_folder + "runtimes/"):
            os.makedirs(results_folder + "runtimes/")
        queries = generate_random_queries(
            4 * input_dim, batch_size=2, input_dim=input_dim, seed=trial
        )
        runtimes = []
    else:
        posterior_mean_maximizers = torch.tensor(
            np.atleast_2d(
                np.loadtxt(
                    results_folder
                    + "posterior_mean_maximizers/posterior_mean_maximizers_"
                    + str(trial)
                    + ".txt"
                )
            )
        )
        runtimes = list(
            np.atleast_1d(
                np.loadtxt(results_folder + "runtimes/runtimes_" + str(trial) + ".txt")
            )
        )

    # fit model
    t0 = time.time()
    model = fit_model(
        queries,
        responses,
        model_type=model_type,
        likelihood="probit",
    )
    t1 = time.time()
    model_training_time = t1 - t0

    # posterior mean maximizer
    posterior_mean_maximizer = compute_posterior_mean_maximizer(
        model=model,
        model_type=model_type,
        input_dim=input_dim,
    )
    if iteration == 1:
        posterior_mean_maximizers = posterior_mean_maximizer
    else:
        posterior_mean_maximizers = torch.cat(
            (posterior_mean_maximizers, posterior_mean_maximizer)
        )

    t0 = time.time()
    new_query = get_new_suggested_query(
        algo=algo,
        model=model,
        batch_size=2,
        input_dim=input_dim,
        model_type=model_type,
    )
    t1 = time.time()
    acquisition_time = t1 - t0
    runtimes.append(acquisition_time + model_training_time)
    np.savetxt(
        results_folder
        + "posterior_mean_maximizers/posterior_mean_maximizers_"
        + str(trial)
        + ".txt",
        np.atleast_1d(posterior_mean_maximizers),
    )
    np.savetxt(
        results_folder + "runtimes/runtimes_" + str(trial) + ".txt",
        np.atleast_1d(runtimes),
    )
    return new_query
