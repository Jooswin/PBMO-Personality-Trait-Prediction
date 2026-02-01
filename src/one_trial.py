#!/usr/bin/env python3
from typing import Callable, Dict, List, Optional
import os
import sys
import time
import torch
from torch import Tensor
import numpy as np

from src.utils.utils import (
    fit_model,
    generate_initial_data,
    optimize_acqf_and_get_suggested_query,
)
from src.personality_bo_utils import evaluate_query

torch.set_default_dtype(torch.float64)  # double precision for BoTorch

# -------------------------------
# Runs a single trial of personality PBMO
# -------------------------------
def one_trial(
    problem: str,
    utility_func: Callable = evaluate_query,
    input_dim: int = 3,
    num_attributes: int = 5,
    obs_attributes: List = None,
    comp_noise_type: str = "probit",
    comp_noise: float = 0.05,
    algo: str = "SDTS",
    batch_size: int = 2,
    num_init_queries: int = 5,
    num_algo_iter: int = 20,
    trial: int = 1,
    restart: bool = False,
    ignore_failures: bool = False,
    model_id: int = 2,
    algo_params: Optional[Dict] = None,
) -> Tensor:
    if obs_attributes is None:
        obs_attributes = list(range(num_attributes))

    # --- Directories ---
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    project_path = os.path.dirname(script_dir)
    results_folder = f"{project_path}/experiments/results/{problem}/{algo}/"

    # --- Restart or generate initial data ---
    if restart:
        try:
            queries = torch.tensor(np.loadtxt(results_folder + f"queries/queries_{trial}.txt"))
            queries = queries.reshape(queries.shape[0], batch_size, -1)
            utility_vals = torch.tensor(np.loadtxt(results_folder + f"utility_vals/utility_vals_{trial}.txt"))
            utility_vals = utility_vals.reshape(utility_vals.shape[0], batch_size, -1)
            responses = torch.tensor(np.loadtxt(results_folder + f"responses/responses_{trial}.txt"))
            runtimes = list(np.atleast_1d(np.loadtxt(results_folder + f"runtimes/runtimes_{trial}.txt")))

            t0 = time.time()
            model = fit_model(
                queries,
                utility_vals,
                responses,
                obs_attributes=obs_attributes,
                model_id=model_id,
                algo=algo,
            )
            t1 = time.time()
            model_training_time = t1 - t0
            iteration = queries.shape[0] - num_init_queries
            print("Restarting experiment from saved data.")
        except Exception:
            restart = False  # fallback to initial data

    if not restart:
        queries, utility_vals, responses = generate_initial_data(
            num_queries=num_init_queries,
            batch_size=batch_size,
            input_dim=input_dim,
            utility_func=utility_func,
            comp_noise_type=comp_noise_type,
            comp_noise=comp_noise,
            algo=algo,
            seed=trial,
        )
        t0 = time.time()
        model = fit_model(
            queries,
            utility_vals,
            responses,
            obs_attributes=obs_attributes,
            model_id=model_id,
            algo=algo,
        )
        t1 = time.time()
        model_training_time = t1 - t0
        runtimes = []
        iteration = 0

    # --- Main loop ---
    while iteration < num_algo_iter:
        iteration += 1
        print(f"Problem: {problem} | Algo: {algo} | Trial: {trial} | Iteration: {iteration}")

        # --- Suggest new queries (batch-aware) ---
        t0 = time.time()
        new_query = optimize_acqf_and_get_suggested_query(
            acq_func=algo_params.get("acq_func"),  # make sure algo_params has the acquisition function
            bounds=torch.tensor([[0.0, 0.0, 0.0], [len(utility_func.__globals__['PROMPT_NAMES'])-1, 1.0, 1.0]], dtype=torch.float64),
            batch_size=batch_size,
            num_restarts=5,
            raw_samples=20,
        )
        t1 = time.time()
        acquisition_time = t1 - t0
        runtimes.append(acquisition_time + model_training_time)

        # --- Evaluate query via personality PBMO ---
        new_utility_vals = utility_func(new_query)
        new_responses = new_utility_vals.clone()  # deterministic here

        # --- Update dataset ---
        queries = torch.cat((queries, new_query), dim=0)
        utility_vals = torch.cat((utility_vals, new_utility_vals), dim=0)
        responses = torch.cat((responses, new_responses), dim=0)

        # --- Refit model ---
        t0 = time.time()
        model = fit_model(
            queries,
            utility_vals,
            responses,
            obs_attributes=obs_attributes,
            model_id=model_id,
            algo=algo,
        )
        t1 = time.time()
        model_training_time = t1 - t0

        # --- Save results ---
        os.makedirs(results_folder + "queries/", exist_ok=True)
        os.makedirs(results_folder + "utility_vals/", exist_ok=True)
        os.makedirs(results_folder + "responses/", exist_ok=True)
        os.makedirs(results_folder + "runtimes/", exist_ok=True)

        np.savetxt(results_folder + f"queries/queries_{trial}.txt", queries.numpy().reshape(queries.shape[0], -1))
        np.savetxt(results_folder + f"utility_vals/utility_vals_{trial}.txt", utility_vals.numpy().reshape(utility_vals.shape[0], -1))
        np.savetxt(results_folder + f"responses/responses_{trial}.txt", responses.numpy())
        np.savetxt(results_folder + f"runtimes/runtimes_{trial}.txt", np.atleast_1d(runtimes))

    return utility_vals
