#!/usr/bin/env python3
from typing import Callable, Dict, List, Optional
import os
import sys
import time
import torch
from torch import Tensor
from botorch.models.model import Model
import numpy as np

from src.utils.utils import (
    fit_model,
    generate_initial_data,
    optimize_acqf_and_get_suggested_query,
    get_utility_vals,
    generate_responses,
)
from src.personality_bo_utils import evaluate_query

torch.set_default_dtype(torch.float64)  # double precision for BoTorch
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from src.acquisition_functions.dueling_thompson_sampling import (
    gen_dueling_thompson_sampling_query,
)

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
        #responses = responses.unsqueeze(0) #if any issues later uncomment this 
        t0 = time.time()
        print("Queries shape in one_trial",queries.shape)
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

        # new suggested query
        t0 = time.time()
        print("Just before calling new suggested query")
        new_query = get_new_suggested_query(
            algo=algo,
            model=model,
            batch_size=batch_size,
            input_dim=input_dim,
            algo_params=algo_params,
        )
        t1 = time.time()
        acquisition_time = t1 - t0
        runtimes.append(acquisition_time + model_training_time)
        print("Check if we reached after new_query")
        # get response at new query
        new_utility_vals = get_utility_vals(new_query, utility_func)
        print("Check if we reached after new_utiltiy_vals")
        new_responses = generate_responses(
            new_utility_vals,
            noise_type=comp_noise_type,
            noise_level=comp_noise,
            algo=algo,
        )
        print("Check if we reached after new_responses")

        # update training data
        print("\nAFTER NEW QUERY, UTILITY VAL AND RESPONSE: ")
        print("new query shape: ",new_query.shape)
        print("query shape: ",queries.shape)
        queries = torch.cat((queries, new_query))
        print("After query concat")
        print("new utility_vals shape: ",new_utility_vals.shape)
        print("utility_vals shape: ",utility_vals.shape)
        utility_vals = torch.cat([utility_vals, new_utility_vals], 0)
        print("After utility_vals concat")
        print("new response shape: ",new_responses.shape)
        print("response shape: ",responses.shape)
        responses = torch.cat((responses, new_responses))
        print("After response concat")

        # fit model
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

        # save data
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        if not os.path.exists(results_folder + "queries/"):
            os.makedirs(results_folder + "queries/")
        if not os.path.exists(results_folder + "utility_vals/"):
            os.makedirs(results_folder + "utility_vals/")
        if not os.path.exists(results_folder + "responses/"):
            os.makedirs(results_folder + "responses/")
        if not os.path.exists(results_folder + "runtimes/"):
            os.makedirs(results_folder + "runtimes/")

        queries_reshaped = queries.numpy().reshape(queries.shape[0], -1)
        np.savetxt(
            results_folder + "queries/queries_" + str(trial) + ".txt", queries_reshaped
        )
        utility_vals_reshaped = utility_vals.numpy().reshape(utility_vals.shape[0], -1)
        np.savetxt(
            results_folder + "utility_vals/utility_vals_" + str(trial) + ".txt",
            utility_vals_reshaped,
        )
        np.savetxt(
            results_folder + "responses/responses_" + str(trial) + ".txt",
            responses.numpy(),
        )
        np.savetxt(
            results_folder + "runtimes/runtimes_" + str(trial) + ".txt",
            np.atleast_1d(runtimes),
        )
    utility_vals_reshaped = utility_vals.numpy().reshape(utility_vals.shape[0], -1)
    return utility_vals_reshaped


# computes the new query to be shown to the DM
def get_new_suggested_query(
    algo: str,
    model: Model,
    batch_size,
    input_dim: int,
    algo_params: Optional[Dict] = None,
) -> Tensor:
    standard_bounds = torch.tensor([[0.0] * input_dim, [1.0] * input_dim])
    num_restarts = 4 * input_dim
    raw_samples = 120 * input_dim

    if algo == "Random":
        new_query = generate_random_queries(
            num_queries=1, batch_size=batch_size, input_dim=input_dim
        )
    elif algo == "SDTS":
        print("Check before SDTS")
        new_query = gen_dueling_thompson_sampling_query(
            model,
            batch_size,
            standard_bounds,
            num_restarts,
            raw_samples,
            scalarize=True,
            fix_scalarization=True,
        )
    elif algo == "SDTS-HVS":
        new_query = gen_dueling_thompson_sampling_query(
            model,
            batch_size,
            standard_bounds,
            num_restarts,
            raw_samples,
            scalarize=True,
            fix_scalarization=True,
            scalarization="hypervolume",
        )
    elif algo == "SDTS-HS":
        new_query = gen_dueling_thompson_sampling_query(
            model,
            batch_size,
            standard_bounds,
            num_restarts,
            raw_samples,
            scalarize=True,
            fix_scalarization=False,
        )
    elif algo == "I-PBO-DTS":
        new_query = gen_dueling_thompson_sampling_query(
            model,
            batch_size,
            standard_bounds,
            num_restarts,
            raw_samples,
            scalarize=False,
        )
    elif algo == "qParEGO":
        mean_at_train_inputs = model.posterior(model.train_inputs[0][0]).mean.detach()
        weights = sample_simplex(mean_at_train_inputs.shape[-1]).squeeze()
        chebyshev_scalarization = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=mean_at_train_inputs))
        acquisition_function = qExpectedImprovement(
            model=model,
            objective=chebyshev_scalarization,
            best_f=chebyshev_scalarization(mean_at_train_inputs).max(),
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
        )
    elif algo == "qEHVI":
        mean_at_train_inputs = model.posterior(model.train_inputs[0][0]).mean.detach()
        ref_point = mean_at_train_inputs.min(0).values
        partitioning = FastNondominatedPartitioning(
            ref_point=ref_point,
            Y=mean_at_train_inputs,
        )
        acquisition_function = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
        )
    elif algo == "qJES":
        num_pareto_samples = 10
        num_pareto_points = 10
        optimizer_kwargs = {
            "pop_size": 5000,
            "max_tries": 10,
        }
        ps, pf = sample_optimal_points(
            model=model,
            bounds=standard_bounds,
            num_samples=num_pareto_samples,
            num_points=num_pareto_points,
            optimizer_kwargs=optimizer_kwargs,
        )
        hypercell_bounds = compute_sample_box_decomposition(pf)
        acquisition_function = qLowerBoundMultiObjectiveJointEntropySearch(
            model=model,
            pareto_sets=ps,
            pareto_fronts=pf,
            hypercell_bounds=hypercell_bounds,
            estimation_type="LB",
        )
    elif algo == "qMES":
        num_pareto_samples = 10
        num_pareto_points = 10
        optimizer_kwargs = {
            "pop_size": 5000,
            "max_tries": 10,
        }
        ps, pf = sample_optimal_points(
            model=model,
            bounds=standard_bounds,
            num_samples=num_pareto_samples,
            num_points=num_pareto_points,
            optimizer_kwargs=optimizer_kwargs,
        )
        hypercell_bounds = compute_sample_box_decomposition(pf)
        acquisition_function = qLowerBoundMultiObjectiveMaxValueEntropySearch(
        model=model,
        pareto_fronts=pf,
        hypercell_bounds=hypercell_bounds,
        estimation_type="LB",
    )
    elif algo == "qPHVS":
        mean_at_train_inputs = model.posterior(model.train_inputs[0][0]).mean.detach()
        ref_point = mean_at_train_inputs.min(0).values
        model_rff_sample = get_preferential_gp_rff_sample(model=model, n_samples=1)
        model = PosteriorMeanModel(model=model_rff_sample)
        sampler = StochasticSampler(sample_shape=torch.Size([1]))  # dummy sampler
        acquisition_function = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=FastNondominatedPartitioning(
                ref_point=ref_point,
                Y=torch.empty(
                    (0, ref_point.shape[0]),
                    dtype=ref_point.dtype,
                    device=ref_point.device,
                ),
            ),  # create empty partitioning
            sampler=sampler,
        )
    if algo != "Random" and "DTS" not in algo:
        new_query = optimize_acqf_and_get_suggested_query(
            acq_func=acquisition_function,
            bounds=standard_bounds,
            batch_size=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        new_query = new_query.unsqueeze(0)
    return new_query
