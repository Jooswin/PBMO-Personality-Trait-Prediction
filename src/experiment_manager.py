from typing import Callable, Dict, List, Optional
from src.one_trial import one_trial
import torch

def experiment_manager(
    problem: str,
    utility_func: Callable,
    input_dim: int,
    num_attributes: int,
    obs_attributes: List,
    comp_noise_type: str,
    comp_noise: float,
    algo: str,
    batch_size: int,
    num_init_queries: int,
    num_algo_iter: int,
    first_trial: int,
    last_trial: int,
    restart: bool = False,
    ignore_failures: bool = False,
    algo_params: Optional[Dict] = None,
):
    """
    Runs multiple trials of PBMO personality experiment.
    Each trial calls `one_trial`.
    """

    for trial in range(first_trial, last_trial + 1):
        print(f"\n=== Starting trial {trial} ===\n")
        try:
            utility_vals = one_trial(
                problem=problem,
                utility_func=utility_func,
                input_dim=input_dim,
                num_attributes=num_attributes,
                obs_attributes=obs_attributes,
                comp_noise_type=comp_noise_type,
                comp_noise=comp_noise,
                algo=algo,
                batch_size=batch_size,
                num_init_queries=num_init_queries,
                num_algo_iter=num_algo_iter,
                trial=trial,
                restart=restart,
                ignore_failures=ignore_failures,
                algo_params=algo_params,
            )
            print(f"Trial {trial} completed successfully. Final utility values:\n{utility_vals}\n")
        except Exception as e:
            if ignore_failures:
                print(f"Trial {trial} failed but ignored: {e}")
            else:
                raise e
