from typing import List, Optional

import torch
from torch import Tensor
from torch.distributions import Bernoulli, Normal, Gumbel

from botorch.acquisition import AcquisitionFunction, PosteriorMean
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex
from gpytorch.mlls import ExactMarginalLogLikelihood

from src.models.variational_preferential_gp import VariationalPreferentialGP
from src.models.pairwise_kernel_variational_gp import PairwiseKernelVariationalGP

# --------------------------------------
# Model fitting
# --------------------------------------
def fit_model(
    queries: Tensor,
    utility_vals: Tensor,
    responses: Tensor,
    obs_attributes: List,
    model_id: int,
    algo: str,
):
    # Choose GP model
    if model_id == 1:
        Model = PairwiseKernelVariationalGP
    elif model_id == 2:
        Model = VariationalPreferentialGP

    queries = queries.to(dtype=torch.float64)
    utility_vals = utility_vals.to(dtype=torch.float64)
    responses = responses.to(dtype=torch.float64)

    for i in range(10):
        try:
            if algo == "Random":
                return None
            elif algo == "I-PBO-DTS":
                model = Model(queries, responses)
            else:
                models = []
                num_attributes = responses.shape[-1]
                print("Test Inside fit_model 0")
                print("queries shape:", queries.shape)
                queries_reshaped = queries.reshape(
                    queries.shape[0] * queries.shape[1], queries.shape[2]
                ).to(dtype=torch.float64)
                print("Test Inside fit_model 1")
                utility_vals_reshaped = utility_vals.reshape(
                    utility_vals.shape[0] * utility_vals.shape[1], utility_vals.shape[2]
                ).to(dtype=torch.float64)

                print("Test inside fit_model 2")

                for j in range(num_attributes):
                    if obs_attributes[j]:
                        train_Yvar = torch.full_like(
                            utility_vals_reshaped[..., [j]], 1e-4, dtype=torch.float64
                        )
                        model = SingleTaskGP(
                            train_X=queries_reshaped,
                            train_Y=utility_vals_reshaped[..., [j]],
                            outcome_transform=Standardize(m=1),
                        )
                        mll = ExactMarginalLogLikelihood(model.likelihood, model)
                        fit_gpytorch_mll(mll)
                    else:
                        model = Model(queries, responses[..., j])
                    models.append(model)
                model = ModelListGP(*models)
            return model
        except Exception as error:
            print(f"Number of failed attempts to train the model: {i+1}")
            print(error)


# --------------------------------------
# Generate initial data
# --------------------------------------
def generate_random_queries(num_queries, batch_size, input_dim, seed=None) -> Tensor:
    if seed is not None:
        old_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
    queries = torch.rand([num_queries, batch_size, input_dim], dtype=torch.float64)
    print("Debug Inside generate_random_queries")
    if seed is not None:
        torch.random.set_rng_state(old_state)
    return queries


def get_utility_vals(queries, utility_func) -> Tensor:
    print("Queries shape inside get_utility_vals: ",queries.shape)
    #print("Queries in get_utility_vals: ",queries)
    queries_2d = queries.reshape(queries.shape[0] * queries.shape[1], queries.shape[2])
    print("2D Query shape in get_utility_vals: ",queries_2d.shape)
    print("2D Queries in get_utility_vals: ",queries_2d)
    
    utility_vals = utility_func(queries_2d).to(dtype=torch.float64)
    print("Utility val shape inside utility val: ",utility_vals.shape)
    utility_vals = utility_vals.reshape(queries.shape[0], queries.shape[1], utility_vals.shape[1])
    print("Debug inside get_utility_vals()")
    return utility_vals


def corrupt_vals(vals, noise_type, noise_level):
    vals = vals.to(dtype=torch.float64)
    if noise_type == "noiseless":
        return vals
    elif noise_type == "probit":
        normal = Normal(0.0, noise_level)
        noise = normal.sample(vals.shape[:-1]).to(dtype=torch.float64)
        return vals + noise
    elif noise_type == "logit":
        gumbel = Gumbel(0.0, noise_level)
        noise = gumbel.sample(vals.shape[:-1]).to(dtype=torch.float64)
        return vals + noise
    elif noise_type == "constant":
        corrupted_vals = vals.clone()
        for i in range(vals.shape[0]):
            for j in range(vals.shape[-1]):
                coin_toss = Bernoulli(noise_level).sample().item()
                if coin_toss == 1.0:
                    corrupted_vals[i, 0, j] = vals[i, 1, j]
                    corrupted_vals[i, 1, j] = vals[i, 0, j]
        return corrupted_vals
    return vals


def generate_responses(utility_vals, noise_type, noise_level, algo):
    corrupted_vals = corrupt_vals(utility_vals, noise_type, noise_level)
    if algo == "I-PBO-DTS":
        weights = sample_simplex(d=utility_vals.shape[-1]).squeeze().to(dtype=torch.float64)
        chebyshev_scalarization = get_chebyshev_scalarization(weights=weights, Y=corrupted_vals[:, 0, :])
        responses = torch.argmax(chebyshev_scalarization(corrupted_vals), dim=-1)
    else:
        responses = torch.argmax(corrupted_vals, dim=-2)
        print("Debug Inside generate_responses()")
    return responses


def generate_initial_data(num_queries, batch_size, input_dim, utility_func, comp_noise_type, comp_noise, algo, seed=None):
    print("Inside generate_initial_data")
    queries = generate_random_queries(num_queries, batch_size, input_dim, seed)
    utility_vals = get_utility_vals(queries, utility_func)
    responses = generate_responses(utility_vals, comp_noise_type, comp_noise, algo)
    return queries, utility_vals, responses


# --------------------------------------
# Acquisition optimization
# --------------------------------------
def optimize_acqf_and_get_suggested_query(
    acq_func: AcquisitionFunction,
    bounds: Tensor,
    batch_size: int,
    num_restarts: int,
    raw_samples: int,
    batch_initial_conditions: Optional[Tensor] = None,
    batch_limit: Optional[int] = 4,
    init_batch_limit: Optional[int] = 20,
) -> Tensor:
    candidates, acq_values = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds.to(dtype=torch.float64),
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        batch_initial_conditions=batch_initial_conditions,
        options={
            "batch_limit": batch_limit,
            "init_batch_limit": init_batch_limit,
            "maxiter": 100,
            "nonnegative": False,
            "method": "L-BFGS-B",
        },
        return_best_only=True,
    )
    candidates = candidates.detach().to(dtype=torch.float64)
    return candidates
