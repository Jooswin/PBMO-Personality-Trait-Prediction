import torch
torch.set_default_dtype(torch.float64)


from src.experiment_manager import experiment_manager
from src.personality_bo_utils import evaluate_query

# --- Experiment hyperparameters ---
PROBLEM = "personality_bo"
INPUT_DIM = 3  # [prompt_idx, temperature, top_p]
NUM_ATTRIBUTES = 5  # OPN, CON, EXT, AGR, NEU
OBS_ATTRIBUTES = list(range(NUM_ATTRIBUTES))
COMP_NOISE_TYPE = "probit"
COMP_NOISE = 0.05
ALGO = "SDTS"
BATCH_SIZE = 5
NUM_INIT_QUERIES = 5
NUM_ALGO_ITER = 2 #20
FIRST_TRIAL = 1
LAST_TRIAL = 3
RESTART = False
IGNORE_FAILURES = True

# Optional algorithm parameters (can be empty)
ALGO_PARAMS = {}

# --- Run experiment ---
experiment_manager(
    problem=PROBLEM,
    utility_func=evaluate_query,
    input_dim=INPUT_DIM,
    num_attributes=NUM_ATTRIBUTES,
    obs_attributes=OBS_ATTRIBUTES,
    comp_noise_type=COMP_NOISE_TYPE,
    comp_noise=COMP_NOISE,
    algo=ALGO,
    batch_size=BATCH_SIZE,
    num_init_queries=NUM_INIT_QUERIES,
    num_algo_iter=NUM_ALGO_ITER,
    first_trial=FIRST_TRIAL,
    last_trial=LAST_TRIAL,
    restart=RESTART,
    ignore_failures=IGNORE_FAILURES,
    algo_params=ALGO_PARAMS,
)
