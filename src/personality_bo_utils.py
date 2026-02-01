import torch
from eval_personality_bo import evaluate_personality_config

# Mapping query vector to actual LLM config
PROMPT_NAMES = ["zero_shot_prompt", "few_shot_prompt", "cot_prompt", "proposed_prompt"]

def query_to_config(query: torch.Tensor):
    """
    Converts a query tensor (size = [batch_size, input_dim]) to LLM configs.
    query[:,0] = prompt index (0..3)
    query[:,1] = temperature (0..1)
    query[:,2] = top_p (0..1)
    """
    configs = []
    for q in query:
        prompt_idx = int(round(q[0].item()))
        temperature = float(q[1].item())
        top_p = float(q[2].item())
        prompt_name = PROMPT_NAMES[prompt_idx]
        configs.append({
            "prompt_name": prompt_name,
            "temperature": temperature,
            "top_p": top_p
        })
    return configs

def evaluate_query(query):
    batch_size = query.shape[0]
    num_traits = 5
    return torch.rand(batch_size, num_traits)


'''def evaluate_query(query: torch.Tensor):
    """
    Takes a query tensor and returns the utility values (trait F1 scores)
    as a torch tensor (shape = [batch_size, num_traits])
    """
    configs = query_to_config(query)
    results = []
    for cfg in configs:
        score = evaluate_personality_config(
            prompt_name=cfg["prompt_name"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"]
        )
        # trait_f1 dict to tensor [OPN, CON, EXT, AGR, NEU]
        trait_vals = [score["trait_f1"][trait] for trait in ["OPN","CON","EXT","AGR","NEU"]]
        results.append(trait_vals)
    return torch.tensor(results, dtype=torch.float32)
'''