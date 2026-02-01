import json
import pandas as pd
import time

from openai_prompting_service import llm
from performance_metrics import evaluate_single_run


# -----------------------------
# Fixed resources (same as before)
# -----------------------------

INPUT_FILE = "F:\\downloads\\essays.csv"
N = 2

with open("prompts.json", "r") as f:
    PROMPTS = json.load(f)

traits = {
    "Openness": "OPN",
    "Conscientiousness": "CON",
    "Extraversion": "EXT",
    "Agreeableness": "AGR",
    "Neuroticism": "NEU"
}


# -----------------------------
# NEW: evaluation function
# -----------------------------

def evaluate_personality_config(
    prompt_name: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    sleep_every: int = 5
):
    """
    Evaluates ONE LLM configuration:
    - one prompt template
    - one temperature
    - one top_p

    Returns:
        dict with metric scores (one per trait)
    """

    df = pd.read_csv(INPUT_FILE, encoding="latin1")
    texts = df["TEXT"].head(N)

    prompt = PROMPTS[prompt_name]

    all_results = []

    for idx, text in enumerate(texts):
        row_result = {"TEXT_ID": idx}

        if (idx + 1) % sleep_every == 0:
            time.sleep(61)

        # ---- LLM call (same logic as before) ----

        time.sleep(0.1)
        out = llm(
            user_input=text,
            system_role=prompt,
            temperature=temperature,
            top_p=top_p
        )

        # Initialize traits
        for short in traits.values():
            row_result[short] = "NA"

        # Parse output
        for line in out.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip().lower()
                if key in traits:
                    row_result[traits[key]] = val

        all_results.append(row_result)

    # Create dataframe (same as before)
    result_df = pd.DataFrame(all_results)

    # TEMP CSV (metrics expects a file)
    tmp_output = "tmp_eval_output.csv"
    result_df.to_csv(tmp_output, index=False)

    # ---- Compute metrics (reuse your code) ----
    scores = evaluate_single_run(
        INPUT_FILE,
        tmp_output
    )

    return scores

scores = evaluate_personality_config(
    prompt_name="few_shot_prompt",
    temperature=0.0,
    top_p=1.0
)

print(scores)

