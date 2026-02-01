import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from scipy.stats import ttest_rel
import numpy as np



def evaluate_single_run(
    GROUND_TRUTH_FILE,
    PREDICTIONS_FILE
):
    """
    Returns numeric scores for ONE prompt/config.
    This is what Bayesian Optimization will use.
    """

    df_truth = pd.read_csv(GROUND_TRUTH_FILE, encoding="latin1")
    df_pred  = pd.read_csv(PREDICTIONS_FILE)

    # Align rows
    processed_ids = df_pred["TEXT_ID"].tolist()
    df_truth_subset = df_truth.iloc[processed_ids].copy()

    truth_cols = {"cOPN": "OPN", "cCON": "CON", "cEXT": "EXT", "cAGR": "AGR", "cNEU": "NEU"}
    for col in truth_cols:
        df_truth_subset[col] = df_truth_subset[col].astype(str).str.lower().str.strip()

    y_true = df_truth_subset[list(truth_cols.keys())].rename(columns=truth_cols)
    y_pred = df_pred[["OPN","CON","EXT","AGR","NEU"]]
    y_pred = y_pred.astype(str).apply(lambda col: col.str.lower().str.strip())

    bin_map = {"y": 1, "n": 0}
    y_true_bin = y_true.apply(lambda col: col.map(bin_map).fillna(0))
    y_pred_bin = y_pred.apply(lambda col: col.map(bin_map).fillna(0))

    trait_f1 = {
    trait: f1_score(
        y_true_bin[trait],
        y_pred_bin[trait],
        zero_division=0
    )
    for trait in ["OPN","CON","EXT","AGR","NEU"]
    }


    # ---- Aggregate metrics ----
    macro_f1 = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0)

    return {
        "trait_f1": trait_f1,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1
    }
