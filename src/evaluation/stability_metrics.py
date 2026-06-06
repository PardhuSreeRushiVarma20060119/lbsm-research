"""
stability_metrics.py
====================
LBSM — Evaluation
------------------
Stability and robustness metrics: bootstrap variance of AUC/ARI,
cross-seed consistency, and detector stability across window sizes.
"""

from __future__ import annotations
from typing import Callable, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def bootstrap_auc(
    scores  : np.ndarray,
    y_gt    : np.ndarray,
    n_boot  : int = 200,
    seed    : int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap mean ± std of ROC-AUC.

    Returns
    -------
    (mean_auc, std_auc, auc_original)
    """
    rng  = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(scores), len(scores), replace=True)
        try:
            aucs.append(float(roc_auc_score(y_gt[idx], scores[idx])))
        except ValueError:
            pass
    auc_orig = float(roc_auc_score(y_gt, scores))
    return float(np.mean(aucs)), float(np.std(aucs)), auc_orig


def detector_stability_table(
    score_dict: dict,
    y_gt      : np.ndarray,
    n_boot    : int = 100,
    seed      : int = 42,
) -> pd.DataFrame:
    """Bootstrap AUC mean ± std for a dictionary of detectors.

    Parameters
    ----------
    score_dict : {detector_name: score_array}

    Returns
    -------
    df : pd.DataFrame  columns=[detector, auc_mean, auc_std, auc_original]
    """
    rows = []
    for name, scores in score_dict.items():
        mu, std, orig = bootstrap_auc(scores, y_gt, n_boot, seed)
        rows.append({"detector": name, "auc_mean": mu,
                     "auc_std": std, "auc_original": orig})
    return pd.DataFrame(rows).set_index("detector")