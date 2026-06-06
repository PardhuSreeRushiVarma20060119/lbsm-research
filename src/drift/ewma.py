"""
ewma.py
=======
LBSM — Drift Detection
-----------------------
Exponentially Weighted Moving Average (EWMA) detector for online
behavioral drift in agent telemetry streams.

EWMA maintains a smoothed estimate of each feature and scores each
incoming observation by its residual distance from the smoothed state.
A spike in the residual indicates a rapid distributional shift —
the signature of a regime transition.

Mathematical formulation
------------------------
EWMA update:    μ̂_t = α · x_t + (1 − α) · μ̂_{t−1}
Residual score: r_t = ||x_t − μ̂_{t−1}||₂

Anomaly flag:   r_t > μ_r + k · σ_r   (k standard deviations above mean)

Reference
---------
"Latent Behavioral State Machines: Manifold Geometry of Adaptive Agent Telemetry"
Section 7.2 — Online Detection: EWMA Residual Scoring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class EWMAResult:
    """EWMA scoring result for one agent sequence.

    Attributes
    ----------
    scores        : residual norm at each timestep  shape (T,)
    ewma_path     : smoothed feature estimate        shape (T, d)
    flags         : binary anomaly flag              shape (T,)
    threshold     : scalar threshold used for flagging
    alpha         : smoothing parameter
    """
    scores    : np.ndarray
    ewma_path : np.ndarray
    flags     : np.ndarray
    threshold : float
    alpha     : float


# ---------------------------------------------------------------------------
# Core EWMA scorer
# ---------------------------------------------------------------------------
def ewma_scores(
    X    : np.ndarray,
    alpha: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute EWMA residual scores for a single agent sequence.

    Parameters
    ----------
    X     : np.ndarray  shape (T, d)  — raw feature sequence
    alpha : smoothing factor ∈ (0, 1)
            smaller → slower adaptation, more sensitive to sustained drift
            larger  → faster adaptation, sensitive only to sharp spikes

    Returns
    -------
    scores    : np.ndarray  shape (T,)  — ||x_t − μ̂_{t−1}||₂
    ewma_path : np.ndarray  shape (T, d) — smoothed state at each step
    """
    T, d   = X.shape
    path   = np.empty((T, d))
    scores = np.empty(T)

    mu = X[0].copy()       # initialise at first observation
    for t in range(T):
        scores[t] = float(np.linalg.norm(X[t] - mu))
        mu        = alpha * X[t] + (1.0 - alpha) * mu
        path[t]   = mu

    return scores, path


def fit_ewma(
    X          : np.ndarray,
    alpha      : float = 0.10,
    threshold_k: float = 3.0,
    warmup     : int = 50,
) -> EWMAResult:
    """Fit EWMA scorer and flag anomalies above the adaptive threshold.

    Threshold is estimated from the warmup period as:
        threshold = mean(scores[:warmup]) + k · std(scores[:warmup])

    Parameters
    ----------
    X           : np.ndarray  shape (T, d)
    alpha       : EWMA smoothing factor
    threshold_k : number of std deviations above warmup mean
    warmup      : number of timesteps used to estimate threshold

    Returns
    -------
    result : EWMAResult
    """
    scores, path = ewma_scores(X, alpha)

    # Estimate threshold from warmup period
    w         = min(warmup, len(scores))
    threshold = float(scores[:w].mean() + threshold_k * scores[:w].std() + 1e-9)
    flags     = (scores > threshold).astype(np.int8)

    return EWMAResult(
        scores    = scores,
        ewma_path = path,
        flags     = flags,
        threshold = threshold,
        alpha     = alpha,
    )


# ---------------------------------------------------------------------------
# Multi-agent EWMA scoring
# ---------------------------------------------------------------------------
def ewma_all_agents(
    df          : pd.DataFrame,
    feature_cols: Tuple[str, ...],
    alpha       : float = 0.10,
    threshold_k : float = 3.0,
    warmup      : int = 50,
    agent_col   : str = "agent_id",
    time_col    : str = "timestep",
) -> pd.DataFrame:
    """Apply EWMA scoring to every agent in the telemetry DataFrame.

    Returns a DataFrame aligned to df_sorted order with columns:
    ``ewma_score``, ``ewma_flag``.

    Parameters
    ----------
    df           : full telemetry DataFrame
    feature_cols : features to use as observations
    alpha        : EWMA smoothing factor
    threshold_k  : flagging threshold in standard deviations
    warmup       : timesteps for threshold estimation

    Returns
    -------
    df_out : df sorted by [agent_col, time_col] with added score/flag columns
    """
    df_s  = df.sort_values([agent_col, time_col]).copy()
    scores_all = np.empty(len(df_s))
    flags_all  = np.empty(len(df_s), dtype=np.int8)

    offset = 0
    for aid in sorted(df_s[agent_col].unique()):
        sub = df_s[df_s[agent_col] == aid]
        X_a = sub[list(feature_cols)].values.astype(np.float64)
        res = fit_ewma(X_a, alpha=alpha, threshold_k=threshold_k, warmup=warmup)
        n   = len(X_a)
        scores_all[offset : offset + n] = res.scores
        flags_all[offset  : offset + n] = res.flags
        offset += n

    df_s["ewma_score"] = scores_all
    df_s["ewma_flag"]  = flags_all
    return df_s


# ---------------------------------------------------------------------------
# Alpha sensitivity sweep
# ---------------------------------------------------------------------------
def alpha_sweep(
    X          : np.ndarray,
    y_gt       : np.ndarray,
    alpha_grid : Tuple[float, ...] = (0.02, 0.05, 0.10, 0.20, 0.40),
) -> pd.DataFrame:
    """Evaluate EWMA AUC across smoothing factors.

    Parameters
    ----------
    X          : np.ndarray  shape (T, d)
    y_gt       : binary ground-truth anomaly labels  shape (T,)
    alpha_grid : smoothing values to test

    Returns
    -------
    df : pd.DataFrame  columns=[alpha, auc_roc, auc_pr]
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    rows = []
    for alpha in alpha_grid:
        scores, _ = ewma_scores(X, alpha)
        try:
            auc_roc = float(roc_auc_score(y_gt, scores))
            auc_pr  = float(average_precision_score(y_gt, scores))
        except ValueError:
            auc_roc = auc_pr = float("nan")
        rows.append({"alpha": alpha, "auc_roc": auc_roc, "auc_pr": auc_pr})

    return pd.DataFrame(rows)