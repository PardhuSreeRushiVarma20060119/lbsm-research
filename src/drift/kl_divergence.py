"""
kl_divergence.py
================
LBSM — Drift Detection
-----------------------
KL-divergence-based distributional drift detector.

Compares the empirical distribution of a sliding test window against
a reference (healthy-regime) distribution. A large KL divergence
signals that the current window's telemetry no longer resembles
healthy behaviour — a distributional regime shift.

Mathematical formulation (Gaussian approximation)
--------------------------------------------------
Assuming Gaussian distributions p ~ N(μ_p, Σ_p) and q ~ N(μ_q, Σ_q):

KL(p || q) = 0.5 · [tr(Σ_q⁻¹ Σ_p) + (μ_q − μ_p)ᵀ Σ_q⁻¹ (μ_q − μ_p) − d + ln(|Σ_q|/|Σ_p|)]

For diagonal covariances this reduces to a sum over features, making it
computationally efficient for online use.

Reference
---------
"Latent Behavioral Structure in Low-Dimensional Statistical Manifolds"
Section 7.3 — Distributional Drift: KL Divergence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class KLDriftResult:
    """KL divergence drift detection result.

    Attributes
    ----------
    kl_scores  : KL(window || reference) at each test window start  shape (n_windows,)
    t_starts   : test window start timestep indices                  shape (n_windows,)
    flags      : binary drift flag (score > threshold)               shape (n_windows,)
    threshold  : scalar threshold
    window_size: test window size used
    """
    kl_scores  : np.ndarray
    t_starts   : np.ndarray
    flags      : np.ndarray
    threshold  : float
    window_size: int


# ---------------------------------------------------------------------------
# Gaussian KL divergence
# ---------------------------------------------------------------------------
def gaussian_kl(
    mu_p : np.ndarray,
    var_p: np.ndarray,
    mu_q : np.ndarray,
    var_q: np.ndarray,
) -> float:
    """KL(p || q) for diagonal-covariance Gaussians.

    KL = 0.5 · Σ_k [ln(σ²_q/σ²_p) + σ²_p/σ²_q + (μ_p − μ_q)²/σ²_q − 1]

    Parameters
    ----------
    mu_p, var_p : mean and variance of distribution p  shape (d,)
    mu_q, var_q : mean and variance of distribution q  shape (d,)

    Returns
    -------
    kl : float ≥ 0
    """
    var_q = var_q + 1e-9
    var_p = var_p + 1e-9
    kl = 0.5 * np.sum(
        np.log(var_q / var_p)
        + var_p / var_q
        + (mu_p - mu_q) ** 2 / var_q
        - 1.0
    )
    return float(max(0.0, kl))


# ---------------------------------------------------------------------------
# Reference distribution fitting
# ---------------------------------------------------------------------------
def fit_reference(
    X_reference: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a diagonal Gaussian reference distribution from healthy data.

    Parameters
    ----------
    X_reference : np.ndarray  shape (N_ref, d)
                  Observations from the healthy (non-anomalous) regime

    Returns
    -------
    mu_ref  : np.ndarray  shape (d,)
    var_ref : np.ndarray  shape (d,)
    """
    return X_reference.mean(axis=0), X_reference.var(axis=0)


# ---------------------------------------------------------------------------
# Sliding-window KL scoring
# ---------------------------------------------------------------------------
def kl_drift_scores(
    X          : np.ndarray,
    mu_ref     : np.ndarray,
    var_ref    : np.ndarray,
    window_size: int = 50,
    step       : int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute KL(window || reference) for each sliding window.

    Parameters
    ----------
    X           : np.ndarray  shape (T, d)  — agent telemetry sequence
    mu_ref      : reference distribution mean   shape (d,)
    var_ref     : reference distribution variance shape (d,)
    window_size : number of timesteps per test window
    step        : stride between windows

    Returns
    -------
    kl_scores : np.ndarray  shape (n_windows,)
    t_starts  : np.ndarray  shape (n_windows,)  — window start indices
    """
    T = len(X)
    kl_scores_list = []
    t_starts_list  = []

    for t in range(0, T - window_size + 1, step):
        win    = X[t : t + window_size]
        mu_w   = win.mean(axis=0)
        var_w  = win.var(axis=0)
        kl     = gaussian_kl(mu_w, var_w, mu_ref, var_ref)
        kl_scores_list.append(kl)
        t_starts_list.append(t)

    return np.array(kl_scores_list), np.array(t_starts_list)


def fit_kl_detector(
    X          : np.ndarray,
    mu_ref     : np.ndarray,
    var_ref    : np.ndarray,
    window_size: int = 50,
    step       : int = 1,
    threshold_k: float = 3.0,
    warmup_frac: float = 0.10,
) -> KLDriftResult:
    """Fit KL drift detector and flag windows exceeding the threshold.

    Threshold estimated from the first ``warmup_frac`` fraction of windows:
        threshold = mean(kl[:n_warmup]) + k · std(kl[:n_warmup])

    Parameters
    ----------
    X            : agent feature sequence  shape (T, d)
    mu_ref       : reference mean
    var_ref      : reference variance
    window_size  : test window size
    step         : stride
    threshold_k  : standard deviations above warmup mean
    warmup_frac  : fraction of windows used to set threshold

    Returns
    -------
    result : KLDriftResult
    """
    kl_scores, t_starts = kl_drift_scores(X, mu_ref, var_ref, window_size, step)

    n_warmup  = max(1, int(len(kl_scores) * warmup_frac))
    threshold = float(
        kl_scores[:n_warmup].mean()
        + threshold_k * kl_scores[:n_warmup].std()
        + 1e-9
    )
    flags = (kl_scores > threshold).astype(np.int8)

    return KLDriftResult(
        kl_scores   = kl_scores,
        t_starts    = t_starts,
        flags       = flags,
        threshold   = threshold,
        window_size = window_size,
    )


# ---------------------------------------------------------------------------
# Multi-agent KL scoring
# ---------------------------------------------------------------------------
def kl_all_agents(
    df          : pd.DataFrame,
    feature_cols: Sequence[str],
    mu_ref      : np.ndarray,
    var_ref     : np.ndarray,
    window_size : int = 50,
    step        : int = 1,
    agent_col   : str = "agent_id",
    time_col    : str = "timestep",
) -> pd.DataFrame:
    """Compute per-window KL scores for every agent.

    Returns a long-format DataFrame with one row per (agent, window).

    Returns
    -------
    df_kl : pd.DataFrame  columns=[agent_id, t_start, t_end, kl_score]
    """
    rows = []
    for aid in sorted(df[agent_col].unique()):
        sub = df[df[agent_col] == aid].sort_values(time_col)
        X_a = sub[list(feature_cols)].values.astype(np.float64)
        kl_scores, t_starts = kl_drift_scores(X_a, mu_ref, var_ref, window_size, step)
        for kl, t in zip(kl_scores, t_starts):
            rows.append({
                agent_col: aid,
                "t_start" : int(t),
                "t_end"   : int(t + window_size - 1),
                "kl_score": float(kl),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Window-size sensitivity
# ---------------------------------------------------------------------------
def window_size_sweep(
    X          : np.ndarray,
    y_gt       : np.ndarray,
    mu_ref     : np.ndarray,
    var_ref    : np.ndarray,
    window_grid: Tuple[int, ...] = (20, 50, 100, 200),
) -> pd.DataFrame:
    """AUC of KL scores across window sizes (aligns scores to per-timestep labels).

    Parameters
    ----------
    X           : agent sequence  shape (T, d)
    y_gt        : binary anomaly labels  shape (T,)
    window_grid : window sizes to evaluate

    Returns
    -------
    df : pd.DataFrame  columns=[window_size, auc_roc, auc_pr]
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    rows = []
    for ws in window_grid:
        kl_scores, t_starts = kl_drift_scores(X, mu_ref, var_ref, ws, step=1)

        # Assign window score to every timestep in the window (forward-fill)
        per_timestep = np.zeros(len(X))
        for kl, t in zip(kl_scores, t_starts):
            per_timestep[t : t + ws] = np.maximum(per_timestep[t : t + ws], kl)

        try:
            auc_roc = float(roc_auc_score(y_gt, per_timestep))
            auc_pr  = float(average_precision_score(y_gt, per_timestep))
        except ValueError:
            auc_roc = auc_pr = float("nan")

        rows.append({"window_size": ws, "auc_roc": auc_roc, "auc_pr": auc_pr})

    return pd.DataFrame(rows)