"""
normalization.py
================
LBSM — Telemetry Processing
-----------------------------
Feature normalization utilities: z-score, min-max, and robust scaling.

The LBSM simulation produces features in heterogeneous physical units
(latency in ms, entropy in bits, memory in MB). Normalization is required
before any distance-based computation (Mahalanobis, EWMA, KL divergence).

Reference
---------
"Latent Behavioral State Machines: Manifold Geometry of Adaptive Agent Telemetry"
Section 4.2 — Feature Standardisation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Scaler containers
# ---------------------------------------------------------------------------
@dataclass
class ZScoreParams:
    """Parameters for z-score standardisation: X_z = (X − μ) / σ."""
    mean : np.ndarray   # shape (d,)
    std  : np.ndarray   # shape (d,)


@dataclass
class MinMaxParams:
    """Parameters for min-max scaling: X_s = (X − min) / (max − min)."""
    min_ : np.ndarray   # shape (d,)
    max_ : np.ndarray   # shape (d,)


# ---------------------------------------------------------------------------
# Z-score
# ---------------------------------------------------------------------------
def fit_zscore(X: np.ndarray) -> ZScoreParams:
    """Compute z-score parameters from a training matrix."""
    return ZScoreParams(
        mean = X.mean(axis=0),
        std  = X.std(axis=0, ddof=1) + 1e-9,
    )


def apply_zscore(X: np.ndarray, params: ZScoreParams) -> np.ndarray:
    """Apply pre-fitted z-score transform."""
    return (X - params.mean) / params.std


def zscore_matrix(X: np.ndarray) -> Tuple[np.ndarray, ZScoreParams]:
    """Fit and apply z-score in one step.

    Returns
    -------
    X_z    : standardised matrix  shape (N, d)
    params : ZScoreParams for later use on new data
    """
    params = fit_zscore(X)
    return apply_zscore(X, params), params


# ---------------------------------------------------------------------------
# Min-max
# ---------------------------------------------------------------------------
def fit_minmax(X: np.ndarray) -> MinMaxParams:
    """Compute min-max parameters from a training matrix."""
    return MinMaxParams(
        min_ = X.min(axis=0),
        max_ = X.max(axis=0),
    )


def apply_minmax(X: np.ndarray, params: MinMaxParams) -> np.ndarray:
    """Apply pre-fitted min-max scaling to [0, 1]."""
    rng = params.max_ - params.min_ + 1e-9
    return (X - params.min_) / rng


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Scale a 1-D anomaly score array to [0, 1] using observed min/max."""
    mn, mx = scores.min(), scores.max()
    return (scores - mn) / (mx - mn + 1e-9)


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------
def zscore_dataframe(
    df          : pd.DataFrame,
    feature_cols: Sequence[str],
    suffix      : str = "_z",
) -> pd.DataFrame:
    """Add z-scored columns to a DataFrame in-place (copy).

    Parameters
    ----------
    df           : input DataFrame
    feature_cols : columns to standardise
    suffix       : appended to column names for the new columns

    Returns
    -------
    df_out : copy of df with additional ``{feat}{suffix}`` columns
    """
    df_out = df.copy()
    for feat in feature_cols:
        mu  = df_out[feat].mean()
        std = df_out[feat].std(ddof=1) + 1e-9
        df_out[f"{feat}{suffix}"] = (df_out[feat] - mu) / std
    return df_out