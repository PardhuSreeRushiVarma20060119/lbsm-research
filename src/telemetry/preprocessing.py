"""
preprocessing.py
================
LBSM — Telemetry Processing
-----------------------------
Data cleaning and preparation pipeline for raw telemetry DataFrames.
Handles clipping, missing values, type enforcement, and train/test split.

Reference
---------
"Latent Behavioral State Machines: Manifold Geometry of Adaptive Agent Telemetry"
Section 4.1 — Data Preparation
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Physical bounds (from behavior_profiles.py)
# ---------------------------------------------------------------------------
FEATURE_BOUNDS = {
    "latency"     : (0.0,    2000.0),
    "entropy"     : (0.0,      10.0),
    "reward"      : (-10.0,   100.0),
    "memory_usage": (0.0,    4096.0),
    "error_rate"  : (0.0,       1.0),
    "action_freq" : (0.0,     200.0),
}


def clip_features(
    df          : pd.DataFrame,
    feature_cols: Sequence[str],
    bounds      : Optional[dict] = None,
) -> pd.DataFrame:
    """Clip feature values to their physically valid ranges.

    Parameters
    ----------
    df           : telemetry DataFrame
    feature_cols : features to clip
    bounds       : dict {feature: (lower, upper)}; defaults to FEATURE_BOUNDS

    Returns
    -------
    df_clipped : copy with values clipped
    """
    if bounds is None:
        bounds = FEATURE_BOUNDS
    df_out = df.copy()
    for feat in feature_cols:
        if feat in bounds:
            lo, hi = bounds[feat]
            df_out[feat] = df_out[feat].clip(lower=lo, upper=hi)
    return df_out


def enforce_dtypes(
    df          : pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Cast telemetry feature columns to float32 for memory efficiency."""
    df_out = df.copy()
    for feat in feature_cols:
        df_out[feat] = df_out[feat].astype(np.float32)
    return df_out


def drop_incomplete(
    df       : pd.DataFrame,
    min_steps: int = 100,
    agent_col: str = "agent_id",
) -> pd.DataFrame:
    """Remove agents with fewer than ``min_steps`` observations.

    Guards against incomplete sequences that could corrupt HMM or
    drift-detection window statistics.
    """
    counts  = df.groupby(agent_col).size()
    valid   = counts[counts >= min_steps].index
    return df[df[agent_col].isin(valid)].copy()


# ---------------------------------------------------------------------------
# Train / test split (temporal)
# ---------------------------------------------------------------------------
def temporal_train_test_split(
    df        : pd.DataFrame,
    test_frac : float = 0.20,
    agent_col : str = "agent_id",
    time_col  : str = "timestep",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split each agent's sequence into train (early) and test (late).

    Temporal split respects the time ordering — no future leakage.
    The last ``test_frac`` fraction of each agent's timesteps forms the test set.

    Returns
    -------
    df_train, df_test : pd.DataFrame
    """
    train_parts, test_parts = [], []

    for aid, grp in df.groupby(agent_col):
        grp_sorted = grp.sort_values(time_col)
        n          = len(grp_sorted)
        split_idx  = int(n * (1 - test_frac))
        train_parts.append(grp_sorted.iloc[:split_idx])
        test_parts.append(grp_sorted.iloc[split_idx:])

    df_train = pd.concat(train_parts, ignore_index=True)
    df_test  = pd.concat(test_parts,  ignore_index=True)
    return df_train, df_test


# ---------------------------------------------------------------------------
# Feature matrix extraction
# ---------------------------------------------------------------------------
def to_feature_matrix(
    df          : pd.DataFrame,
    feature_cols: Sequence[str],
    z_scored    : bool = False,
    dtype       : type = np.float64,
) -> np.ndarray:
    """Extract a 2-D feature matrix from a DataFrame.

    Parameters
    ----------
    df           : telemetry DataFrame
    feature_cols : features to extract
    z_scored     : if True use ``{feat}_z`` columns instead
    dtype        : output numpy dtype

    Returns
    -------
    X : np.ndarray  shape (N, d)
    """
    cols = [f"{f}_z" for f in feature_cols] if z_scored else list(feature_cols)
    return df[cols].values.astype(dtype)