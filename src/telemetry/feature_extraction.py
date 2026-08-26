"""
feature_extraction.py
=====================
LBSM — Telemetry Processing
-----------------------------
Feature engineering: derives additional diagnostic signals from raw
telemetry columns that improve anomaly detection and drift sensitivity.

Reference
---------
"Latent Behavioral Structure in Low-Dimensional Statistical Manifolds"
Section 4.4 — Augmented Feature Space
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------
def rolling_mean(
    df          : pd.DataFrame,
    feature_cols: Sequence[str],
    window      : int = 10,
    agent_col   : str = "agent_id",
    time_col    : str = "timestep",
    suffix      : str = "_rmean",
) -> pd.DataFrame:
    """Per-agent rolling mean for each feature.

    Parameters
    ----------
    df           : telemetry DataFrame
    feature_cols : features to smooth
    window       : rolling window size (timesteps)
    suffix       : column name suffix for new columns

    Returns
    -------
    df_out : copy of df with additional rolling mean columns
    """
    df_out = df.copy().sort_values([agent_col, time_col])
    for feat in feature_cols:
        df_out[f"{feat}{suffix}"] = (
            df_out.groupby(agent_col)[feat]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
    return df_out


def rolling_std(
    df          : pd.DataFrame,
    feature_cols: Sequence[str],
    window      : int = 10,
    agent_col   : str = "agent_id",
    time_col    : str = "timestep",
    suffix      : str = "_rstd",
) -> pd.DataFrame:
    """Per-agent rolling standard deviation for each feature."""
    df_out = df.copy().sort_values([agent_col, time_col])
    for feat in feature_cols:
        df_out[f"{feat}{suffix}"] = (
            df_out.groupby(agent_col)[feat]
            .transform(lambda x: x.rolling(window, min_periods=2).std().fillna(0))
        )
    return df_out


# ---------------------------------------------------------------------------
# Temporal difference features
# ---------------------------------------------------------------------------
def temporal_diff(
    df          : pd.DataFrame,
    feature_cols: Sequence[str],
    lag         : int = 1,
    agent_col   : str = "agent_id",
    time_col    : str = "timestep",
    suffix      : str = "_diff",
) -> pd.DataFrame:
    """First-order temporal difference (velocity) per agent per feature.

    Δx_t = x_t − x_{t−lag}

    High |Δx_t| signals rapid behavioral change — useful for detecting
    regime transitions online.
    """
    df_out = df.copy().sort_values([agent_col, time_col])
    for feat in feature_cols:
        df_out[f"{feat}{suffix}"] = (
            df_out.groupby(agent_col)[feat]
            .transform(lambda x: x.diff(lag).fillna(0))
        )
    return df_out


# ---------------------------------------------------------------------------
# Composite health score
# ---------------------------------------------------------------------------
def composite_health_score(
    df         : pd.DataFrame,
    latency_col: str = "latency",
    error_col  : str = "error_rate",
    reward_col : str = "reward",
    entropy_col: str = "entropy",
) -> pd.Series:
    """Compute a heuristic composite health score in [0, 1].

    Higher score → healthier (stable-like) behaviour.

    Score = 1 − (w_lat·lat_norm + w_err·err_norm + w_ent·ent_norm − w_rew·rew_norm)

    where each term is min-max normalised across the dataset.
    """
    def norm(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)

    lat_n = norm(df[latency_col])
    err_n = norm(df[error_col])
    ent_n = norm(df[entropy_col])
    rew_n = norm(df[reward_col])

    raw = 0.30 * lat_n + 0.35 * err_n + 0.20 * ent_n - 0.15 * rew_n
    return 1.0 - norm(raw)


# ---------------------------------------------------------------------------
# Phase-space augmentation
# ---------------------------------------------------------------------------
def augment_phase_space(
    df          : pd.DataFrame,
    feature_cols: Sequence[str],
    agent_col   : str = "agent_id",
    time_col    : str = "timestep",
) -> pd.DataFrame:
    """Augment the feature matrix with first-order temporal differences.

    Produces X_t = [z_t, Δz_t] for each feature z, doubling the
    feature dimensionality. This captures velocity in feature space
    and improves detection of sharp-manoeuvre anomaly types.

    Returns
    -------
    df_out : DataFrame with additional ``{feat}_diff`` columns
    """
    return temporal_diff(df, feature_cols, lag=1,
                         agent_col=agent_col, time_col=time_col)