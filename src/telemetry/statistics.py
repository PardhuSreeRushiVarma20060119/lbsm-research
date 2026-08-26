"""
statistics.py
=============
LBSM — Telemetry Processing
-----------------------------
Descriptive and inferential statistics on the telemetry dataset.
Produces the per-regime and per-agent summaries used across notebooks.

Reference
---------
"Latent Behavioral Structure in Low-Dimensional Statistical Manifolds"
Section 4.3 — Dataset Statistics
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Per-regime statistics
# ---------------------------------------------------------------------------
def regime_summary(
    df          : pd.DataFrame,
    feature_cols: Sequence[str],
    regime_col  : str = "hidden_state",
    regime_order: Optional[Tuple[str, ...]] = None,
) -> pd.DataFrame:
    """Compute mean ± std per regime for every telemetry feature.

    Returns
    -------
    df_stats : pd.DataFrame  MultiIndex (regime, statistic) × feature
    """
    grp = df.groupby(regime_col)[list(feature_cols)]
    stats = grp.agg(["mean", "std", "min", "max", "median"])
    if regime_order:
        stats = stats.reindex(list(regime_order))
    return stats


# ---------------------------------------------------------------------------
# Regime separability (Fisher criterion)
# ---------------------------------------------------------------------------
def fisher_separability(
    df          : pd.DataFrame,
    feature_cols: Sequence[str],
    regime_col  : str = "hidden_state",
) -> pd.Series:
    """Univariate Fisher separability ratio per feature.

    F_k = σ²_between / σ²_within

    Higher → more discriminative feature.

    Returns
    -------
    ratios : pd.Series  feature → F_k, sorted descending
    """
    groups = [
        df[df[regime_col] == r][list(feature_cols)].values
        for r in df[regime_col].unique()
    ]
    grand_mean = df[list(feature_cols)].mean().values

    between_var = np.zeros(len(feature_cols))
    within_var  = np.zeros(len(feature_cols))

    for g in groups:
        between_var += len(g) * (g.mean(0) - grand_mean) ** 2
    between_var /= len(df)

    for g in groups:
        within_var += ((g - g.mean(0)) ** 2).sum(0)
    within_var /= len(df)

    ratios = between_var / (within_var + 1e-12)
    return pd.Series(ratios, index=list(feature_cols)).sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Anomaly rate breakdown
# ---------------------------------------------------------------------------
def anomaly_rate_by_regime(
    df         : pd.DataFrame,
    anomaly_col: str = "is_anomaly",
    regime_col : str = "hidden_state",
) -> pd.DataFrame:
    """Fraction of anomalous timesteps per regime.

    Returns
    -------
    df : pd.DataFrame  columns=[regime, n_obs, n_anomaly, anomaly_rate]
    """
    rows = []
    for regime, grp in df.groupby(regime_col):
        n     = len(grp)
        n_anom = int(grp[anomaly_col].sum())
        rows.append({
            "regime"      : regime,
            "n_obs"       : n,
            "n_anomaly"   : n_anom,
            "anomaly_rate": n_anom / n if n > 0 else 0.0,
        })
    return pd.DataFrame(rows).set_index("regime")


# ---------------------------------------------------------------------------
# Per-agent statistics
# ---------------------------------------------------------------------------
def per_agent_summary(
    df          : pd.DataFrame,
    feature_cols: Sequence[str],
    agent_col   : str = "agent_id",
    regime_col  : str = "hidden_state",
) -> pd.DataFrame:
    """Per-agent mean reward, error rate, latency, and dominant regime.

    Returns
    -------
    df : pd.DataFrame  index=agent_id
    """
    rows = []
    for aid, grp in df.groupby(agent_col):
        row = {"agent_id": aid}
        for feat in feature_cols:
            row[f"mean_{feat}"] = float(grp[feat].mean())
        row["dominant_regime"] = grp[regime_col].mode()[0]
        row["n_transitions"]   = int(
            (grp.sort_values("timestep")[regime_col] !=
             grp.sort_values("timestep")[regime_col].shift()).sum()
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("agent_id")


# ---------------------------------------------------------------------------
# Pairwise regime Bhattacharyya distance
# ---------------------------------------------------------------------------
def bhattacharyya_distance(
    df          : pd.DataFrame,
    feature_cols: Sequence[str],
    regime_col  : str = "hidden_state",
) -> pd.DataFrame:
    """Pairwise Bhattacharyya distance between regime distributions.

    Assumes Gaussian distributions per regime (diagonal covariance).
    BC = 0.25·ln(0.25·(σ²_p/σ²_q + σ²_q/σ²_p + 2)) + 0.25·(μ_p−μ_q)²/(σ²_p+σ²_q)

    Returns
    -------
    D : pd.DataFrame  shape (n_regimes, n_regimes)
    """
    regimes = sorted(df[regime_col].unique())
    params  = {}
    for r in regimes:
        sub = df[df[regime_col] == r][list(feature_cols)].values
        params[r] = (sub.mean(0), sub.var(0) + 1e-9)

    n = len(regimes)
    D = np.zeros((n, n))

    for i, ri in enumerate(regimes):
        mu_i, var_i = params[ri]
        for j, rj in enumerate(regimes):
            mu_j, var_j = params[rj]
            # Scalar Bhattacharyya per feature, then sum
            t1 = 0.25 * np.log(0.25 * (var_i/var_j + var_j/var_i + 2.0))
            t2 = 0.25 * (mu_i - mu_j)**2 / (var_i + var_j)
            D[i, j] = float((t1 + t2).sum())

    return pd.DataFrame(D, index=regimes, columns=regimes)