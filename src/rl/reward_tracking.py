"""
reward_tracking.py
==================
LBSM — Reinforcement Learning Layer
------------------------------------
Episode-level reward tracking, smoothed learning curves, regime-dwell
time evolution, and convergence diagnostics for NB05.

This module is purely analytical — it operates on the episode logs
produced by QLearningAgent.train() and produces the DataFrames and
summary statistics reported in NB05 §§ 5–7.

Key outputs for NB05
--------------------
  - learning_curve_df   : smoothed reward and unstable_frac per episode
  - convergence_episode : episode at which unstable_frac drops below threshold
  - dwell_evolution_df  : per-episode regime dwell-time fractions (all 4 regimes)
  - regime_delta_table  : before/after training change in dwell times

Reference
---------
"Latent Behavioral Structure in Low-Dimensional Statistical Manifolds"
Section 8.5 — Training Convergence & Behavioral Stabilisation
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Learning curve smoothing
# ---------------------------------------------------------------------------

def smooth(
    values   : np.ndarray,
    window   : int = 10,
    mode     : str = "valid",
) -> np.ndarray:
    """Uniform moving-average smoothing.

    Parameters
    ----------
    values : 1-D array
    window : smoothing window length
    mode   : 'valid' (shorten output) or 'same' (pad with edge values)

    Returns
    -------
    smoothed : 1-D array
    """
    kernel   = np.ones(window) / window
    if mode == "same":
        # Pad edges with boundary values
        pad_l    = window // 2
        pad_r    = window - pad_l - 1
        padded   = np.concatenate([
            np.full(pad_l, values[0]),
            values,
            np.full(pad_r, values[-1]),
        ])
        return np.convolve(padded, kernel, mode="valid")
    return np.convolve(values, kernel, mode=mode)


def learning_curve_df(
    train_df     : pd.DataFrame,
    smooth_window: int = 10,
) -> pd.DataFrame:
    """Build a smoothed learning-curve DataFrame from a training log.

    Parameters
    ----------
    train_df : DataFrame produced by QLearningAgent.training_dataframe()
    smooth_window : rolling mean window

    Returns
    -------
    DataFrame with columns:
      episode, total_reward, unstable_frac, frac_stable, frac_exploratory,
      frac_adaptive, mean_mah, epsilon,
      reward_smooth, unstable_smooth
    """
    df = train_df.copy().reset_index(drop=True)
    df["reward_smooth"]   = smooth(df["total_reward"].values,   smooth_window, "same")
    df["unstable_smooth"] = smooth(df["unstable_frac"].values, smooth_window, "same")
    return df


def pool_learning_curves(
    train_dfs    : List[pd.DataFrame],
    smooth_window: int = 10,
) -> pd.DataFrame:
    """Average learning curves across a pool of agents.

    Returns
    -------
    DataFrame with episode-indexed mean ± std across agents for key metrics.
    """
    # Align on episode index
    stacked_reward   = np.stack([df["total_reward"].values   for df in train_dfs])
    stacked_unstable = np.stack([df["unstable_frac"].values  for df in train_dfs])
    stacked_mah      = np.stack([df["mean_mah"].values       for df in train_dfs])

    n_episodes = stacked_reward.shape[1]
    episodes   = np.arange(n_episodes)

    mean_r  = stacked_reward.mean(axis=0)
    std_r   = stacked_reward.std(axis=0)
    mean_u  = stacked_unstable.mean(axis=0)
    std_u   = stacked_unstable.std(axis=0)
    mean_m  = stacked_mah.mean(axis=0)
    std_m   = stacked_mah.std(axis=0)

    return pd.DataFrame({
        "episode"       : episodes,
        "mean_reward"   : smooth(mean_r,  smooth_window, "same"),
        "std_reward"    : std_r,
        "mean_unstable" : smooth(mean_u,  smooth_window, "same"),
        "std_unstable"  : std_u,
        "mean_mah"      : smooth(mean_m,  smooth_window, "same"),
        "std_mah"       : std_m,
    })


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------

def convergence_episode(
    unstable_fracs  : np.ndarray,
    threshold       : float = 0.10,
    n_consecutive   : int   = 5,
) -> Optional[int]:
    """Find the first episode after which unstable_frac stays below threshold.

    Parameters
    ----------
    unstable_fracs : per-episode unstable fraction (1-D array)
    threshold      : target unstable fraction (default 10%)
    n_consecutive  : number of consecutive episodes below threshold required

    Returns
    -------
    episode index (int) or None if convergence not achieved
    """
    below = unstable_fracs < threshold
    for i in range(len(below) - n_consecutive + 1):
        if below[i : i + n_consecutive].all():
            return int(i)
    return None


def convergence_table(
    train_dfs: List[pd.DataFrame],
    threshold: float = 0.10,
) -> pd.DataFrame:
    """Per-agent convergence episode and final-10-episode mean unstable_frac."""
    rows = []
    for i, df in enumerate(train_dfs):
        u       = df["unstable_frac"].values
        ep      = convergence_episode(u, threshold)
        rows.append({
            "agent"                : f"agent_{i:04d}",
            "convergence_episode"  : ep if ep is not None else len(u),
            "converged"            : ep is not None,
            "final_unstable_frac"  : float(u[-10:].mean()),
            "initial_unstable_frac": float(u[:10].mean()),
            "reduction"            : float(u[:10].mean() - u[-10:].mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Regime dwell-time evolution
# ---------------------------------------------------------------------------

def dwell_evolution_df(
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return episode-indexed dwell fractions for all four regimes.

    Expects columns frac_stable, frac_exploratory, frac_adaptive,
    frac_unstable in train_df.
    """
    cols = ["episode"] + [f"frac_{r}" for r in
                          ["stable", "exploratory", "adaptive", "unstable"]]
    available = [c for c in cols if c in train_df.columns]
    return train_df[available].copy()


def regime_delta_table(
    train_df : pd.DataFrame,
    n_window : int = 10,
) -> pd.DataFrame:
    """Compare first-n_window vs last-n_window episode mean regime fractions.

    Returns
    -------
    DataFrame with columns: regime, initial_frac, final_frac, delta, pct_change
    """
    rows = []
    for regime in ["stable", "exploratory", "adaptive", "unstable"]:
        col = f"frac_{regime}"
        if col not in train_df.columns:
            continue
        initial = float(train_df[col].values[:n_window].mean())
        final   = float(train_df[col].values[-n_window:].mean())
        delta   = final - initial
        pct     = (delta / (initial + 1e-9)) * 100.0
        rows.append({
            "regime"      : regime,
            "initial_frac": initial,
            "final_frac"  : final,
            "delta"       : delta,
            "pct_change"  : pct,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pool-level regime summary
# ---------------------------------------------------------------------------

def pool_regime_summary(
    train_dfs : List[pd.DataFrame],
    n_window  : int = 10,
) -> pd.DataFrame:
    """Aggregate regime_delta_table across a pool of agents.

    Returns mean ± std of initial_frac, final_frac, delta for each regime.
    """
    all_deltas = {r: [] for r in ["stable", "exploratory", "adaptive", "unstable"]}
    all_initial = {r: [] for r in all_deltas}
    all_final   = {r: [] for r in all_deltas}

    for df in train_dfs:
        dt = regime_delta_table(df, n_window)
        for _, row in dt.iterrows():
            r = row["regime"]
            all_initial[r].append(row["initial_frac"])
            all_final[r].append(row["final_frac"])
            all_deltas[r].append(row["delta"])

    rows = []
    for regime in ["stable", "exploratory", "adaptive", "unstable"]:
        rows.append({
            "regime"       : regime,
            "initial_mean" : float(np.mean(all_initial[regime])),
            "initial_std"  : float(np.std(all_initial[regime])),
            "final_mean"   : float(np.mean(all_final[regime])),
            "final_std"    : float(np.std(all_final[regime])),
            "delta_mean"   : float(np.mean(all_deltas[regime])),
            "delta_std"    : float(np.std(all_deltas[regime])),
        })
    return pd.DataFrame(rows)
