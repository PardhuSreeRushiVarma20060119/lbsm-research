"""
regime_shift_analysis.py
========================
LBSM — Drift Detection
-----------------------
Post-hoc regime shift analysis: change-point localisation, shift magnitude
quantification, and detection latency measurement.

After online detectors flag anomalies, this module answers:
- Where in the sequence did each regime shift actually occur?
- How large was the distributional shift?
- How quickly did each detector respond?

Reference
---------
"Latent Behavioral Structure in Low-Dimensional Statistical Manifolds"
Section 7.5 — Regime Shift Characterisation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Change-point extraction from ground truth
# ---------------------------------------------------------------------------
def ground_truth_changepoints(
    df        : pd.DataFrame,
    regime_col: str = "hidden_state",
    agent_col : str = "agent_id",
    time_col  : str = "timestep",
) -> pd.DataFrame:
    """Extract every regime transition timestep from ground-truth labels.

    Returns
    -------
    df_cp : pd.DataFrame  columns=[agent_id, timestep, from_regime, to_regime]
    """
    rows = []
    for aid, grp in df.groupby(agent_col):
        grp_s  = grp.sort_values(time_col).reset_index(drop=True)
        states = grp_s[regime_col].values
        times  = grp_s[time_col].values
        for t in range(1, len(states)):
            if states[t] != states[t - 1]:
                rows.append({
                    agent_col    : aid,
                    time_col     : int(times[t]),
                    "from_regime": states[t - 1],
                    "to_regime"  : states[t],
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Detection latency
# ---------------------------------------------------------------------------
def detection_latency(
    flags        : np.ndarray,
    changepoints : np.ndarray,
    max_lag      : int = 100,
) -> np.ndarray:
    """Compute the number of timesteps between each change-point and the
    first subsequent detection flag.

    Parameters
    ----------
    flags        : binary detection flags  shape (T,)
    changepoints : timestep indices of ground-truth regime transitions
    max_lag      : maximum lag to consider; undetected shifts → max_lag

    Returns
    -------
    latencies : np.ndarray  shape (n_changepoints,)
                Number of steps until detection; max_lag if not detected.
    """
    T         = len(flags)
    latencies = []
    for cp in changepoints:
        cp = int(cp)
        detected = False
        for lag in range(max_lag + 1):
            if cp + lag < T and flags[cp + lag]:
                latencies.append(lag)
                detected = True
                break
        if not detected:
            latencies.append(max_lag)
    return np.array(latencies)


def detection_latency_summary(
    df_cp   : pd.DataFrame,
    df_flags: pd.DataFrame,
    flag_col: str,
    agent_col: str = "agent_id",
    time_col : str = "timestep",
    max_lag  : int = 100,
) -> pd.DataFrame:
    """Per-agent detection latency summary.

    Parameters
    ----------
    df_cp    : change-point DataFrame from :func:`ground_truth_changepoints`
    df_flags : DataFrame with flag column aligned to telemetry order
    flag_col : name of the binary flag column in df_flags
    max_lag  : maximum lag window

    Returns
    -------
    df : pd.DataFrame  columns=[agent_id, mean_latency, median_latency,
                                 n_transitions, n_detected, detection_rate]
    """
    rows = []
    for aid, cp_grp in df_cp.groupby(agent_col):
        agent_flags = df_flags[df_flags[agent_col] == aid].sort_values(time_col)
        flags_arr   = agent_flags[flag_col].values.astype(int)
        cp_times    = cp_grp[time_col].values

        # Re-index change-points to within-agent positions
        t0      = agent_flags[time_col].min()
        cp_pos  = np.clip(cp_times - t0, 0, len(flags_arr) - 1)
        lats    = detection_latency(flags_arr, cp_pos, max_lag)

        n_det   = int((lats < max_lag).sum())
        rows.append({
            agent_col       : aid,
            "mean_latency"  : float(lats.mean()),
            "median_latency": float(np.median(lats)),
            "n_transitions" : len(lats),
            "n_detected"    : n_det,
            "detection_rate": n_det / max(1, len(lats)),
        })
    return pd.DataFrame(rows).set_index(agent_col)


# ---------------------------------------------------------------------------
# Shift magnitude
# ---------------------------------------------------------------------------
def shift_magnitude(
    df        : pd.DataFrame,
    feature_cols: Tuple[str, ...],
    df_cp     : pd.DataFrame,
    window    : int = 20,
    agent_col : str = "agent_id",
    time_col  : str = "timestep",
) -> pd.DataFrame:
    """Quantify the feature-space magnitude of each regime transition.

    For each change-point, computes:
    - Pre-shift mean (window before CP)
    - Post-shift mean (window after CP)
    - L2 distance between pre and post means

    Returns
    -------
    df_mag : pd.DataFrame  columns=[agent_id, timestep, from_regime, to_regime,
                                     shift_magnitude, *feature_deltas]
    """
    feat = list(feature_cols)
    rows = []

    for aid, grp in df.groupby(agent_col):
        grp_s  = grp.sort_values(time_col).reset_index(drop=True)
        X_a    = grp_s[feat].values
        times  = grp_s[time_col].values
        t0     = times[0]

        agent_cps = df_cp[df_cp[agent_col] == aid]
        for _, cp_row in agent_cps.iterrows():
            cp_t = int(cp_row[time_col]) - t0
            pre_start  = max(0, cp_t - window)
            post_end   = min(len(X_a), cp_t + window)

            pre_mean  = X_a[pre_start : cp_t].mean(axis=0)  if cp_t > 0 else X_a[0]
            post_mean = X_a[cp_t : post_end].mean(axis=0)   if cp_t < len(X_a) else X_a[-1]

            delta = post_mean - pre_mean
            mag   = float(np.linalg.norm(delta))

            row = {
                agent_col      : aid,
                time_col       : cp_row[time_col],
                "from_regime"  : cp_row["from_regime"],
                "to_regime"    : cp_row["to_regime"],
                "shift_magnitude": mag,
            }
            for fi, fname in enumerate(feat):
                row[f"delta_{fname}"] = float(delta[fi])
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Transition-type aggregation
# ---------------------------------------------------------------------------
def transition_shift_summary(
    df_mag   : pd.DataFrame,
) -> pd.DataFrame:
    """Mean shift magnitude grouped by (from_regime → to_regime) transition type.

    Returns
    -------
    df : pd.DataFrame  index=(from_regime, to_regime)
         columns=[mean_magnitude, std_magnitude, n_transitions]
    """
    return (
        df_mag.groupby(["from_regime", "to_regime"])["shift_magnitude"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "mean_magnitude", "std": "std_magnitude", "count": "n_transitions"})
    )