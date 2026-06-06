"""
windowing.py
============
LBSM — Telemetry Processing
-----------------------------
Sliding-window and rolling-window utilities for temporal telemetry streams.

Used by the drift detection layer (src/drift/) to convert per-timestep
observations into window-level feature summaries for distributional comparison.

Reference
---------
"Latent Behavioral State Machines: Manifold Geometry of Adaptive Agent Telemetry"
Section 7.1 — Online Drift Detection: Window Construction
"""

from __future__ import annotations

from typing import Generator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core sliding window
# ---------------------------------------------------------------------------
def sliding_windows(
    X          : np.ndarray,
    window_size: int,
    step       : int = 1,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Yield (start_index, window_array) tuples over a 2-D array.

    Parameters
    ----------
    X           : np.ndarray  shape (T, d)
    window_size : number of timesteps per window
    step        : stride between window starts

    Yields
    ------
    (t_start, window) where window has shape (window_size, d)
    """
    T = len(X)
    for t in range(0, T - window_size + 1, step):
        yield t, X[t : t + window_size]


def window_statistics(
    X          : np.ndarray,
    window_size: int,
    step       : int = 1,
    feature_names: Optional[Tuple[str, ...]] = None,
) -> pd.DataFrame:
    """Compute per-window summary statistics (mean, std, min, max) for each feature.

    Parameters
    ----------
    X             : np.ndarray  shape (T, d)
    window_size   : window length in timesteps
    step          : stride
    feature_names : column names for the output DataFrame

    Returns
    -------
    df : pd.DataFrame  shape (n_windows, 4*d)
         columns = [feat_mean, feat_std, feat_min, feat_max, ...]
    """
    d = X.shape[1]
    if feature_names is None:
        feature_names = tuple(f"f{i}" for i in range(d))

    rows = []
    for t_start, win in sliding_windows(X, window_size, step):
        row = {"t_start": t_start, "t_end": t_start + window_size - 1}
        for fi, fname in enumerate(feature_names):
            col = win[:, fi]
            row[f"{fname}_mean"] = float(col.mean())
            row[f"{fname}_std"]  = float(col.std())
            row[f"{fname}_min"]  = float(col.min())
            row[f"{fname}_max"]  = float(col.max())
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reference vs test window split
# ---------------------------------------------------------------------------
def reference_test_split(
    X              : np.ndarray,
    reference_size : int,
    test_size      : int,
    step           : int = 1,
) -> Generator[Tuple[np.ndarray, np.ndarray, int], None, None]:
    """Yield (reference_window, test_window, test_start) tuples.

    The reference window immediately precedes the test window.
    Used for KL-divergence and distributional drift detection.

    Parameters
    ----------
    X              : np.ndarray  shape (T, d)
    reference_size : length of the reference (baseline) window
    test_size      : length of the test (current) window
    step           : stride between test window starts

    Yields
    ------
    (ref_window, test_window, test_start_idx)
    """
    T     = len(X)
    start = reference_size
    while start + test_size <= T:
        ref  = X[start - reference_size : start]
        test = X[start : start + test_size]
        yield ref, test, start
        start += step


# ---------------------------------------------------------------------------
# Per-agent windowing
# ---------------------------------------------------------------------------
def per_agent_windows(
    df          : pd.DataFrame,
    feature_cols: Sequence[str],
    window_size : int,
    step        : int = 1,
    agent_col   : str = "agent_id",
    time_col    : str = "timestep",
) -> pd.DataFrame:
    """Apply sliding windows to each agent's sequence independently.

    Returns a DataFrame where each row is one window from one agent,
    with ``agent_id``, ``t_start``, ``t_end``, and per-feature statistics.

    Parameters
    ----------
    df           : full telemetry DataFrame
    feature_cols : features to summarise
    window_size  : window length in timesteps
    step         : stride
    agent_col    : column identifying agents
    time_col     : column for temporal ordering

    Returns
    -------
    df_windows : pd.DataFrame
    """
    feat_names = tuple(feature_cols)
    all_rows   = []

    for aid in sorted(df[agent_col].unique()):
        sub = df[df[agent_col] == aid].sort_values(time_col)
        X_a = sub[list(feat_names)].values.astype(np.float64)

        for t_start, win in sliding_windows(X_a, window_size, step):
            row = {
                agent_col: aid,
                "t_start" : int(sub["timestep"].iloc[t_start]),
                "t_end"   : int(sub["timestep"].iloc[t_start + window_size - 1]),
            }
            for fi, fname in enumerate(feat_names):
                col = win[:, fi]
                row[f"{fname}_mean"] = float(col.mean())
                row[f"{fname}_std"]  = float(col.std())
            all_rows.append(row)

    return pd.DataFrame(all_rows)