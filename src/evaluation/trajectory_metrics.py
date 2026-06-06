"""
trajectory_metrics.py
=====================
LBSM — Evaluation
------------------
Trajectory-level evaluation metrics for NB02 (manifold) and NB04 (drift):
path length, tortuosity, manifold speed, and arc-length by regime.
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd


def path_length(traj: np.ndarray) -> float:
    """Total arc length of a 2-D or n-D trajectory."""
    return float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())


def displacement(traj: np.ndarray) -> float:
    """Straight-line distance from start to end of trajectory."""
    return float(np.linalg.norm(traj[-1] - traj[0]))


def tortuosity(traj: np.ndarray) -> float:
    """Path length / displacement (1 = straight line, higher = more winding)."""
    d = displacement(traj)
    return path_length(traj) / (d + 1e-9)


def mean_speed(traj: np.ndarray) -> float:
    """Mean step-wise displacement (manifold speed)."""
    return float(np.linalg.norm(np.diff(traj, axis=0), axis=1).mean())


def trajectory_summary(
    trajectories: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compute path_length, displacement, tortuosity, mean_speed for each agent.

    Parameters
    ----------
    trajectories : {agent_id: np.ndarray shape (T, 2)}

    Returns
    -------
    df : pd.DataFrame  index=agent_id
    """
    rows = []
    for aid, traj in trajectories.items():
        rows.append({
            "agent_id"   : aid,
            "path_length": path_length(traj),
            "displacement": displacement(traj),
            "tortuosity" : tortuosity(traj),
            "mean_speed" : mean_speed(traj),
        })
    return pd.DataFrame(rows).set_index("agent_id")