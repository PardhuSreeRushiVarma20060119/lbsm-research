"""
trajectory_geometry.py
=======================
Latent Behavioral State Machine (LBSM) — Manifold Analysis
-----------------------------------------------------------
Temporal trajectory analysis of agent behavioral paths in the
learned low-dimensional manifold.

This is the module that makes LBSM *distinct* from ordinary
cluster analysis. Because telemetry is temporal, the embedding
is not merely a scatter of points but a family of continuous
curves — one per agent — that trace behavioral trajectories
through latent space.

This module quantifies:
  - Path length per agent (how far does the agent travel?)
  - Path tortuosity (ratio of path length to displacement)
  - Regime transition geometry (where in the manifold do transitions occur?)
  - Arc continuity (do trajectories stay smooth across transitions?)
  - Manifold velocity (rate of change of embedding coordinates over time)

Reference
---------
"Latent Behavioral State Machines: Manifold Geometry of Adaptive Agent Telemetry"
Section 5.5 — Temporal Behavioral Geometry
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class TrajectoryStats:
    """Per-agent trajectory statistics in the manifold.

    Attributes
    ----------
    agent_id    : str
    path_length : float  — total arc length in embedded space
    displacement: float  — straight-line distance start → end
    tortuosity  : float  — path_length / displacement  (1 = straight)
    mean_speed  : float  — mean step-to-step displacement
    max_speed   : float  — maximum single-step displacement (proxy for shock)
    n_transitions: int   — number of regime changes
    """

    agent_id     : str
    path_length  : float
    displacement : float
    tortuosity   : float
    mean_speed   : float
    max_speed    : float
    n_transitions: int


# ---------------------------------------------------------------------------
# Core: extract per-agent trajectories from embedding
# ---------------------------------------------------------------------------
def extract_agent_trajectories(
    embedding : np.ndarray,
    df_full   : pd.DataFrame,
    agent_ids : Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Map embedding coordinates back to per-agent temporal sequences.

    Parameters
    ----------
    embedding  : np.ndarray  shape (N_total, 2)  — rows match df_full row order
    df_full    : pd.DataFrame  — the full telemetry DataFrame (with agent_id, timestep)
    agent_ids  : which agents to extract (None → all agents)

    Returns
    -------
    trajectories : dict  agent_id → np.ndarray shape (T, 2)
                   sorted by timestep for each agent
    """
    df = df_full.copy()
    df["_emb_x"] = embedding[:, 0]
    df["_emb_y"] = embedding[:, 1]

    all_agents = df["agent_id"].unique()
    if agent_ids is not None:
        all_agents = [a for a in agent_ids if a in set(all_agents)]

    trajectories = {}
    for aid in all_agents:
        sub = df[df["agent_id"] == aid].sort_values("timestep")
        trajectories[aid] = sub[["_emb_x", "_emb_y"]].values

    return trajectories


# ---------------------------------------------------------------------------
# Per-agent statistics
# ---------------------------------------------------------------------------
def compute_trajectory_stats(
    trajectories : Dict[str, np.ndarray],
    df_full      : pd.DataFrame,
) -> pd.DataFrame:
    """Compute :class:`TrajectoryStats` for every agent.

    Parameters
    ----------
    trajectories : output of :func:`extract_agent_trajectories`
    df_full      : full telemetry DataFrame (for regime transition counts)

    Returns
    -------
    df : pd.DataFrame  index = agent_id,  columns from TrajectoryStats fields
    """
    rows = []
    for aid, traj in trajectories.items():
        # Step-wise displacements
        steps = np.linalg.norm(np.diff(traj, axis=0), axis=1)

        path_length  = float(steps.sum())
        displacement = float(np.linalg.norm(traj[-1] - traj[0]))
        tortuosity   = path_length / (displacement + 1e-9)
        mean_speed   = float(steps.mean()) if len(steps) > 0 else 0.0
        max_speed    = float(steps.max())  if len(steps) > 0 else 0.0

        # Count transitions from the DataFrame
        agent_df = df_full[df_full["agent_id"] == aid].sort_values("timestep")
        hs       = agent_df["hidden_state"].values
        n_trans  = int((hs[:-1] != hs[1:]).sum())

        rows.append({
            "agent_id"    : aid,
            "path_length" : path_length,
            "displacement": displacement,
            "tortuosity"  : tortuosity,
            "mean_speed"  : mean_speed,
            "max_speed"   : max_speed,
            "n_transitions": n_trans,
        })

    return pd.DataFrame(rows).set_index("agent_id")


# ---------------------------------------------------------------------------
# Transition point geometry
# ---------------------------------------------------------------------------
def transition_embedding_coords(
    embedding: np.ndarray,
    df_full  : pd.DataFrame,
) -> pd.DataFrame:
    """Return the embedding coordinates at each regime transition.

    Useful for visualising *where* in latent space agents switch regimes —
    do transitions cluster near the regime boundary?

    Returns
    -------
    df : pd.DataFrame  columns = [agent_id, timestep, from_state, to_state,
                                   emb_x, emb_y]
    """
    df = df_full.copy().reset_index(drop=True)
    df["_emb_x"] = embedding[:, 0]
    df["_emb_y"] = embedding[:, 1]

    rows = []
    for aid, grp in df.groupby("agent_id"):
        grp = grp.sort_values("timestep").reset_index(drop=True)
        states = grp["hidden_state"].values
        for t in range(1, len(states)):
            if states[t] != states[t - 1]:
                rows.append({
                    "agent_id"  : aid,
                    "timestep"  : int(grp["timestep"].iloc[t]),
                    "from_state": states[t - 1],
                    "to_state"  : states[t],
                    "emb_x"     : float(grp["_emb_x"].iloc[t]),
                    "emb_y"     : float(grp["_emb_y"].iloc[t]),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Manifold velocity (temporal rate-of-change)
# ---------------------------------------------------------------------------
def manifold_velocity(
    trajectories : Dict[str, np.ndarray],
    df_full      : pd.DataFrame,
    window       : int = 5,
) -> pd.DataFrame:
    """Compute instantaneous manifold speed per timestep for each agent.

    Speed = ||emb(t+1) - emb(t)||₂ (Euclidean displacement per step).
    A rolling mean is applied to smooth AR(1) noise.

    Parameters
    ----------
    trajectories : dict from :func:`extract_agent_trajectories`
    df_full      : full telemetry DataFrame
    window       : rolling mean window for smoothing

    Returns
    -------
    df : pd.DataFrame  columns = [agent_id, timestep, speed, speed_smooth,
                                   hidden_state]
    """
    rows = []
    for aid, traj in trajectories.items():
        agent_df = df_full[df_full["agent_id"] == aid].sort_values("timestep").reset_index(drop=True)
        steps    = np.linalg.norm(np.diff(traj, axis=0), axis=1)

        # Align: speed at t corresponds to step t → t+1
        T = len(steps)
        states   = agent_df["hidden_state"].values[:T]
        timesteps= agent_df["timestep"].values[:T]

        speed_smooth = pd.Series(steps).rolling(window, center=True, min_periods=1).mean().values

        for t in range(T):
            rows.append({
                "agent_id"    : aid,
                "timestep"    : int(timesteps[t]),
                "speed"       : float(steps[t]),
                "speed_smooth": float(speed_smooth[t]),
                "hidden_state": states[t],
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Regime arc length
# ---------------------------------------------------------------------------
def regime_arc_statistics(
    trajectories: Dict[str, np.ndarray],
    df_full     : pd.DataFrame,
    profile_names: Tuple[str, ...],
) -> pd.DataFrame:
    """Compute mean trajectory speed and path length *within* each regime.

    Allows comparison: does the unstable regime produce faster-moving
    trajectories than the stable regime?

    Returns
    -------
    df : pd.DataFrame  index = regime, columns = [mean_speed, std_speed,
                                                    mean_arc_length, n_segments]
    """
    speed_by_regime: Dict[str, List[float]] = {name: [] for name in profile_names}

    for aid, traj in trajectories.items():
        agent_df = df_full[df_full["agent_id"] == aid].sort_values("timestep").reset_index(drop=True)
        states   = agent_df["hidden_state"].values
        steps    = np.linalg.norm(np.diff(traj, axis=0), axis=1)

        for t, spd in enumerate(steps):
            regime = states[t]
            if regime in speed_by_regime:
                speed_by_regime[regime].append(spd)

    rows = []
    for name in profile_names:
        spds = np.array(speed_by_regime[name])
        if len(spds) == 0:
            continue
        rows.append({
            "regime"         : name,
            "mean_speed"     : float(spds.mean()),
            "std_speed"      : float(spds.std()),
            "mean_arc_length": float(spds.sum() / max(1,
                df_full[df_full["hidden_state"] == name]["agent_id"].nunique()
            )),
            "n_steps"        : len(spds),
        })

    return pd.DataFrame(rows).set_index("regime")