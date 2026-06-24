"""
adaptation_dynamics.py
======================
LBSM — Reinforcement Learning Layer
------------------------------------
Analysis of how RL training reshapes agent trajectories in the latent
behavioral manifold characterised in NB01–NB04.

This module is the primary interface between the RL training results and the
geometric/statistical infrastructure built in earlier notebooks. It answers
the core NB05 research questions:

  Q1 — Manifold alignment:
       Do RL learning trajectories follow existing manifold directions (NB02)
       or open new regions of UMAP space?

  Q2 — HMM complexity reduction:
       Does training reduce regime-switching entropy and HMM transition
       complexity (NB03)?

  Q3 — Anomaly score evolution:
       Do early-training behavioral innovations register as anomalies under
       the NB04 detector, and do they stabilise into new clusters over time?

Key functions
-------------
  manifold_trajectory_stats  — displacement, speed, tortuosity in UMAP space
  cluster_migration_table    — regime-dwell fractions before / during / after
  anomaly_score_evolution    — rolling mean Mah. score over training episodes
  transition_entropy_series  — empirical H(s'|s) per episode
  umap_episode_centroids     — episode-wise UMAP centroids for trajectory plot
  regime_novelty_score       — fraction of UMAP cells not previously visited

Reference
---------
"Latent Behavioral State Machines: Manifold Geometry of Adaptive Agent Telemetry"
Section 8.6 — Latent Manifold Analysis of RL Adaptation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


# ---------------------------------------------------------------------------
# Manifold trajectory statistics
# ---------------------------------------------------------------------------

def manifold_trajectory_stats(
    episode_trajectories : List[List[Dict]],
    umap_embedding       : Optional[np.ndarray] = None,
    feature_cols         : Tuple[str, ...] = (
        "latency", "entropy", "reward",
        "memory_usage", "error_rate", "action_freq",
    ),
) -> pd.DataFrame:
    """Compute trajectory statistics in feature space (or UMAP space).

    For each episode, computes:
      - path_length : sum of step-wise L2 displacements
      - displacement: start-to-end distance
      - tortuosity  : path_length / (displacement + ε)
      - mean_speed  : path_length / n_steps
      - unstable_frac: fraction of steps in unstable regime

    Parameters
    ----------
    episode_trajectories : list (one per episode) of trajectory dicts.
        Each dict must have keys matching feature_cols and 'hidden_state'.
    umap_embedding : (N_total_steps, 2) optional — if provided, distances
        are computed in UMAP space rather than raw feature space.
    feature_cols : which features to use for distance (raw space fallback)

    Returns
    -------
    DataFrame with one row per episode.
    """
    rows = []
    for ep, traj in enumerate(episode_trajectories):
        if len(traj) < 2:
            continue

        hidden_states = [r.get("hidden_state", "stable") for r in traj]
        unstable_frac = sum(1 for h in hidden_states if h == "unstable") / len(hidden_states)

        # Build trajectory matrix
        if umap_embedding is not None and len(umap_embedding) >= len(traj):
            # Use UMAP coords — assumes traj indices align with embedding rows
            coords = umap_embedding[:len(traj)]
        else:
            coords = np.array([
                [r.get(f, 0.0) for f in feature_cols]
                for r in traj
            ], dtype=float)

        # Step-wise displacements
        diffs     = np.diff(coords, axis=0)
        step_dist = np.linalg.norm(diffs, axis=1)
        path_len  = float(step_dist.sum())
        displace  = float(np.linalg.norm(coords[-1] - coords[0]))
        tort      = path_len / (displace + 1e-6)
        speed     = path_len / max(len(traj) - 1, 1)

        rows.append({
            "episode"      : ep,
            "path_length"  : path_len,
            "displacement" : displace,
            "tortuosity"   : tort,
            "mean_speed"   : speed,
            "unstable_frac": unstable_frac,
            "n_steps"      : len(traj),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cluster migration (regime dwell-time phasing)
# ---------------------------------------------------------------------------

def cluster_migration_table(
    episode_trajectories : List[List[Dict]],
    phase_boundaries     : Tuple[float, float] = (0.33, 0.67),
) -> pd.DataFrame:
    """Regime dwell fractions split into early / mid / late training phases.

    Parameters
    ----------
    episode_trajectories : list of trajectory-dicts (one per episode)
    phase_boundaries     : (early_end, late_start) as fractions of n_episodes

    Returns
    -------
    DataFrame: regime × phase with mean dwell fraction.
    """
    from ..simulation.behavior_profiles import PROFILE_NAMES
    n_ep  = len(episode_trajectories)
    early_end  = int(phase_boundaries[0] * n_ep)
    late_start = int(phase_boundaries[1] * n_ep)

    phases = {
        "early" : episode_trajectories[:early_end],
        "mid"   : episode_trajectories[early_end:late_start],
        "late"  : episode_trajectories[late_start:],
    }

    rows = []
    for phase_name, phase_trajs in phases.items():
        if not phase_trajs:
            continue
        # Aggregate regime counts across all episodes in this phase
        total_counts = {r: 0 for r in PROFILE_NAMES}
        total_steps  = 0
        for traj in phase_trajs:
            for rec in traj:
                h = rec.get("hidden_state", "stable")
                if h in total_counts:
                    total_counts[h] += 1
                total_steps += 1
        for regime, cnt in total_counts.items():
            rows.append({
                "phase"   : phase_name,
                "regime"  : regime,
                "dwell"   : cnt / max(total_steps, 1),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Anomaly score evolution
# ---------------------------------------------------------------------------

def anomaly_score_evolution(
    episode_mah_means : np.ndarray,
    smooth_window     : int = 5,
) -> pd.DataFrame:
    """Build a DataFrame tracking mean Mahalanobis score across training.

    Parameters
    ----------
    episode_mah_means : per-episode mean Mah. score (from training log)
    smooth_window     : rolling mean window

    Returns
    -------
    DataFrame with columns: episode, mah_mean, mah_smooth
    """
    from .reward_tracking import smooth
    n_ep = len(episode_mah_means)
    smoothed = smooth(episode_mah_means, smooth_window, "same")
    return pd.DataFrame({
        "episode"   : np.arange(n_ep),
        "mah_mean"  : episode_mah_means,
        "mah_smooth": smoothed,
    })


# ---------------------------------------------------------------------------
# HMM transition entropy reduction
# ---------------------------------------------------------------------------

def transition_entropy_series(
    episode_trajectories : List[List[Dict]],
) -> pd.DataFrame:
    """Compute empirical H(s'|s) per episode as a proxy for HMM complexity.

    H(s'|s) = Σ_s p(s) · H(s'|s=s)
             = Σ_s p(s) · (-Σ_{s'} P(s'|s) log P(s'|s))

    A lower value means the agent's transition behaviour is more deterministic
    (less regime-switching entropy), consistent with NB03's evidence that
    learning reduces the effective complexity of the latent Markov chain.

    Parameters
    ----------
    episode_trajectories : list of trajectory-dicts

    Returns
    -------
    DataFrame: episode, transition_entropy
    """
    from ..simulation.behavior_profiles import PROFILE_NAMES
    k      = len(PROFILE_NAMES)
    idx_map = {n: i for i, n in enumerate(PROFILE_NAMES)}

    rows = []
    for ep, traj in enumerate(episode_trajectories):
        states = [r.get("hidden_state", "stable") for r in traj]
        if len(states) < 2:
            continue

        # Empirical transition counts
        counts = np.zeros((k, k), dtype=float)
        for s_from, s_to in zip(states[:-1], states[1:]):
            i = idx_map.get(s_from, 0)
            j = idx_map.get(s_to,   0)
            counts[i, j] += 1

        # Normalise rows → empirical transition probs
        row_totals = counts.sum(axis=1, keepdims=True)
        T_emp  = counts / np.where(row_totals > 0, row_totals, 1.0)

        # Stationary distribution approximation
        state_counts = counts.sum(axis=1)
        pi = state_counts / (state_counts.sum() + 1e-9)

        # Conditional entropy H(s'|s) = Σ_s π(s) H(row_s)
        H_rows = np.array([scipy_entropy(T_emp[i] + 1e-12) for i in range(k)])
        H_cond = float((pi * H_rows).sum())

        rows.append({"episode": ep, "transition_entropy": H_cond})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# UMAP episode centroids for trajectory visualisation
# ---------------------------------------------------------------------------

def umap_episode_centroids(
    episode_trajectories : List[List[Dict]],
    X_umap               : np.ndarray,
    global_df            : pd.DataFrame,
    agent_id             : str,
) -> pd.DataFrame:
    """Compute mean UMAP position per training episode for one agent.

    Parameters
    ----------
    episode_trajectories : trajectory dicts with 'timestep' key
    X_umap               : (N_total_obs, 2) global UMAP embedding from NB02
    global_df            : the sorted df from which X_umap was derived
    agent_id             : which agent's rows to look up in global_df

    Returns
    -------
    DataFrame: episode, umap1_mean, umap2_mean, unstable_frac
    """
    # Build timestep → umap index lookup for this agent
    agent_mask  = global_df["agent_id"] == agent_id
    agent_df    = global_df[agent_mask].reset_index()
    ts_to_umap  = {
        int(row["timestep"]): int(row["index"])
        for _, row in agent_df.iterrows()
    }

    rows = []
    for ep, traj in enumerate(episode_trajectories):
        umaps = []
        hidden_states = []
        for rec in traj:
            ts  = rec.get("timestep", rec.get("episode_step", -1))
            idx = ts_to_umap.get(ts)
            if idx is not None and idx < len(X_umap):
                umaps.append(X_umap[idx])
            hidden_states.append(rec.get("hidden_state", "stable"))

        if not umaps:
            continue
        umaps_arr     = np.array(umaps)
        unstable_frac = sum(1 for h in hidden_states if h == "unstable") / len(hidden_states)
        rows.append({
            "episode"      : ep,
            "umap1_mean"   : float(umaps_arr[:, 0].mean()),
            "umap2_mean"   : float(umaps_arr[:, 1].mean()),
            "unstable_frac": unstable_frac,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Regime novelty score
# ---------------------------------------------------------------------------

def regime_novelty_score(
    episode_trajectories : List[List[Dict]],
    n_bins               : int = 20,
    feature_pair         : Tuple[str, str] = ("latency", "entropy"),
) -> pd.DataFrame:
    """Fraction of (latency, entropy) grid cells newly visited per episode.

    Tracks whether RL exploration opens new regions of behavioral space
    (novelty > 0) or stays within previously visited territory (novelty ≈ 0).

    Parameters
    ----------
    episode_trajectories : list of trajectory dicts
    n_bins               : number of bins per feature axis for the grid
    feature_pair         : which two features to use for the grid

    Returns
    -------
    DataFrame: episode, novelty_frac, cumulative_coverage
    """
    f1, f2  = feature_pair
    visited = set()
    rows    = []

    # Determine global bounds across all trajectories
    all_f1 = [r.get(f1, 0.0) for traj in episode_trajectories for r in traj]
    all_f2 = [r.get(f2, 0.0) for traj in episode_trajectories for r in traj]
    lo1, hi1 = min(all_f1), max(all_f1)
    lo2, hi2 = min(all_f2), max(all_f2)

    def to_cell(v1, v2):
        i = min(int((v1 - lo1) / (hi1 - lo1 + 1e-9) * n_bins), n_bins - 1)
        j = min(int((v2 - lo2) / (hi2 - lo2 + 1e-9) * n_bins), n_bins - 1)
        return (i, j)

    total_cells = n_bins * n_bins

    for ep, traj in enumerate(episode_trajectories):
        new_cells = set()
        for rec in traj:
            cell = to_cell(rec.get(f1, lo1), rec.get(f2, lo2))
            if cell not in visited:
                new_cells.add(cell)
        novelty = len(new_cells) / total_cells
        visited.update(new_cells)
        rows.append({
            "episode"            : ep,
            "novelty_frac"       : novelty,
            "cumulative_coverage": len(visited) / total_cells,
        })

    return pd.DataFrame(rows)
