"""
reward_dynamics.py
==================
LBSM — Reinforcement Learning Layer
------------------------------------
Reward shaping, regime-conditional reward analysis, and curriculum-based
reward scheduling for the LBSM Q-learning experiments.

The base reward function (defined in environment.py) is:
  +R_HEALTHY  if hidden_state ∈ {stable, exploratory, adaptive}
  -R_UNSTABLE if hidden_state == unstable
  +R_EXIT     one-time bonus on first step leaving unstable

This module provides:
  1. Potential-based reward shaping: augments the base reward with a
     distance-to-healthy-manifold potential, tightening the gradient
     toward healthy behavioral regions.
  2. Reward curriculum: linearly ramps R_UNSTABLE penalty over training
     episodes, starting mild (−0.5) and converging to the full penalty
     (−2.0). This prevents early-training divergence.
  3. Reward decomposition utilities: splits cumulative reward into
     intrinsic (healthy-regime dwell) and extrinsic (exit-bonus) components
     for NB05 analysis.

Reference
---------
"Latent Behavioral State Machines: Manifold Geometry of Adaptive Agent Telemetry"
Section 8.4 — Reward Shaping in Latent Behavioral Space
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..simulation.behavior_profiles import PROFILE_NAMES, BEHAVIOR_PROFILES, TELEMETRY_FEATURES


# ---------------------------------------------------------------------------
# Potential-based reward shaping
# ---------------------------------------------------------------------------

class ManifoldPotential:
    """Φ(x) = −d_M(x, healthy_centroid) — healthy-manifold potential.

    Reward shaping adds γΦ(s') − Φ(s) to the base reward; this is
    *potential-based* and therefore does not alter the optimal policy.

    The potential is the negative Mahalanobis distance from x to the
    pooled healthy regime centroid (mean of stable, exploratory, adaptive).

    Parameters
    ----------
    shaping_coeff : multiplier applied to the shaping term (λ in paper)
    """

    def __init__(self, shaping_coeff: float = 0.3) -> None:
        self.shaping_coeff = shaping_coeff

        # Pool healthy-regime centroids (exclude unstable)
        healthy_names = [n for n in PROFILE_NAMES if n != "unstable"]
        means_stack   = np.stack(
            [BEHAVIOR_PROFILES[n].means for n in healthy_names]
        )
        stds_stack    = np.stack(
            [BEHAVIOR_PROFILES[n].stds  for n in healthy_names]
        )
        self._mu      = means_stack.mean(axis=0)
        self._sigma   = stds_stack.mean(axis=0)

    def potential(self, telemetry_vec: np.ndarray) -> float:
        """Compute Φ(x) = −||( x − μ_healthy ) / σ_healthy||₂."""
        delta = (telemetry_vec - self._mu) / (self._sigma + 1e-9)
        dist  = float(np.linalg.norm(delta))
        return -dist * self.shaping_coeff

    def shaping_bonus(
        self,
        prev_telemetry: np.ndarray,
        next_telemetry: np.ndarray,
        gamma          : float = 0.95,
    ) -> float:
        """Compute potential-based shaping: γΦ(s') − Φ(s)."""
        return gamma * self.potential(next_telemetry) - self.potential(prev_telemetry)


# ---------------------------------------------------------------------------
# Reward curriculum
# ---------------------------------------------------------------------------

@dataclass
class RewardCurriculum:
    """Linearly ramps the unstable penalty over training.

    Attributes
    ----------
    r_unstable_start : initial (mild) penalty for being in unstable
    r_unstable_final : final (full) penalty
    n_warmup_episodes: number of episodes over which ramp occurs
    r_healthy        : reward for healthy regimes (fixed)
    r_exit           : one-time exit bonus (fixed)
    """
    r_unstable_start  : float = -0.5
    r_unstable_final  : float = -2.0
    n_warmup_episodes : int   = 40
    r_healthy         : float =  1.0
    r_exit            : float =  3.0

    def unstable_penalty(self, episode: int) -> float:
        """Return the unstable penalty at a given training episode."""
        frac = min(episode / max(self.n_warmup_episodes, 1), 1.0)
        return self.r_unstable_start + frac * (
            self.r_unstable_final - self.r_unstable_start
        )

    def compute_reward(
        self,
        hidden_state     : str,
        was_in_unstable  : bool,
        episode          : int,
    ) -> Tuple[float, bool]:
        """Return (shaped_reward, exit_bonus_triggered) with curriculum penalty."""
        if hidden_state == "unstable":
            return self.unstable_penalty(episode), False
        else:
            exit_bonus = was_in_unstable
            r = self.r_healthy + (self.r_exit if exit_bonus else 0.0)
            return r, exit_bonus


# ---------------------------------------------------------------------------
# Reward decomposition
# ---------------------------------------------------------------------------

def decompose_episode_rewards(
    trajectory: List[Dict],
) -> Dict[str, float]:
    """Split cumulative episode reward into components.

    Parameters
    ----------
    trajectory : list of step-dicts with keys 'reward', 'hidden_state'

    Returns
    -------
    dict with keys:
      'total'          : sum of all rewards
      'healthy_dwell'  : reward from healthy-regime timesteps (> 0 steps)
      'unstable_penalty': penalty from unstable timesteps
      'exit_bonus'     : inferred exit bonuses (reward > R_HEALTHY in a
                          healthy step following an unstable step)
    """
    from .environment import R_HEALTHY, R_UNSTABLE, R_EXIT

    total       = 0.0
    healthy_r   = 0.0
    unstable_r  = 0.0
    exit_r      = 0.0

    prev_hidden = None
    for rec in trajectory:
        r      = rec.get("reward", 0.0)
        hidden = rec.get("hidden_state", "stable")

        total += r
        if hidden == "unstable":
            unstable_r += r
        else:
            # Detect exit bonus: reward > R_HEALTHY implies bonus was added
            if prev_hidden == "unstable" and r > R_HEALTHY + 0.5:
                exit_r   += R_EXIT
                healthy_r += R_HEALTHY
            else:
                healthy_r += r
        prev_hidden = hidden

    return {
        "total"           : total,
        "healthy_dwell"   : healthy_r,
        "unstable_penalty": unstable_r,
        "exit_bonus"      : exit_r,
    }


def reward_by_regime(
    trajectory: List[Dict],
) -> pd.DataFrame:
    """Per-regime mean reward from a trajectory.

    Returns
    -------
    DataFrame with columns: regime, mean_reward, std_reward, n_steps
    """
    records: Dict[str, List[float]] = {n: [] for n in PROFILE_NAMES}
    for rec in trajectory:
        hidden = rec.get("hidden_state", "stable")
        r      = rec.get("reward", 0.0)
        if hidden in records:
            records[hidden].append(r)

    rows = []
    for regime, rewards in records.items():
        if rewards:
            rows.append({
                "regime"     : regime,
                "mean_reward": float(np.mean(rewards)),
                "std_reward" : float(np.std(rewards)),
                "n_steps"    : len(rewards),
            })
    return pd.DataFrame(rows)
