"""
exploration.py
==============
LBSM — Reinforcement Learning Layer
------------------------------------
Exploration schedules and curiosity-driven bonuses for the LBSM Q-learning
experiments.

Three exploration strategies are implemented:
  1. ε-greedy with geometric decay (the primary strategy used in NB05)
  2. ε-greedy with linear decay
  3. Count-based curiosity bonus (adds an exploration term to the reward
     proportional to 1/sqrt(visit_count(s, a)), encouraging the agent to
     visit under-explored (latency, entropy) grid cells)

The curiosity bonus is used in the ablation study (NB05 §10) to test
whether extra exploration in the healthy-regime corridor matters for
convergence speed.

Reference
---------
"Latent Behavioral State Machines: Manifold Geometry of Adaptive Agent Telemetry"
Section 8.3 — Exploration in Latent Behavioral Space
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .environment import N_STATES, N_ACTIONS


# ---------------------------------------------------------------------------
# Epsilon schedules
# ---------------------------------------------------------------------------

@dataclass
class EpsilonSchedule:
    """ε-greedy exploration schedule.

    Attributes
    ----------
    epsilon_start : initial ε
    epsilon_end   : minimum ε (floor)
    decay_mode    : 'geometric' or 'linear'
    decay_param   : for geometric: multiplicative factor per episode
                    for linear   : total number of decay steps
    """
    epsilon_start : float = 1.0
    epsilon_end   : float = 0.05
    decay_mode    : Literal["geometric", "linear"] = "geometric"
    decay_param   : float = 0.97   # geometric decay factor / linear n_steps

    def __post_init__(self) -> None:
        self._current = self.epsilon_start
        self._step    = 0

    def step(self) -> float:
        """Advance the schedule by one episode and return new ε."""
        if self.decay_mode == "geometric":
            self._current = max(
                self.epsilon_end,
                self._current * self.decay_param,
            )
        else:  # linear
            frac = min(self._step / max(self.decay_param, 1), 1.0)
            self._current = self.epsilon_start + frac * (
                self.epsilon_end - self.epsilon_start
            )
        self._step += 1
        return self._current

    @property
    def epsilon(self) -> float:
        return self._current

    def reset(self) -> None:
        self._current = self.epsilon_start
        self._step    = 0


def geometric_epsilon(
    episode      : int,
    epsilon_start: float = 1.0,
    epsilon_end  : float = 0.05,
    decay        : float = 0.97,
) -> float:
    """Compute ε at a given episode index without state (functional form)."""
    eps = epsilon_start * (decay ** episode)
    return float(max(epsilon_end, eps))


def linear_epsilon(
    episode      : int,
    n_episodes   : int,
    epsilon_start: float = 1.0,
    epsilon_end  : float = 0.05,
) -> float:
    """Linearly anneal ε from start to end over n_episodes."""
    frac = min(episode / max(n_episodes - 1, 1), 1.0)
    return float(epsilon_start + frac * (epsilon_end - epsilon_start))


# ---------------------------------------------------------------------------
# Count-based curiosity bonus
# ---------------------------------------------------------------------------

class CuriosityBonus:
    """Count-based exploration bonus: β / sqrt(N(s,a) + 1).

    The bonus is added to the extrinsic reward to encourage the agent to
    visit rare (state, action) pairs. In the LBSM context this means the
    agent is pushed toward unexplored regions of the (latency, entropy) grid,
    which may correspond to boundary zones between behavioral regimes.

    Parameters
    ----------
    beta         : bonus magnitude (typical range 0.1 – 1.0)
    decay_factor : multiplicative decay applied to beta each episode
                   (set to 1.0 for no decay)
    """

    def __init__(
        self,
        beta         : float = 0.5,
        decay_factor : float = 0.99,
    ) -> None:
        self.beta_init   = beta
        self.beta        = beta
        self.decay_factor = decay_factor
        # Visit count table
        self._counts     = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)

    def bonus(self, state: int, action: int) -> float:
        """Return the curiosity bonus for (state, action) before updating counts."""
        n = self._counts[state, action]
        return float(self.beta / np.sqrt(n + 1.0))

    def update(self, state: int, action: int) -> None:
        """Increment visit count for (state, action)."""
        self._counts[state, action] += 1.0

    def step_episode(self) -> None:
        """Decay beta at the end of each training episode."""
        self.beta = max(self.beta * self.decay_factor, 0.01)

    def reset(self) -> None:
        self._counts[:] = 0.0
        self.beta = self.beta_init

    @property
    def visit_counts(self) -> np.ndarray:
        """Return (N_STATES, N_ACTIONS) visit count array."""
        return self._counts.copy()

    @property
    def state_visit_counts(self) -> np.ndarray:
        """Return (N_STATES,) total visit counts per state."""
        return self._counts.sum(axis=1)


# ---------------------------------------------------------------------------
# Exploration summary for NB05 reporting
# ---------------------------------------------------------------------------

def exploration_coverage(
    visit_counts: np.ndarray,
    threshold   : int = 1,
) -> float:
    """Fraction of (state, action) pairs visited at least *threshold* times.

    Parameters
    ----------
    visit_counts : (N_STATES, N_ACTIONS) int array
    threshold    : minimum visit count to be considered 'explored'

    Returns
    -------
    coverage : float in [0, 1]
    """
    return float((visit_counts >= threshold).mean())


def state_coverage(
    visit_counts: np.ndarray,
    threshold   : int = 1,
) -> float:
    """Fraction of states visited at least *threshold* times (any action)."""
    state_totals = visit_counts.sum(axis=1)
    return float((state_totals >= threshold).mean())
