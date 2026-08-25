"""
q_learning.py
=============
LBSM — Reinforcement Learning Layer
------------------------------------
Tabular Q-learning over the discrete LBSM behavioral environment.

Q-learning update rule
----------------------
Q(s, a) ← Q(s, a) + α [r + γ · max_{a'} Q(s', a') − Q(s, a)]

where:
  s      = current discrete obs (latency × entropy grid index)
  a      = action in {push_stable, push_exploratory, do_nothing}
  r      = shaped regime reward
  s'     = next discrete obs after action
  α      = learning rate (may be annealed)
  γ      = discount factor

The Q-table is initialised to zeros and updated in-place throughout
training. The resulting policy is greedy with respect to the final Q-table.

Training produces per-episode statistics:
  - cumulative reward
  - fraction of timesteps in each regime
  - mean Mahalanobis score under NB04 detector (if envelope supplied)
  - manifold displacement (supplied from NB02 UMAP coordinates)

Reference
---------
"Latent Behavioral State Machines: Manifold Geometry of Adaptive Agent Telemetry"
Section 8.1 — Tabular Q-Learning Over Behavioral State Space
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .environment import BehavioralEnv, N_STATES, N_ACTIONS
from ..simulation.behavior_profiles import PROFILE_NAMES


# ---------------------------------------------------------------------------
# Hyperparameter container
# ---------------------------------------------------------------------------

@dataclass
class QLearningConfig:
    """Hyperparameters for tabular Q-learning.

    Attributes
    ----------
    alpha        : learning rate
    gamma        : discount factor
    epsilon_start: initial exploration probability (ε-greedy)
    epsilon_end  : minimum exploration probability after annealing
    epsilon_decay: multiplicative decay applied per episode
    n_episodes   : total training episodes per agent
    seed         : global PRNG seed for exploration noise
    """
    alpha         : float = 0.15
    gamma         : float = 0.95
    epsilon_start : float = 1.0
    epsilon_end   : float = 0.05
    epsilon_decay : float = 0.97
    n_episodes    : int   = 120
    seed          : int   = 42


# ---------------------------------------------------------------------------
# Per-episode statistics
# ---------------------------------------------------------------------------

@dataclass
class EpisodeStats:
    """Statistics collected over a single training episode.

    Attributes
    ----------
    episode          : episode index (0-based)
    total_reward     : sum of shaped rewards over the episode
    regime_fractions : dict regime→fraction of steps spent there
    mean_mah_score   : mean Mahalanobis score (if envelope available)
    epsilon          : ε value used for this episode
    n_steps          : actual number of steps taken
    unstable_frac    : shorthand for regime_fractions["unstable"]
    """
    episode          : int
    total_reward     : float
    regime_fractions : Dict[str, float]
    mean_mah_score   : float
    epsilon          : float
    n_steps          : int

    @property
    def unstable_frac(self) -> float:
        return self.regime_fractions.get("unstable", 0.0)


# ---------------------------------------------------------------------------
# Q-learning agent
# ---------------------------------------------------------------------------

class QLearningAgent:
    """Tabular Q-learning agent operating in :class:`BehavioralEnv`.

    Parameters
    ----------
    env    : BehavioralEnv instance (one env per agent)
    config : QLearningConfig
    """

    def __init__(
        self,
        env    : BehavioralEnv,
        config : QLearningConfig = QLearningConfig(),
    ) -> None:
        self.env    = env
        self.cfg    = config
        self.rng    = np.random.default_rng(config.seed)

        # Q-table: rows = states, cols = actions — initialised optimistically
        self.Q  = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)

        # Training log
        self.episode_log: List[EpisodeStats] = []
        self._epsilon = config.epsilon_start

        # Optional full per-step trajectory capture (one list per episode).
        # Off by default: env.trajectory only ever holds the *last* episode
        # (reset() clears it), so anything wanting per-step history across
        # the whole training run must opt in via train(collect_trajectories=True)
        # and have it snapshotted here before the next reset() wipes it.
        self.trajectory_log: List[List[Dict]] = []

        # Optional: Mahalanobis envelope for scoring mid-training
        self._envelope = None

    # ------------------------------------------------------------------ #
    # Core training loop
    # ------------------------------------------------------------------ #

    def train(
        self,
        healthy_envelope=None,
        verbose: bool = False,
        collect_trajectories: bool = False,
    ) -> List[EpisodeStats]:
        """Run Q-learning for cfg.n_episodes episodes.

        Parameters
        ----------
        healthy_envelope : HealthyEnvelope (from NB04 drift module) or None.
            If supplied, each episode records mean Mahalanobis score so NB05
            can track anomaly score evolution alongside reward.
        verbose : print episode stats every 10 episodes
        collect_trajectories : if True, snapshot ``self.env.trajectory`` after
            every episode into ``self.trajectory_log`` (one list of per-step
            dicts per episode). Off by default — full per-step history across
            120 episodes x 500 steps is ~1000x larger than the episode-level
            stats most callers need. Needed for e.g. NB06's manifold overlay,
            which requires the true visited feature-space points across the
            whole run, not just the last episode.

        Returns
        -------
        episode_log : list of EpisodeStats
        """
        self._envelope = healthy_envelope
        self.episode_log = []
        self.trajectory_log = []

        for ep in range(self.cfg.n_episodes):
            stats = self._run_episode(ep)
            self.episode_log.append(stats)
            if collect_trajectories:
                # list(...) snapshots the current episode's records before
                # the next reset() reassigns self.env._trajectory to a new,
                # empty list — the dicts themselves are never mutated after
                # being appended, so a shallow copy of the list is sufficient.
                self.trajectory_log.append(list(self.env.trajectory))
            self._epsilon = max(
                self.cfg.epsilon_end,
                self._epsilon * self.cfg.epsilon_decay,
            )
            if verbose and ep % 10 == 0:
                print(
                    f"  ep {ep:3d}  reward={stats.total_reward:7.1f}  "
                    f"unstable={stats.unstable_frac:.3f}  "
                    f"ε={stats.epsilon:.3f}"
                )

        return self.episode_log

    def _run_episode(self, ep_idx: int) -> EpisodeStats:
        """Execute one full episode and return statistics."""
        obs    = self.env.reset(rng_seed=self.cfg.seed + ep_idx)
        done   = False
        total_r = 0.0
        regime_counts: Dict[str, int] = {n: 0 for n in PROFILE_NAMES}
        mah_scores_ep: List[float] = []
        n_steps = 0

        while not done:
            action     = self._select_action(obs)
            result     = self.env.step(action)
            next_obs   = result.obs
            reward     = result.reward
            done       = result.done
            hidden     = result.info["hidden_state"]

            # Q-update
            td_target  = reward + self.cfg.gamma * self.Q[next_obs].max()
            td_error   = td_target - self.Q[obs, action]
            self.Q[obs, action] += self.cfg.alpha * td_error

            # Accumulate stats
            total_r   += reward
            regime_counts[hidden] += 1
            n_steps   += 1

            # Optional Mahalanobis scoring
            if self._envelope is not None:
                feats = np.array([
                    result.info["latency"],
                    result.info["entropy"],
                    result.info["reward"],
                    result.info["memory_usage"],
                    result.info["error_rate"],
                    result.info["action_freq"],
                ], dtype=np.float64)
                mah = _mahalanobis_from_envelope(feats, self._envelope)
                mah_scores_ep.append(mah)

            obs = next_obs

        regime_fractions = {
            name: regime_counts[name] / max(n_steps, 1)
            for name in PROFILE_NAMES
        }
        mean_mah = float(np.mean(mah_scores_ep)) if mah_scores_ep else 0.0

        return EpisodeStats(
            episode          = ep_idx,
            total_reward     = total_r,
            regime_fractions = regime_fractions,
            mean_mah_score   = mean_mah,
            epsilon          = self._epsilon,
            n_steps          = n_steps,
        )

    # ------------------------------------------------------------------ #
    # Policy
    # ------------------------------------------------------------------ #

    def _select_action(self, obs: int) -> int:
        """ε-greedy action selection."""
        if self.rng.random() < self._epsilon:
            return int(self.rng.integers(0, N_ACTIONS))
        return int(self.Q[obs].argmax())

    def greedy_action(self, obs: int) -> int:
        """Pure greedy action (no exploration) — used at evaluation time."""
        return int(self.Q[obs].argmax())

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        n_episodes   : int = 10,
        healthy_envelope = None,
    ) -> pd.DataFrame:
        """Evaluate the learned policy (ε=0) and return per-episode stats."""
        saved_eps   = self._epsilon
        self._epsilon = 0.0
        self._envelope = healthy_envelope

        rows = []
        for ep in range(n_episodes):
            stats = self._run_episode(ep_idx=1000 + ep)
            rows.append({
                "episode"        : ep,
                "total_reward"   : stats.total_reward,
                "unstable_frac"  : stats.unstable_frac,
                "stable_frac"    : stats.regime_fractions.get("stable", 0.0),
                "mean_mah_score" : stats.mean_mah_score,
            })
        self._epsilon = saved_eps
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Q-table access helpers
    # ------------------------------------------------------------------ #

    @property
    def policy_map(self) -> np.ndarray:
        """Greedy policy: array of shape (N_STATES,) — action per state."""
        return self.Q.argmax(axis=1)

    @property
    def value_map(self) -> np.ndarray:
        """State value V(s) = max_a Q(s, a), shape (N_STATES,)."""
        return self.Q.max(axis=1)

    def training_dataframe(self) -> pd.DataFrame:
        """Return episode_log as a tidy DataFrame for plotting."""
        rows = []
        for s in self.episode_log:
            row = {
                "episode"      : s.episode,
                "total_reward" : s.total_reward,
                "unstable_frac": s.unstable_frac,
                "mean_mah"     : s.mean_mah_score,
                "epsilon"      : s.epsilon,
                "n_steps"      : s.n_steps,
            }
            for name in PROFILE_NAMES:
                row[f"frac_{name}"] = s.regime_fractions.get(name, 0.0)
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Multi-agent trainer
# ---------------------------------------------------------------------------

def train_agent_pool(
    envs     : list,
    config   : QLearningConfig = QLearningConfig(),
    healthy_envelope = None,
    verbose  : bool = False,
) -> Tuple[List[QLearningAgent], List[pd.DataFrame]]:
    """Train one QLearningAgent per environment.

    Parameters
    ----------
    envs             : list of BehavioralEnv
    config           : shared hyperparameters (seed offset applied per agent)
    healthy_envelope : optional NB04 HealthyEnvelope for Mah. tracking

    Returns
    -------
    agents    : list of trained QLearningAgent
    train_dfs : list of training DataFrame (one per agent)
    """
    agents    = []
    train_dfs = []
    for i, env in enumerate(envs):
        agent_cfg = QLearningConfig(
            alpha         = config.alpha,
            gamma         = config.gamma,
            epsilon_start = config.epsilon_start,
            epsilon_end   = config.epsilon_end,
            epsilon_decay = config.epsilon_decay,
            n_episodes    = config.n_episodes,
            seed          = config.seed + i,
        )
        agent = QLearningAgent(env=env, config=agent_cfg)
        agent.train(healthy_envelope=healthy_envelope, verbose=verbose)
        agents.append(agent)
        train_dfs.append(agent.training_dataframe())
        if verbose:
            print(f"Agent {i:02d} training complete.")

    return agents, train_dfs


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _mahalanobis_from_envelope(
    x        : np.ndarray,
    envelope,                 # HealthyEnvelope from src.drift
) -> float:
    """Compute Mahalanobis distance from x to the healthy envelope."""
    delta = x - envelope.mu
    return float(np.sqrt(np.clip(delta @ envelope.cov_inv @ delta, 0.0, None)))
