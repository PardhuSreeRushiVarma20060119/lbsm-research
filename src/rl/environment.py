"""
environment.py
==============
LBSM — Reinforcement Learning Layer
------------------------------------
Wraps AdaptiveAgent as a Gymnasium-style (step / reset) environment for
tabular Q-learning experiments in NB05.

The RL environment answers a research question: can a learning controller
*reduce* the fraction of time agents spend in the anomalous (unstable)
regime by adaptively biasing the agent's Markov transition dynamics?

Design choices
--------------
State space
    The observable state is a discrete (grid_latency × grid_entropy) index
    derived by binning the two most discriminative telemetry features
    (latency — Fisher rank 3; entropy — rank 2 from NB01 §3.3).
    Grid size: n_lat × n_ent = 10 × 10 → 100 discrete states.
    Discretisation bounds are set from the healthy-regime 1–99th percentile
    so that the unstable regime maps to the high-index corner of the grid.

Action space
    Three actions that nudge the *current row* of the agent's Markov
    transition matrix toward a target regime profile:
        0 — push_stable       (increase P(→ stable), decrease P(→ unstable))
        1 — push_exploratory  (increase P(→ exploratory), slight ↑ adaptability)
        2 — do_nothing        (identity; transition matrix unchanged this step)

    The nudge strength δ controls how aggressively the action reshapes the
    transition matrix; it decays over training (curriculum schedule).

Reward structure
    +R_healthy if hidden_state ∈ {stable, exploratory, adaptive}
    -R_unstable if hidden_state == unstable
    +R_exit bonus once per unstable-episode when the agent leaves unstable
    Shaped to produce a strong gradient away from the unstable attractor.

Episode termination
    Each episode runs for exactly N_STEPS_PER_EPISODE timesteps with no
    early termination (ensures full manifold coverage).

Reference
---------
"Latent Behavioral State Machines: Manifold Geometry of Adaptive :Agent Telemetry"
Section 8 — Reinforcement Learning Over Latent Behavioral Space
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..simulation.agent import AdaptiveAgent, DEFAULT_TRANSITION_MATRIX
from ..simulation.behavior_profiles import PROFILE_NAMES, BEHAVIOR_PROFILES, TELEMETRY_FEATURES


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Discrete action labels
ACTION_PUSH_STABLE      = 0
ACTION_PUSH_EXPLORATORY = 1
ACTION_DO_NOTHING       = 2
N_ACTIONS               = 3

# State grid dimensions
N_GRID_LATENCY = 10
N_GRID_ENTROPY = 10
N_STATES       = N_GRID_LATENCY * N_GRID_ENTROPY  # 100

# Feature indices within TELEMETRY_FEATURES (latency=0, entropy=1)
IDX_LATENCY = 0
IDX_ENTROPY = 1

# Regime index map (must match PROFILE_NAMES order)
_REGIME_IDX = {name: i for i, name in enumerate(PROFILE_NAMES)}

# Binning bounds derived from 1st–99th percentile of healthy regimes
# (stable, exploratory, adaptive profiles — excludes unstable tails)
_LATENCY_BOUNDS = (30.0, 280.0)   # ms — covers stable…exploratory range + buffer
_ENTROPY_BOUNDS  = (0.3,  5.0)    # bits

# Reward magnitudes
R_HEALTHY  =  1.0
R_UNSTABLE = -2.0
R_EXIT     =  3.0   # one-time bonus for leaving unstable

# Nudge strength (base value; may be annealed externally)
# The nudge is cumulative: each push action is applied relative to the *current*
# T_current row, so repeated same-direction actions compound. Validity (no negative
# or >1 probabilities) is guaranteed by the floor-and-renormalise step in
# _nudge_transition, not by resetting to the default matrix every call.
DELTA_BASE = 0.02

# Episode length
N_STEPS_PER_EPISODE = 500


# ---------------------------------------------------------------------------
# Dataclass for step returns
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Single-step environment return.

    Attributes
    ----------
    obs      : int   — discrete state index in [0, N_STATES)
    reward   : float — shaped reward signal
    done     : bool  — True when episode terminates
    info     : dict  — diagnostic metadata (hidden_state, telemetry, …)
    """
    obs      : int
    reward   : float
    done     : bool
    info     : Dict


# ---------------------------------------------------------------------------
# Discretisation helpers
# ---------------------------------------------------------------------------

def _digitise(value: float, lo: float, hi: float, n_bins: int) -> int:
    """Clip-and-bin a scalar into [0, n_bins - 1]."""
    clipped = float(np.clip(value, lo, hi))
    fraction = (clipped - lo) / (hi - lo)
    return min(int(fraction * n_bins), n_bins - 1)


def obs_to_grid(latency: float, entropy: float) -> int:
    """Map (latency, entropy) to a flat discrete state index.

    Returns
    -------
    state_idx : int in [0, N_STATES)
    """
    i_lat = _digitise(latency, *_LATENCY_BOUNDS, N_GRID_LATENCY)
    i_ent = _digitise(entropy, *_ENTROPY_BOUNDS,  N_GRID_ENTROPY)
    return i_lat * N_GRID_ENTROPY + i_ent


def grid_to_coords(state_idx: int) -> Tuple[int, int]:
    """Inverse of obs_to_grid — returns (latency_bin, entropy_bin)."""
    return divmod(state_idx, N_GRID_ENTROPY)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class BehavioralEnv:
    """Markov Decision Process over agent behavioral regimes.

    The environment wraps a single :class:`AdaptiveAgent` and exposes a
    ``step / reset`` API compatible with standard tabular RL loops.

    Parameters
    ----------
    agent_id     : identifier forwarded to the wrapped :class:`AdaptiveAgent`
    rng_seed     : reproducibility seed
    delta        : nudge strength for transition-matrix modulation actions
    n_steps      : episode length (timesteps per episode)
    record_traj  : if True, store full telemetry history for analysis
    """

    def __init__(
        self,
        agent_id  : str  = "rl_agent",
        rng_seed  : int  = 42,
        delta     : float = DELTA_BASE,
        n_steps   : int   = N_STEPS_PER_EPISODE,
        record_traj: bool = True,
    ) -> None:
        self.agent_id   = agent_id
        self.rng_seed   = rng_seed
        self.delta      = delta
        self.n_steps    = n_steps
        self.record_traj = record_traj

        # Underlying agent (reset on each episode)
        self._agent = AdaptiveAgent(
            agent_id         = agent_id,
            initial_state    = "stable",
            transition_matrix = DEFAULT_TRANSITION_MATRIX.copy(),
            rng_seed         = rng_seed,
        )
        # Mutable transition matrix (reset each episode)
        self._T_current: np.ndarray = DEFAULT_TRANSITION_MATRIX.copy()

        # Episode state
        self._t           : int   = 0
        self._in_unstable  : bool  = False  # for exit-bonus tracking

        # Trajectory buffer (telemetry rows for NB05 analysis)
        self._trajectory  : List[Dict] = []

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _emit_record(self, timestep: int) -> Dict:
        """Emit telemetry for the agent's *current* hidden state, with no transition.

        Mirrors the emission half of :meth:`AdaptiveAgent.step`, but deliberately
        does not call ``_transition()`` — that happens separately in :meth:`step`,
        *before* this is called, so that the nudge applied for a given action and
        the state whose telemetry ends up in the returned observation/reward are
        the same state the action was chosen in reaction to.
        """
        telemetry_vec = self._agent._emit()
        record: Dict = {
            "agent_id": self._agent.agent_id,
            "timestep": timestep,
            "hidden_state": self._agent.current_state,
        }
        for feat, val in zip(TELEMETRY_FEATURES, telemetry_vec):
            record[feat] = float(val)
        self._agent._history.append(record)
        return record

    # ------------------------------------------------------------------ #
    # Gym-style API
    # ------------------------------------------------------------------ #

    def reset(self, rng_seed: Optional[int] = None) -> int:
        """Reset the environment to the start of a new episode.

        Returns
        -------
        obs : int — initial discrete state
        """
        seed = rng_seed if rng_seed is not None else self.rng_seed
        self._agent = AdaptiveAgent(
            agent_id          = self.agent_id,
            initial_state     = "stable",
            transition_matrix = DEFAULT_TRANSITION_MATRIX.copy(),
            rng_seed          = seed,
        )
        self._T_current  = DEFAULT_TRANSITION_MATRIX.copy()
        self._t          = 0
        self._in_unstable = False
        if self.record_traj:
            self._trajectory = []

        # Emit first observation WITHOUT transitioning yet — this is the state
        # the first call to step() will nudge (see _emit_record docstring).
        rec = self._emit_record(timestep=0)
        obs = obs_to_grid(rec["latency"], rec["entropy"])
        if self.record_traj:
            self._trajectory.append({**rec, "action": -1, "reward": 0.0,
                                      "obs": obs, "episode_step": 0})
        self._t = 1
        return obs

    def step(self, action: int) -> StepResult:
        """Apply action, advance agent one timestep, return (obs, r, done, info).

        Parameters
        ----------
        action : int in {0, 1, 2}

        Returns
        -------
        StepResult
        """
        # 1. Modulate transition matrix for the state we are CURRENTLY in — i.e.
        # the same state that was just reported as `obs` and that this action was
        # chosen in reaction to. Cumulative: nudges self._T_current (not the
        # pristine default), so repeated same-direction actions compound.
        if action != ACTION_DO_NOTHING:
            self._T_current = _nudge_transition(
                self._T_current,
                from_state = self._agent._state_idx,
                action     = action,
                delta      = self.delta,
            )
            self._agent._T = self._T_current

        # 2. Transition using the (possibly just-nudged) row, THEN emit telemetry
        # for the resulting state. This ordering — transition before emit — is
        # what makes this call's reward/obs reflect the state the action caused,
        # instead of the state the agent was already in before the action ran.
        self._agent._transition()
        rec = self._emit_record(self._t)
        hidden = rec["hidden_state"]

        # 3. Compute reward
        reward, exit_bonus = _compute_reward(hidden, self._in_unstable)
        was_in_unstable    = self._in_unstable
        self._in_unstable  = (hidden == "unstable")
        if exit_bonus:
            reward += R_EXIT

        # 4. Build observation
        obs = obs_to_grid(rec["latency"], rec["entropy"])

        # 5. Done?
        self._t += 1
        done = self._t >= self.n_steps

        # 6. Info dict
        info = {
            "hidden_state": hidden,
            "latency"     : rec["latency"],
            "entropy"     : rec["entropy"],
            "reward"      : rec["reward"],
            "memory_usage": rec["memory_usage"],
            "error_rate"  : rec["error_rate"],
            "action_freq" : rec["action_freq"],
            "action"      : action,
            "exit_bonus"  : exit_bonus,
            "timestep"    : self._t,
        }

        if self.record_traj:
            self._trajectory.append({
                **rec,
                "action"      : action,
                "reward"      : reward,
                "obs"         : obs,
                "episode_step": self._t,
            })

        return StepResult(obs=obs, reward=reward, done=done, info=info)

    # ------------------------------------------------------------------ #
    # Trajectory access
    # ------------------------------------------------------------------ #

    @property
    def trajectory(self) -> List[Dict]:
        """Recorded telemetry trajectory for the current / last episode."""
        return self._trajectory

    @property
    def n_states(self) -> int:
        return N_STATES

    @property
    def n_actions(self) -> int:
        return N_ACTIONS


# ---------------------------------------------------------------------------
# Transition-matrix modulation
# ---------------------------------------------------------------------------

def _nudge_transition(
    T_base     : np.ndarray,
    from_state : int,
    action     : int,
    delta      : float,
) -> np.ndarray:
    """Return a new row-stochastic matrix with the *from_state* row nudged.

    **Cumulative design**: the nudge is computed relative to ``T_base``, which
    the caller passes as the *current* (possibly already-nudged) transition
    matrix — so repeated same-direction actions on the same state compound
    instead of each being discarded back to the pristine default. Validity is
    guaranteed by the floor-and-renormalise step below, not by resetting to a
    fixed base every call: a row can drift arbitrarily far from its default
    under sustained pushes, but can never leave [0.01, 1] or fail to sum to 1.

    Push-stable (action=0):
        Increase P(→ stable) by delta, decrease P(→ unstable) by delta.
    Push-exploratory (action=1):
        Increase P(→ exploratory) by delta/2 and P(→ adaptive) by delta/2,
        decrease P(→ unstable) by delta.
    Both actions clip to [0.01, 1] and re-normalise the row.

    Parameters
    ----------
    T_base     : the transition matrix to nudge from — pass the environment's
                 live ``self._T_current`` for cumulative behaviour
    from_state : current state index (0=stable,1=exploratory,2=adaptive,3=unstable)
    action     : ACTION_PUSH_STABLE or ACTION_PUSH_EXPLORATORY
    delta      : nudge magnitude
    """
    T_new = T_base.copy()
    row   = T_new[from_state].copy()   # start from the current (live) row

    STABLE_IDX      = _REGIME_IDX["stable"]
    EXPLOR_IDX      = _REGIME_IDX["exploratory"]
    ADAPTIVE_IDX    = _REGIME_IDX["adaptive"]
    UNSTABLE_IDX    = _REGIME_IDX["unstable"]

    if action == ACTION_PUSH_STABLE:
        row[STABLE_IDX]   += delta
        row[UNSTABLE_IDX] -= delta
    elif action == ACTION_PUSH_EXPLORATORY:
        row[EXPLOR_IDX]   += delta * 0.5
        row[ADAPTIVE_IDX] += delta * 0.5
        row[UNSTABLE_IDX] -= delta

    # Floor at 0.01 to keep all regime entry paths permanently open.
    row = np.clip(row, 0.01, 1.0)
    row /= row.sum()
    T_new[from_state] = row
    return T_new


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def _compute_reward(
    hidden_state: str,
    was_in_unstable: bool,
) -> Tuple[float, bool]:
    """Return (base_reward, exit_bonus_triggered).

    The exit bonus fires on the first step the agent leaves unstable.
    """
    if hidden_state == "unstable":
        return R_UNSTABLE, False
    else:
        exit_bonus = was_in_unstable  # leaving unstable → bonus on this step
        return R_HEALTHY, exit_bonus


# ---------------------------------------------------------------------------
# Multi-agent environment factory
# ---------------------------------------------------------------------------

def make_env_pool(
    n_envs   : int,
    base_seed: int = 42,
    delta    : float = DELTA_BASE,
    n_steps  : int  = N_STEPS_PER_EPISODE,
) -> List[BehavioralEnv]:
    """Create a pool of independent BehavioralEnv instances."""
    return [
        BehavioralEnv(
            agent_id    = f"rl_agent_{i:04d}",
            rng_seed    = base_seed + i,
            delta       = delta,
            n_steps     = n_steps,
            record_traj = True,
        )
        for i in range(n_envs)
    ]
