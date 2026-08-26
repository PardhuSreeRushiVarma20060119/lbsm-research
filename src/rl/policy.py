"""
policy.py
=========
LBSM — Reinforcement Learning Layer
------------------------------------
Policy analysis utilities: extracting greedy policies from Q-tables,
measuring policy entropy (confidence), computing action-frequency maps,
and comparing learned vs. random vs. baseline policies.

These tools support NB05's research questions:
  - Which regions of (latency, entropy) grid does the policy act on most?
  - Does the policy concentrate on the unstable-regime corner?
  - How does policy entropy (spread of Q-values) evolve during training?

Reference
---------
"Latent Behavioral Structure in Low-Dimensional Statistical Manifolds"
Section 8.2 — Policy Analysis & Behavioral Action Maps
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .environment import (
    N_STATES, N_ACTIONS, N_GRID_LATENCY, N_GRID_ENTROPY,
    ACTION_PUSH_STABLE, ACTION_PUSH_EXPLORATORY, ACTION_DO_NOTHING,
    grid_to_coords,
)

# Action name map for display
ACTION_NAMES = {
    ACTION_PUSH_STABLE      : "push_stable",
    ACTION_PUSH_EXPLORATORY : "push_exploratory",
    ACTION_DO_NOTHING       : "do_nothing",
}


# ---------------------------------------------------------------------------
# Policy extraction
# ---------------------------------------------------------------------------

def greedy_policy(Q: np.ndarray) -> np.ndarray:
    """Return the greedy policy π(s) = argmax_a Q(s, a).

    Parameters
    ----------
    Q : np.ndarray  shape (N_STATES, N_ACTIONS)

    Returns
    -------
    policy : np.ndarray  shape (N_STATES,)  dtype int
    """
    return Q.argmax(axis=1).astype(int)


def policy_action_grid(policy: np.ndarray) -> np.ndarray:
    """Reshape flat policy into a (N_GRID_LATENCY, N_GRID_ENTROPY) action grid.

    Useful for heatmap visualisation: each cell shows which action the policy
    selects when the agent is in that (latency_bin, entropy_bin) cell.

    Parameters
    ----------
    policy : np.ndarray  shape (N_STATES,)

    Returns
    -------
    grid : np.ndarray  shape (N_GRID_LATENCY, N_GRID_ENTROPY)  dtype int
    """
    return policy.reshape(N_GRID_LATENCY, N_GRID_ENTROPY)


def value_grid(Q: np.ndarray) -> np.ndarray:
    """Reshape V(s) = max_a Q(s,a) into (N_GRID_LATENCY, N_GRID_ENTROPY).

    Returns
    -------
    grid : np.ndarray  shape (N_GRID_LATENCY, N_GRID_ENTROPY)
    """
    return Q.max(axis=1).reshape(N_GRID_LATENCY, N_GRID_ENTROPY)


# ---------------------------------------------------------------------------
# Policy entropy (Q-value spread)
# ---------------------------------------------------------------------------

def policy_entropy(Q: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute per-state entropy of the softmax policy π_τ(a|s).

    A high-entropy state means the Q-values are nearly uniform → the agent
    is uncertain about the best action. Low entropy → confident policy.

    H(s) = -Σ_a π_τ(a|s) log π_τ(a|s)
    π_τ(a|s) = softmax(Q(s,·) / τ)

    Parameters
    ----------
    Q           : np.ndarray  shape (N_STATES, N_ACTIONS)
    temperature : τ; lower → sharper policy distribution

    Returns
    -------
    entropy : np.ndarray  shape (N_STATES,)
    """
    scaled = Q / (temperature + 1e-9)
    # Numerically stable softmax
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp_Q   = np.exp(shifted)
    probs   = exp_Q / exp_Q.sum(axis=1, keepdims=True)
    # Shannon entropy
    log_probs = np.log(probs + 1e-12)
    return -(probs * log_probs).sum(axis=1)


def policy_entropy_grid(Q: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Reshape policy_entropy into (N_GRID_LATENCY, N_GRID_ENTROPY)."""
    return policy_entropy(Q, temperature).reshape(N_GRID_LATENCY, N_GRID_ENTROPY)


# ---------------------------------------------------------------------------
# Action frequency analysis
# ---------------------------------------------------------------------------

def action_frequency_from_trajectory(
    trajectory : List[Dict],
) -> Dict[str, float]:
    """Count action frequencies in a recorded episode trajectory.

    Parameters
    ----------
    trajectory : list of dicts with 'action' key

    Returns
    -------
    freq_dict : {action_name: fraction}
    """
    actions = [r["action"] for r in trajectory if r["action"] >= 0]
    if not actions:
        return {name: 0.0 for name in ACTION_NAMES.values()}
    total = len(actions)
    return {
        ACTION_NAMES[a]: sum(1 for x in actions if x == a) / total
        for a in range(N_ACTIONS)
    }


def action_state_heatmap(
    trajectory : List[Dict],
) -> np.ndarray:
    """Build a (N_STATES, N_ACTIONS) action-count matrix from a trajectory.

    Entry [s, a] = number of times action a was taken in state s.
    """
    counts = np.zeros((N_STATES, N_ACTIONS), dtype=int)
    for r in trajectory:
        if r["action"] >= 0:
            counts[r["obs"], r["action"]] += 1
    return counts


# ---------------------------------------------------------------------------
# Policy comparison
# ---------------------------------------------------------------------------

def policy_agreement(policy_a: np.ndarray, policy_b: np.ndarray) -> float:
    """Fraction of states where two policies agree.

    Parameters
    ----------
    policy_a, policy_b : np.ndarray  shape (N_STATES,)

    Returns
    -------
    agreement : float in [0, 1]
    """
    return float((policy_a == policy_b).mean())


def policy_summary_table(
    agents      : list,    # list of QLearningAgent
    agent_ids   : Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build a DataFrame summarising per-agent policy characteristics.

    Columns: agent_id, push_stable_frac, push_exploratory_frac,
             do_nothing_frac, mean_policy_entropy, mean_value
    """
    rows = []
    for i, agent in enumerate(agents):
        pid    = agent_ids[i] if agent_ids else f"agent_{i:04d}"
        pol    = greedy_policy(agent.Q)
        ent    = policy_entropy(agent.Q)
        counts = np.bincount(pol, minlength=N_ACTIONS)
        frac   = counts / N_STATES
        rows.append({
            "agent_id"              : pid,
            "push_stable_frac"      : frac[ACTION_PUSH_STABLE],
            "push_exploratory_frac" : frac[ACTION_PUSH_EXPLORATORY],
            "do_nothing_frac"       : frac[ACTION_DO_NOTHING],
            "mean_policy_entropy"   : float(ent.mean()),
            "mean_value"            : float(agent.Q.max(axis=1).mean()),
        })
    return pd.DataFrame(rows)
