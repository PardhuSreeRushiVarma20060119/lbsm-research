"""
transition_analysis.py
======================
Latent Behavioral State Machine (LBSM) — Transition Matrix Analysis
--------------------------------------------------------------------
Utilities for comparing learned HMM transition matrices against
ground-truth Markov dynamics: element-wise error metrics, spectral
diagnostics, and dwell-time analysis.

All functions accept plain numpy arrays; they are decoupled from
``HMMResult`` to allow reuse with any row-stochastic matrix.

Reference
---------
"Latent Behavioral Structure in Low-Dimensional Statistical Manifolds"
Section 6.4 — Structural Validation of Learned Transition Dynamics
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .hidden_state_model import HMMResult


# ---------------------------------------------------------------------------
# Element-wise error
# ---------------------------------------------------------------------------
def transition_matrix_error(
    result        : HMMResult,
    T_gt          : np.ndarray,
    profile_names : Sequence[str],
) -> pd.DataFrame:
    """Compute element-wise absolute error between learned and GT transition matrices.

    The learned matrix is permuted into GT ordering via ``result.mapping``
    before comparison.

    Parameters
    ----------
    result        : fitted HMMResult (contains model and state mapping)
    T_gt          : ground-truth transition matrix  shape (K, K), row-stochastic
    profile_names : ordered regime names (length K)

    Returns
    -------
    DataFrame with columns:
        from_regime, to_regime, gt_prob, learned_prob, abs_error, rel_error
    Plus scalar summary attributes: mae, max_error, frobenius_norm_error
    """
    K       = len(profile_names)
    inv_map = {v: k for k, v in result.mapping.items()}   # gt_idx → hmm_state

    T_learned = result.model.transmat_

    # Permute learned matrix into GT ordering
    T_perm = np.zeros((K, K))
    for gi in range(K):
        for gj in range(K):
            hi = inv_map.get(gi)
            hj = inv_map.get(gj)
            if hi is not None and hj is not None:
                T_perm[gi, gj] = T_learned[hi, hj]

    T_error = np.abs(T_perm - T_gt)

    records = []
    for i, from_r in enumerate(profile_names):
        for j, to_r in enumerate(profile_names):
            gt_p = float(T_gt[i, j])
            lp   = float(T_perm[i, j])
            ae   = float(T_error[i, j])
            re   = ae / (gt_p + 1e-12)
            records.append({
                "from_regime" : from_r,
                "to_regime"   : to_r,
                "gt_prob"     : round(gt_p, 6),
                "learned_prob": round(lp, 6),
                "abs_error"   : round(ae, 6),
                "rel_error"   : round(re, 6),
            })

    df = pd.DataFrame(records)

    # Attach scalar summaries as DataFrame attrs for downstream access
    df.attrs["mae"]                  = float(T_error.mean())
    df.attrs["max_error"]            = float(T_error.max())
    df.attrs["frobenius_norm_error"] = float(np.linalg.norm(T_perm - T_gt, "fro"))
    df.attrs["T_learned_perm"]       = T_perm   # aligned matrix for plotting

    return df


# ---------------------------------------------------------------------------
# Spectral gap
# ---------------------------------------------------------------------------
def spectral_gap(transmat: np.ndarray) -> float:
    """Return the spectral gap of a row-stochastic matrix.

    The spectral gap is  1 − |λ₂|  where λ₂ is the second-largest eigenvalue
    by absolute value (λ₁ = 1 always for an ergodic chain).
    A larger gap implies faster mixing / clearer regime separation.

    Parameters
    ----------
    transmat : np.ndarray  shape (K, K) — row-stochastic

    Returns
    -------
    gap : float  ∈ (0, 1]
    """
    eigenvalues = np.linalg.eigvals(transmat)
    abs_ev      = np.sort(np.abs(eigenvalues))[::-1]   # descending
    lambda2     = abs_ev[1] if len(abs_ev) > 1 else 0.0
    return float(1.0 - lambda2)


# ---------------------------------------------------------------------------
# Expected dwell times
# ---------------------------------------------------------------------------
def expected_dwell_times(
    transmat     : np.ndarray,
    profile_names: Optional[Sequence[str]] = None,
) -> pd.Series:
    """Expected number of consecutive timesteps spent in each state.

    For a first-order Markov chain, the expected dwell time in state s is:
        E[dwell_s] = 1 / (1 − T_{ss})

    Parameters
    ----------
    transmat      : row-stochastic transition matrix  (K, K)
    profile_names : optional state labels for the returned Series index

    Returns
    -------
    pd.Series indexed by state index (or profile_names)
    """
    K      = transmat.shape[0]
    p_self = np.diag(transmat)
    dwell  = 1.0 / (1.0 - p_self + 1e-12)

    index  = list(profile_names) if profile_names is not None else list(range(K))
    return pd.Series(dwell, index=index, name="expected_dwell_steps")


# ---------------------------------------------------------------------------
# Empirical transition counts from a decoded sequence
# ---------------------------------------------------------------------------
def empirical_transition_counts(
    state_seq    : np.ndarray,
    n_states     : int,
    profile_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Count observed transitions in a decoded state sequence.

    Parameters
    ----------
    state_seq     : 1-D integer array of decoded states
    n_states      : total number of states K
    profile_names : optional labels (length K)

    Returns
    -------
    DataFrame of shape (K, K) — raw transition counts
    """
    counts = np.zeros((n_states, n_states), dtype=int)
    for t in range(len(state_seq) - 1):
        counts[state_seq[t], state_seq[t + 1]] += 1

    index = list(profile_names) if profile_names is not None else list(range(n_states))
    return pd.DataFrame(counts, index=index, columns=index)