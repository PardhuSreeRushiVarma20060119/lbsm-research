"""
latent_state_metrics.py
=======================
Latent Behavioral State Machine (LBSM) — HMM Evaluation Metrics
-----------------------------------------------------------------
Per-regime accuracy, per-agent breakdown, and posterior-entropy
utilities.  All functions consume an ``HMMResult`` produced by
:func:`hidden_state_model.fit_hmm`.

Reference
---------
"Latent Behavioral Structure in Low-Dimensional Statistical Manifolds"
Section 6.3 — Evaluation of Unsupervised Regime Recovery
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, adjusted_rand_score, confusion_matrix

from .hidden_state_model import HMMResult


# ---------------------------------------------------------------------------
# Per-regime accuracy
# ---------------------------------------------------------------------------
def per_regime_accuracy(
    result        : HMMResult,
    profile_names : Sequence[str],
) -> pd.DataFrame:
    """Compute per-regime precision, recall, and accuracy.

    Uses ``result.pred_aligned`` (Hungarian-matched) against the ground-truth
    labels stored implicitly in ``result.confusion`` (rows = GT, cols = pred).

    Parameters
    ----------
    result        : fitted HMMResult (must contain ``confusion`` matrix)
    profile_names : ordered regime names (length == n_components)

    Returns
    -------
    DataFrame indexed by regime name, columns:
        support   — number of GT observations in that regime
        n_correct — correctly decoded observations
        accuracy  — per-regime accuracy (= recall / sensitivity)
        precision — TP / (TP + FP) for the decoded state
        f1        — harmonic mean of precision and recall
    """
    cm      = result.confusion              # (n, n) — GT rows, pred cols
    n       = cm.shape[0]
    records = []

    for i, name in enumerate(profile_names):
        tp      = cm[i, i]
        support = cm[i, :].sum()
        recall  = float(tp) / (support + 1e-12)
        pred_total = cm[:, i].sum()
        precision  = float(tp) / (pred_total + 1e-12)
        f1 = (2 * precision * recall) / (precision + recall + 1e-12)
        records.append({
            "regime"   : name,
            "support"  : int(support),
            "n_correct": int(tp),
            "accuracy" : round(recall, 6),
            "precision": round(precision, 6),
            "f1"       : round(f1, 6),
        })

    df = pd.DataFrame(records).set_index("regime")
    return df


# ---------------------------------------------------------------------------
# Per-agent metrics
# ---------------------------------------------------------------------------
def per_agent_metrics(
    result       : HMMResult,
    y_gt         : np.ndarray,
    lengths      : List[int],
    agent_ids    : List[str],
    profile_names: Sequence[str],
) -> pd.DataFrame:
    """Compute ARI and Hungarian-aligned accuracy for every agent separately.

    Each agent's subsequence is extracted from the concatenated arrays using
    the ``lengths`` list; Hungarian alignment is recomputed independently
    per-agent so idiosyncratic state usage is handled correctly.

    Parameters
    ----------
    result       : global HMMResult (pred_aligned is used as starting point)
    y_gt         : concatenated ground-truth integer labels  (N_total,)
    lengths      : per-agent sequence lengths (sum == N_total)
    agent_ids    : ordered agent identifiers matching ``lengths``
    profile_names: ordered regime name tuple

    Returns
    -------
    DataFrame indexed by agent_id, columns: ari, accuracy, n_obs, dominant_regime
    """
    n_regimes = len(profile_names)
    records   = []
    offset    = 0

    for aid, length in zip(agent_ids, lengths):
        pred_raw_a = result.pred_raw[offset : offset + length]
        gt_a       = y_gt[offset : offset + length]

        # Per-agent Hungarian alignment
        C          = confusion_matrix(gt_a, pred_raw_a,
                                      labels=list(range(n_regimes)))
        row, col   = linear_sum_assignment(-C)
        mapping_a  = {int(c): int(r) for c, r in zip(col, row)}
        pred_aln_a = np.array([mapping_a.get(s, s) for s in pred_raw_a])

        ari = float(adjusted_rand_score(gt_a, pred_raw_a))
        acc = float(accuracy_score(gt_a, pred_aln_a))

        # Most frequent GT regime for this agent
        dominant = profile_names[int(np.bincount(gt_a, minlength=n_regimes).argmax())]

        records.append({
            "agent_id"       : aid,
            "ari"            : round(ari, 6),
            "accuracy"       : round(acc, 6),
            "n_obs"          : length,
            "dominant_regime": dominant,
        })
        offset += length

    df = pd.DataFrame(records).set_index("agent_id")
    return df


# ---------------------------------------------------------------------------
# Posterior entropy
# ---------------------------------------------------------------------------
def posterior_entropy(posteriors: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) of the forward-backward posterior at each timestep.

    H_t = -∑_s  P(s_t | x_{1:T}) · ln P(s_t | x_{1:T})

    Entropy is zero when the posterior is concentrated on one state (high
    certainty) and ln(n_states) when it is uniform (maximum uncertainty).
    High-entropy timesteps typically correspond to regime-boundary crossings.

    Parameters
    ----------
    posteriors : np.ndarray  shape (N, n_states) — output of
                 ``model.predict_proba()`` or ``HMMResult.posteriors_all``

    Returns
    -------
    H : np.ndarray  shape (N,)  — per-timestep entropy in nats
    """
    p   = np.clip(posteriors, 1e-15, 1.0)
    H   = -np.sum(p * np.log(p), axis=1)
    return H