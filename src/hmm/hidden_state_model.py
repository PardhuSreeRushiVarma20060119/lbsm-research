"""
hidden_state_model.py
=====================
Latent Behavioral State Machine (LBSM) — HMM Inference
--------------------------------------------------------
Core HMM model: result container, sequence preparation, and the
primary Baum-Welch / Viterbi fit.

This module owns the single entry-point the notebook calls:
``fit_hmm`` — everything else in src/hmm is downstream analysis
of the ``HMMResult`` this function returns.

Reference
---------
"Latent Behavioral State Machines: Manifold Geometry of Adaptive Agent Telemetry"
Section 6.1 — Unsupervised Regime Recovery via Hidden Markov Models
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, adjusted_rand_score, confusion_matrix

try:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM
    _HMMLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HMMLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class HMMResult:
    """Full HMM inference result.

    Attributes
    ----------
    model          : fitted GaussianHMM
    pred_raw       : Viterbi state sequence (HMM indices, not GT-aligned)
    pred_aligned   : Viterbi sequence after Hungarian alignment to GT
    posteriors_all : forward-backward posteriors  shape (N_total, n_comp)
    mapping        : dict  hmm_state → gt_regime_index
    ari            : Adjusted Rand Index (raw, permutation-invariant)
    accuracy       : accuracy after Hungarian alignment
    log_likelihood : total log-likelihood (per observation)
    confusion      : confusion matrix (GT rows, pred cols) after alignment
    convergence_ll : log-likelihood history across EM iterations (list)
    n_iter_actual  : number of EM iterations actually run
    """

    model          : object              # hmmlearn GaussianHMM
    pred_raw       : np.ndarray          # (N_total,)
    pred_aligned   : np.ndarray          # (N_total,)
    posteriors_all : np.ndarray          # (N_total, n_comp)
    mapping        : Dict[int, int]      # hmm_state → gt_index
    ari            : float
    accuracy       : float
    log_likelihood : float               # per-observation LL
    confusion      : np.ndarray          # (n_regimes, n_regimes)
    convergence_ll : List[float]
    n_iter_actual  : int


# ---------------------------------------------------------------------------
# Sequence preparation
# ---------------------------------------------------------------------------
def prepare_sequences(
    df           : pd.DataFrame,
    feature_cols : Sequence[str],
    agent_col    : str = "agent_id",
    time_col     : str = "timestep",
    z_scored     : bool = False,
) -> Tuple[np.ndarray, List[int], List[str]]:
    """Stack per-agent telemetry into the concatenated array hmmlearn expects.

    hmmlearn's multi-sequence API takes a single ``(N_total, d)`` matrix
    plus a ``lengths`` list encoding where each sequence starts/ends.
    This function produces exactly that from the LBSM DataFrame format.

    Parameters
    ----------
    df           : full telemetry DataFrame (all agents)
    feature_cols : telemetry feature column names (raw or z-scored)
    agent_col    : column identifying agents
    time_col     : column for temporal ordering within each agent
    z_scored     : if True, prepend ``_z`` suffix to feature column names

    Returns
    -------
    X_concat  : np.ndarray  shape (N_total, d)
    lengths   : list of int  — per-agent sequence length (sum = N_total)
    agent_ids : list of str  — ordered agent identifiers (matches lengths)
    """
    cols = [f"{f}_z" for f in feature_cols] if z_scored else list(feature_cols)
    df_s = df.sort_values([agent_col, time_col])

    Xs, lengths, agent_ids = [], [], []
    for aid in sorted(df_s[agent_col].unique()):
        sub = df_s[df_s[agent_col] == aid]
        X_a = sub[cols].values.astype(np.float64)
        Xs.append(X_a)
        lengths.append(len(X_a))
        agent_ids.append(str(aid))

    return np.concatenate(Xs, axis=0), lengths, agent_ids


# ---------------------------------------------------------------------------
# Core fit
# ---------------------------------------------------------------------------
def fit_hmm(
    X_concat      : np.ndarray,
    lengths       : List[int],
    y_gt          : np.ndarray,
    n_components  : int = 4,
    covariance_type: str = "diag",
    n_iter        : int = 200,
    tol           : float = 1e-4,
    random_state  : int = 42,
    profile_names : Tuple[str, ...] = ("stable", "exploratory", "adaptive", "unstable"),
) -> HMMResult:
    """Fit a Gaussian-emission HMM via Baum-Welch (EM) and decode via Viterbi.

    The model is fit **without** accessing ``y_gt``; labels are used only
    post-hoc for Hungarian alignment and metric computation.

    Parameters
    ----------
    X_concat        : concatenated observation matrix  (N_total, d)
    lengths         : per-sequence lengths (output of :func:`prepare_sequences`)
    y_gt            : ground-truth integer labels  (N_total,)
    n_components    : number of hidden states to learn
    covariance_type : ``'diag'`` | ``'full'`` | ``'tied'`` | ``'spherical'``
    n_iter          : maximum Baum-Welch EM iterations
    tol             : convergence tolerance on log-likelihood change
    random_state    : reproducibility seed
    profile_names   : ordered regime name tuple (for confusion matrix axes)

    Returns
    -------
    result : HMMResult
    """
    if not _HMMLEARN_AVAILABLE:
        raise ImportError("hmmlearn is required.  pip install hmmlearn")

    model = _GaussianHMM(
        n_components    = n_components,
        covariance_type = covariance_type,
        n_iter          = n_iter,
        tol             = tol,
        random_state    = random_state,
    )
    model.fit(X_concat, lengths)

    # ── Viterbi decoding (most probable state path)
    pred_raw = model.predict(X_concat, lengths)

    # ── Forward-backward posteriors P(s_t | x_{1:T})
    posteriors_all = model.predict_proba(X_concat)

    # ── Hungarian alignment: match HMM states to GT regime indices
    n_gt     = len(profile_names)
    C        = confusion_matrix(y_gt, pred_raw, labels=list(range(n_gt)))
    row, col = linear_sum_assignment(-C)
    mapping  = {int(c): int(r) for c, r in zip(col, row)}
    pred_aligned = np.array([mapping.get(s, s) for s in pred_raw])

    # ── Scalar metrics
    ari      = float(adjusted_rand_score(y_gt, pred_raw))
    acc      = float(accuracy_score(y_gt, pred_aligned))
    ll_obs   = float(model.score(X_concat, lengths)) / len(X_concat)
    conf_mat = confusion_matrix(y_gt, pred_aligned, labels=list(range(n_gt)))

    # ── EM convergence history
    conv_ll  = list(model.monitor_.history)
    n_actual = len(conv_ll)

    return HMMResult(
        model          = model,
        pred_raw       = pred_raw,
        pred_aligned   = pred_aligned,
        posteriors_all = posteriors_all,
        mapping        = mapping,
        ari            = ari,
        accuracy       = acc,
        log_likelihood = ll_obs,
        confusion      = conf_mat,
        convergence_ll = conv_ll,
        n_iter_actual  = n_actual,
    )