"""
sequence_inference.py
=====================
Latent Behavioral State Machine (LBSM) — Sequence-Level HMM Utilities
----------------------------------------------------------------------
Model selection (BIC / AIC sweep) and stationary distribution computation.
These functions operate on fitted or candidate HMM models and the
multi-sequence concatenated data format produced by
:func:`hidden_state_model.prepare_sequences`.

Reference
---------
"Latent Behavioral State Machines: Manifold Geometry of Adaptive Agent Telemetry"
Section 6.2 — Model Order Selection and Identifiability
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM
    _HMMLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HMMLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model selection sweep — BIC / AIC
# ---------------------------------------------------------------------------
def model_selection_sweep(
    X_concat       : np.ndarray,
    lengths        : List[int],
    n_comp_grid    : Sequence[int] = (2, 3, 4, 5, 6),
    covariance_type: str           = "diag",
    n_iter         : int           = 200,
    tol            : float         = 1e-4,
    random_state   : int           = 42,
    min_covar      : float         = 1e-3,
) -> pd.DataFrame:
    """Fit HMMs with varying n_components and return BIC / AIC scores.

    For each candidate model size we fit a fresh Baum-Welch HMM, compute
    the total log-likelihood on the training data, count free parameters,
    and derive BIC and AIC.

    Parameter count for a Gaussian HMM with diagonal covariance:
        - Initial state distribution  : K − 1
        - Transition matrix            : K(K − 1)
        - Emission means               : K × d
        - Emission diagonal variances  : K × d
        Total p = K − 1 + K(K − 1) + 2Kd = K² + 2Kd − 1

    Parameters
    ----------
    X_concat        : concatenated observation matrix  (N_total, d)
    lengths         : per-sequence lengths
    n_comp_grid     : iterable of candidate n_components values
    covariance_type : HMM covariance structure (only 'diag' has exact param count here)
    n_iter          : maximum EM iterations per model
    tol             : convergence tolerance
    random_state    : reproducibility seed
    min_covar       : Tikhonov/diagonal-loading floor before covariance
        inversion (see ``LBSM-ISSUE-NB07-001``, ``outputs/reports/issues/``,
        and :func:`src.hmm.robust_fitting.fit_hmm_robust` for the full
        mitigation stack this single-fit sweep does not by itself provide
        multi-restart/fallback protection against)

    Returns
    -------
    DataFrame with columns:
        n_components, log_likelihood (per obs), total_ll, n_params, bic, aic
    Sorted ascending by n_components.
    """
    if not _HMMLEARN_AVAILABLE:
        raise ImportError("hmmlearn is required.  pip install hmmlearn")

    N, d   = X_concat.shape
    records = []

    for K in n_comp_grid:
        model = _GaussianHMM(
            n_components    = K,
            covariance_type = covariance_type,
            n_iter          = n_iter,
            tol             = tol,
            min_covar       = min_covar,
            random_state    = random_state,
        )
        model.fit(X_concat, lengths)

        total_ll = float(model.score(X_concat, lengths))
        ll_obs   = total_ll / N

        # Free parameter count
        if covariance_type == "diag":
            n_params = K * K + 2 * K * d - 1
        elif covariance_type == "full":
            n_params = K * K + K * d + K * d * (d + 1) // 2 - 1
        elif covariance_type == "tied":
            n_params = K * K + K * d + d * (d + 1) // 2 - 1
        elif covariance_type == "spherical":
            n_params = K * K + K * d + K - 1
        else:
            n_params = K * K + 2 * K * d - 1

        bic = -2 * total_ll + n_params * np.log(N)
        aic = -2 * total_ll + 2 * n_params

        records.append({
            "n_components" : K,
            "log_likelihood": round(ll_obs, 6),
            "total_ll"     : round(total_ll, 4),
            "n_params"     : n_params,
            "bic"          : round(bic, 4),
            "aic"          : round(aic, 4),
        })

    return pd.DataFrame(records).sort_values("n_components").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stationary distribution
# ---------------------------------------------------------------------------
def stationary_distribution(transmat: np.ndarray) -> np.ndarray:
    """Compute the stationary (limiting) distribution of a row-stochastic matrix.

    Solves π T = π  subject to  Σ π_i = 1  via the left eigenvector of T
    corresponding to eigenvalue 1.

    For ergodic Markov chains, this is unique; for near-degenerate matrices
    we fall back to the power-iteration result.

    Parameters
    ----------
    transmat : np.ndarray  shape (K, K) — row-stochastic transition matrix

    Returns
    -------
    pi : np.ndarray  shape (K,) — stationary probabilities (sums to 1)
    """
    T  = np.array(transmat, dtype=np.float64)
    K  = T.shape[0]

    # Left eigenvectors of T  ↔  right eigenvectors of T.T
    eigenvalues, eigenvectors = np.linalg.eig(T.T)

    # Eigenvalue closest to 1
    idx = int(np.argmin(np.abs(eigenvalues - 1.0)))
    pi  = np.real(eigenvectors[:, idx])

    # Ensure non-negativity (may be up to sign)
    if pi.sum() < 0:
        pi = -pi

    pi = np.clip(pi, 0.0, None)
    total = pi.sum()

    if total < 1e-12:
        # Degenerate: power iteration fallback
        v = np.ones(K) / K
        for _ in range(10_000):
            v_new = v @ T
            if np.max(np.abs(v_new - v)) < 1e-12:
                break
            v = v_new
        pi = v
    else:
        pi /= total

    return pi