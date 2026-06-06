"""
drift_detection.py
==================
LBSM — Drift Detection
-----------------------
Point-level anomaly detectors: Mahalanobis distance from the healthy
regime envelope, and a combined multi-detector score.

Mahalanobis distance measures how many standard deviations a telemetry
observation lies from the centroid of the healthy distribution — taking
feature correlations into account. It is the theoretically correct
distance measure for Gaussian-emission HMMs and serves as the primary
geometric anomaly score in the LBSM paper.

Reference
---------
"Latent Behavioral State Machines: Manifold Geometry of Adaptive Agent Telemetry"
Section 7.4 — Point-Level Detection: Mahalanobis Scoring
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class HealthyEnvelope:
    """Fitted Gaussian envelope for the healthy behavioural regime.

    Attributes
    ----------
    mu      : mean vector            shape (d,)
    cov_inv : inverse covariance     shape (d, d)
    cov     : covariance matrix      shape (d, d)
    regime_params : per-regime (mu, cov_inv)  for multi-regime scoring
    """
    mu           : np.ndarray
    cov_inv      : np.ndarray
    cov          : np.ndarray
    regime_params: Dict[str, Tuple[np.ndarray, np.ndarray]]


@dataclass
class MahalanobisResult:
    """Mahalanobis scoring result.

    Attributes
    ----------
    scores    : Mahalanobis distance per observation  shape (N,)
    flags     : binary anomaly flag                   shape (N,)
    threshold : scalar threshold (chi² or percentile)
    """
    scores   : np.ndarray
    flags    : np.ndarray
    threshold: float


# ---------------------------------------------------------------------------
# Fit healthy envelope
# ---------------------------------------------------------------------------
def fit_healthy_envelope(
    df          : pd.DataFrame,
    feature_cols: Sequence[str],
    regime_col  : str = "hidden_state",
    healthy_regimes: Tuple[str, ...] = ("stable", "exploratory", "adaptive"),
    regularize  : float = 1e-6,
) -> HealthyEnvelope:
    """Fit a Gaussian envelope on healthy-regime observations.

    Parameters
    ----------
    df              : full telemetry DataFrame
    feature_cols    : feature columns to use
    regime_col      : column containing regime labels
    healthy_regimes : regime names considered healthy (not anomalous)
    regularize      : small diagonal regularisation for covariance invertibility

    Returns
    -------
    envelope : HealthyEnvelope
    """
    feat = list(feature_cols)
    healthy_mask = df[regime_col].isin(healthy_regimes)
    X_healthy    = df[healthy_mask][feat].values.astype(np.float64)

    mu      = X_healthy.mean(axis=0)
    cov     = np.cov(X_healthy.T) + np.eye(len(feat)) * regularize
    cov_inv = np.linalg.inv(cov)

    # Per-regime parameters
    regime_params: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for regime in healthy_regimes:
        sub = df[df[regime_col] == regime][feat].values.astype(np.float64)
        if len(sub) < 2:
            continue
        mu_r    = sub.mean(axis=0)
        cov_r   = np.cov(sub.T) + np.eye(len(feat)) * regularize
        cov_r_i = np.linalg.inv(cov_r)
        regime_params[regime] = (mu_r, cov_r_i)

    return HealthyEnvelope(
        mu            = mu,
        cov_inv       = cov_inv,
        cov           = cov,
        regime_params = regime_params,
    )


# ---------------------------------------------------------------------------
# Mahalanobis distance scoring
# ---------------------------------------------------------------------------
def mahalanobis_scores(
    X       : np.ndarray,
    envelope: HealthyEnvelope,
    mode    : str = "global",
) -> np.ndarray:
    """Compute Mahalanobis distance from the healthy envelope for each observation.

    Parameters
    ----------
    X        : np.ndarray  shape (N, d)
    envelope : HealthyEnvelope from :func:`fit_healthy_envelope`
    mode     : ``'global'``  — distance from pooled healthy centroid
               ``'min'``    — minimum distance across per-regime centroids

    Returns
    -------
    scores : np.ndarray  shape (N,)
    """
    if mode == "global":
        diff   = X - envelope.mu
        scores = np.sqrt(np.einsum("ni,ij,nj->n", diff, envelope.cov_inv, diff))
    elif mode == "min":
        per_regime = np.stack([
            _mah_dist(X, mu_r, cov_r_i)
            for mu_r, cov_r_i in envelope.regime_params.values()
        ], axis=1)
        scores = per_regime.min(axis=1)
    else:
        raise ValueError(f"mode must be 'global' or 'min', got {mode!r}")

    return scores


def _mah_dist(X: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    diff = X - mu
    return np.sqrt(np.einsum("ni,ij,nj->n", diff, cov_inv, diff))


def fit_mahalanobis(
    X             : np.ndarray,
    envelope      : HealthyEnvelope,
    mode          : str = "global",
    threshold_pct : float = 97.5,
    y_healthy     : Optional[np.ndarray] = None,
) -> MahalanobisResult:
    """Score observations and flag those above a percentile threshold.

    The threshold is calibrated from healthy observations only, so it
    represents the tail of the in-distribution score rather than a
    percentile of a mixed (healthy + anomalous) population.  Computing
    the threshold from the full dataset — including anomalies — inflates
    it and suppresses detections.

    Parameters
    ----------
    X             : feature matrix  shape (N, d)
    envelope      : fitted HealthyEnvelope
    mode          : ``'global'`` or ``'min'``
    threshold_pct : percentile of *healthy* scores used as the anomaly
                    threshold; 97.5 ≈ ~2σ tail under Gaussianity
    y_healthy     : optional boolean / 0-1 mask  shape (N,) indicating
                    which rows of *X* are healthy.  The threshold is
                    derived from ``scores[y_healthy]`` only.
                    If ``None``, the full score distribution is used and
                    a ``UserWarning`` is raised, because the resulting
                    threshold will be inflated whenever anomalies are
                    present in *X*.

    Returns
    -------
    result : MahalanobisResult
    """
    import warnings

    scores = mahalanobis_scores(X, envelope, mode)

    if y_healthy is not None:
        mask = np.asarray(y_healthy, dtype=bool)
        if mask.sum() == 0:
            raise ValueError("y_healthy contains no True entries; cannot fit threshold.")
        ref_scores = scores[mask]
    else:
        warnings.warn(
            "fit_mahalanobis: y_healthy not provided. Threshold is being computed "
            "from the full score distribution, which includes any anomalies present "
            "in X. This inflates the threshold and suppresses detections. Pass a "
            "healthy-only mask to get a calibrated threshold.",
            UserWarning,
            stacklevel=2,
        )
        ref_scores = scores

    threshold = float(np.percentile(ref_scores, threshold_pct))
    flags     = (scores > threshold).astype(np.int8)

    return MahalanobisResult(scores=scores, flags=flags, threshold=threshold)


# ---------------------------------------------------------------------------
# Combined detector
# ---------------------------------------------------------------------------
def combined_anomaly_score(
    mah_scores : np.ndarray,
    ewma_scores: np.ndarray,
    w_mah      : float = 0.6,
    w_ewma     : float = 0.4,
) -> np.ndarray:
    """Combine Mahalanobis and EWMA scores into a single anomaly score.

    Both inputs are min-max normalised to [0, 1] before weighted combination.

    Parameters
    ----------
    mah_scores  : Mahalanobis distance array  shape (N,)
    ewma_scores : EWMA residual score array    shape (N,)
    w_mah       : weight for Mahalanobis component
    w_ewma      : weight for EWMA component

    Returns
    -------
    combined : np.ndarray  shape (N,)  ∈ [0, 1]
    """
    def _norm(x: np.ndarray) -> np.ndarray:
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-9)

    return w_mah * _norm(mah_scores) + w_ewma * _norm(ewma_scores)


# ---------------------------------------------------------------------------
# Threshold sweep (ROC)
# ---------------------------------------------------------------------------
def threshold_sweep(
    scores: np.ndarray,
    y_gt  : np.ndarray,
    n_pts : int = 200,
) -> pd.DataFrame:
    """Compute precision, recall, F1, and FPR across a threshold grid.

    Parameters
    ----------
    scores : anomaly score array  shape (N,)
    y_gt   : binary ground-truth  shape (N,)
    n_pts  : number of threshold points

    Returns
    -------
    df : pd.DataFrame  columns=[threshold, precision, recall, f1, fpr, tpr]
    """
    thresholds = np.linspace(scores.min(), scores.max(), n_pts)
    rows = []
    for thr in thresholds:
        pred     = (scores >= thr).astype(int)
        tp       = int(((pred == 1) & (y_gt == 1)).sum())
        fp       = int(((pred == 1) & (y_gt == 0)).sum())
        fn       = int(((pred == 0) & (y_gt == 1)).sum())
        tn       = int(((pred == 0) & (y_gt == 0)).sum())
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        fpr       = fp / (fp + tn + 1e-9)
        rows.append({
            "threshold": float(thr),
            "precision": precision,
            "recall"   : recall,
            "f1"       : f1,
            "fpr"      : fpr,
            "tpr"      : recall,
        })
    return pd.DataFrame(rows)