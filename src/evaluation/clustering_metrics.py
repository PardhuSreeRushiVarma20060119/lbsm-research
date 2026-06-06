"""
clustering_metrics.py
=====================
LBSM — Evaluation
------------------
Clustering quality metrics for embedding and regime-recovery evaluation.
Used across NB02 (manifold), NB03 (HMM), NB04 (drift).
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score,
)


def clustering_scorecard(
    X      : np.ndarray,
    labels : np.ndarray,
    name   : str = "method",
    sample : int = 5000,
    seed   : int = 42,
) -> dict:
    """Compute silhouette, Davies-Bouldin, and Calinski-Harabasz scores.

    Returns
    -------
    dict  metric → value
    """
    sil = float(silhouette_score(X, labels, sample_size=min(sample, len(labels)),
                                  random_state=seed))
    db  = float(davies_bouldin_score(X, labels))
    ch  = float(calinski_harabasz_score(X, labels))
    return {"method": name, "silhouette": sil, "davies_bouldin": db,
            "calinski_harabasz": ch}


def per_class_silhouette(
    X             : np.ndarray,
    labels        : np.ndarray,
    class_names   : Tuple[str, ...],
) -> pd.DataFrame:
    """Mean silhouette per class."""
    sil = silhouette_samples(X, labels)
    rows = []
    for i, name in enumerate(class_names):
        mask = labels == i
        rows.append({"class": name, "mean_sil": float(sil[mask].mean()),
                     "std_sil": float(sil[mask].std()), "n": int(mask.sum())})
    return pd.DataFrame(rows).set_index("class")


def ari_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Adjusted Rand Index (permutation-invariant cluster similarity)."""
    return float(adjusted_rand_score(y_true, y_pred))