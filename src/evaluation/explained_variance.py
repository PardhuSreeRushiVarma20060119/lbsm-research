"""
explained_variance.py
=====================
LBSM — Evaluation
------------------
Explained variance utilities for PCA and dimensionality diagnostics.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def pca_explained_variance(
    X           : np.ndarray,
    n_components: int = 6,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fit PCA and return per-component explained variance table."""
    pca  = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X)
    ev   = pca.explained_variance_ratio_
    cumev = np.cumsum(ev)
    return pd.DataFrame({
        "component"       : [f"PC{i+1}" for i in range(n_components)],
        "explained_var"   : ev,
        "cumulative_var"  : cumev,
    })


def n_components_for_threshold(
    explained_var: np.ndarray,
    threshold    : float = 0.90,
) -> int:
    """Minimum number of PCs to exceed a cumulative variance threshold."""
    cumulative = np.cumsum(explained_var)
    idx = np.searchsorted(cumulative, threshold)
    return int(idx) + 1


def intrinsic_dimensionality_estimate(
    X   : np.ndarray,
    seed: int = 42,
) -> float:
    """Participation ratio estimate of intrinsic dimensionality.

    PR = (Σλ_i)² / Σλ_i²   — ranges from 1 (rank-1) to d (uniform).
    """
    pca = PCA(random_state=seed)
    pca.fit(X)
    ev = pca.explained_variance_
    return float(ev.sum() ** 2 / (ev ** 2).sum())