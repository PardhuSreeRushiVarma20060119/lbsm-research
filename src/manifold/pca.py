"""
pca.py
======
Latent Behavioral State Machine (LBSM) — Manifold Analysis
-----------------------------------------------------------
Principal Component Analysis baseline for the LBSM telemetry feature space.

Provides:
  - Full PCA embedding with explained-variance diagnostics
  - Scree plot data generation
  - Loading matrix extraction and interpretation
  - PC-space regime centroid computation
  - Comparison baseline for nonlinear embeddings (UMAP, t-SNE)

Design note
-----------
PCA is deliberately kept as the *linear* baseline. Its role in Notebook 02 is
not to compete with UMAP/t-SNE but to:
  1. Anchor the comparison (quantify what linear projection achieves)
  2. Reveal which features dominate PC1 (the healthy/unstable axis)
  3. Show why higher PCs are needed for intra-healthy separation

Reference
---------
"Latent Behavioral State Machines: Manifold Geometry of Adaptive Agent Telemetry"
Section 5.1 — Linear Baseline: Principal Component Analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class PCAResult:
    """Full PCA result container.

    Attributes
    ----------
    embedding      : np.ndarray  shape (N, n_components)
        Projected coordinates in PC space.
    explained_var  : np.ndarray  shape (n_components,)
        Fraction of variance explained by each PC.
    cumulative_var : np.ndarray  shape (n_components,)
        Cumulative explained variance.
    loadings       : pd.DataFrame  shape (n_features, n_components)
        Feature loadings (eigenvectors) — rows = features, cols = PCs.
    pca_model      : sklearn PCA
        Fitted sklearn PCA object for transform / inverse_transform.
    n_components_90: int
        Minimum number of components to exceed 90% cumulative variance.
    """

    embedding      : np.ndarray
    explained_var  : np.ndarray
    cumulative_var : np.ndarray
    loadings       : pd.DataFrame
    pca_model      : PCA
    n_components_90: int


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------
def fit_pca(
    X: np.ndarray,
    feature_names: Tuple[str, ...],
    n_components: Optional[int] = None,
    random_state: int = 42,
) -> PCAResult:
    """Fit PCA on a z-scored feature matrix and return a :class:`PCAResult`.

    Parameters
    ----------
    X            : np.ndarray  shape (N, d)
        Input feature matrix (should already be z-scored / standardised).
    feature_names: tuple of str
        Column names of *X* (length d).
    n_components : int | None
        Number of components to retain.  If None, keeps ``min(N, d)``.
    random_state : int
        Reproducibility seed passed to sklearn PCA.

    Returns
    -------
    result : PCAResult
    """
    n_comp = n_components or min(X.shape)

    pca = PCA(n_components=n_comp, random_state=random_state)
    embedding = pca.fit_transform(X)

    explained  = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # Minimum PCs to hit 90% variance
    n90 = int(np.searchsorted(cumulative, 0.90)) + 1

    # Loading matrix: shape (d, n_comp)
    loading_cols = [f"PC{i+1}" for i in range(n_comp)]
    loadings = pd.DataFrame(
        pca.components_.T,         # (d, n_comp) — features as rows
        index=list(feature_names),
        columns=loading_cols,
    )

    return PCAResult(
        embedding       = embedding,
        explained_var   = explained,
        cumulative_var  = cumulative,
        loadings        = loadings,
        pca_model       = pca,
        n_components_90 = n90,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def regime_centroids_pca(
    embedding: np.ndarray,
    labels: np.ndarray,
    profile_names: Tuple[str, ...],
    n_pcs: int = 3,
) -> pd.DataFrame:
    """Compute per-regime centroids in PC space.

    Parameters
    ----------
    embedding     : np.ndarray  shape (N, n_pcs)
    labels        : np.ndarray  shape (N,)  integer regime indices
    profile_names : ordered tuple of regime name strings
    n_pcs         : how many PCs to include in the output

    Returns
    -------
    centroids : pd.DataFrame  shape (n_regimes, n_pcs)
        Rows = regimes, Columns = PC1, PC2, …
    """
    cols = [f"PC{i+1}" for i in range(n_pcs)]
    rows = {}
    for idx, name in enumerate(profile_names):
        mask = labels == idx
        rows[name] = embedding[mask, :n_pcs].mean(axis=0)
    return pd.DataFrame(rows, index=cols).T


def inter_regime_pc_distances(
    embedding: np.ndarray,
    labels: np.ndarray,
    profile_names: Tuple[str, ...],
    n_pcs: int = 2,
) -> pd.DataFrame:
    """Pairwise Euclidean distances between regime centroids in PC space.

    Returns
    -------
    D : pd.DataFrame  shape (n_regimes, n_regimes)
    """
    centroids = regime_centroids_pca(embedding, labels, profile_names, n_pcs)
    k = len(profile_names)
    D = np.zeros((k, k))
    vals = centroids.values
    for i in range(k):
        for j in range(k):
            D[i, j] = np.linalg.norm(vals[i] - vals[j])
    return pd.DataFrame(D, index=list(profile_names), columns=list(profile_names))


def loading_dominance(loadings: pd.DataFrame, pc: str = "PC1") -> pd.Series:
    """Return features sorted by absolute loading on a given PC.

    Parameters
    ----------
    loadings : pd.DataFrame  from :func:`fit_pca`
    pc       : e.g. ``"PC1"``, ``"PC2"``

    Returns
    -------
    ranked : pd.Series  feature → |loading|, descending
    """
    return loadings[pc].abs().sort_values(ascending=False)


def print_pca_summary(result: PCAResult, top_n: int = 6) -> None:  # pragma: no cover
    """Pretty-print key PCA diagnostics to stdout."""
    print("=" * 54)
    print("PCA SUMMARY")
    print("=" * 54)
    print(f"{'PC':<6} {'Var Explained':>14} {'Cumulative':>12}")
    print("-" * 36)
    for i, (ev, cv) in enumerate(
        zip(result.explained_var, result.cumulative_var), start=1
    ):
        marker = "  ← 90%" if i == result.n_components_90 else ""
        print(f"PC{i:<4} {ev:>13.1%}  {cv:>11.1%}{marker}")
    print()
    print("Feature loadings (top PCs):")
    print(result.loadings.iloc[:, :min(3, result.loadings.shape[1])].round(4).to_string())
    print()
    print(f"Components needed for 90% variance: {result.n_components_90}")