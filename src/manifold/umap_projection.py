"""
umap_projection.py
==================
Latent Behavioral State Machine (LBSM) — Manifold Analysis
-----------------------------------------------------------
UMAP (Uniform Manifold Approximation and Projection) embeddings of
the LBSM agent telemetry feature space.

UMAP is the *primary* nonlinear embedding method in the LBSM paper.
Unlike PCA it preserves local topology, allowing the three healthy
behavioral regimes (stable / exploratory / adaptive) to express their
latent curvature—not just their linear projections.

Module responsibilities
-----------------------
1. ``fit_umap``        — single embedding (2-D or 3-D)
2. ``hyperparameter_sweep`` — grid over (n_neighbors, min_dist) and return
                              silhouette scores for each setting
3. ``umap_per_regime_density`` — KDE density estimate in 2-D embedding
4. ``regime_connectivity``     — fraction of k-NN cross-regime edges
                                 (quantifies boundary porosity)

Reference
---------
"Latent Behavioral State Machines: Manifold Geometry of Adaptive Agent Telemetry"
Section 5.2 — Nonlinear Manifold Geometry: UMAP
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import umap as umap_lib
    _UMAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _UMAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class UMAPResult:
    """UMAP embedding result.

    Attributes
    ----------
    embedding     : np.ndarray  shape (N, n_components)
    n_neighbors   : int         — hyperparameter used
    min_dist      : float       — hyperparameter used
    n_components  : int
    reducer       : fitted UMAP object (for transform of new data)
    """

    embedding    : np.ndarray
    n_neighbors  : int
    min_dist     : float
    n_components : int
    reducer      : object   # umap.UMAP instance


# ---------------------------------------------------------------------------
# Core fit
# ---------------------------------------------------------------------------
def fit_umap(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 30,
    min_dist: float = 0.10,
    metric: str = "euclidean",
    random_state: int = 42,
    verbose: bool = False,
) -> UMAPResult:
    """Fit a UMAP embedding.

    Parameters
    ----------
    X            : np.ndarray  shape (N, d)  — z-scored feature matrix
    n_components : 2 for visualisation, 3 for surface plots
    n_neighbors  : UMAP local neighbourhood size (larger → more global)
    min_dist     : minimum distance in embedded space (smaller → tighter clusters)
    metric       : distance metric (``"euclidean"`` for standardised features)
    random_state : reproducibility seed
    verbose      : pass to UMAP (progress logging)

    Returns
    -------
    result : UMAPResult
    """
    if not _UMAP_AVAILABLE:
        raise ImportError(
            "umap-learn is required.  Install with: pip install umap-learn"
        )

    reducer = umap_lib.UMAP(
        n_components  = n_components,
        n_neighbors   = n_neighbors,
        min_dist      = min_dist,
        metric        = metric,
        random_state  = random_state,
        verbose       = verbose,
        low_memory    = False,
    )
    embedding = reducer.fit_transform(X)

    return UMAPResult(
        embedding    = embedding,
        n_neighbors  = n_neighbors,
        min_dist     = min_dist,
        n_components = n_components,
        reducer      = reducer,
    )


# ---------------------------------------------------------------------------
# Hyperparameter sweep
# ---------------------------------------------------------------------------
def hyperparameter_sweep(
    X: np.ndarray,
    labels: np.ndarray,
    n_neighbors_grid: Sequence[int]  = (10, 30, 50, 100),
    min_dist_grid   : Sequence[float] = (0.05, 0.10, 0.25, 0.50),
    random_state    : int = 42,
) -> pd.DataFrame:
    """Grid search over UMAP hyperparameters, scored by silhouette coefficient.

    For each (n_neighbors, min_dist) combination:
      1. Fit UMAP 2-D embedding
      2. Compute silhouette score on the embedded coordinates

    Parameters
    ----------
    X                : feature matrix (N, d), z-scored
    labels           : integer regime labels (N,)
    n_neighbors_grid : values of n_neighbors to test
    min_dist_grid    : values of min_dist to test
    random_state     : fixed seed for reproducibility

    Returns
    -------
    results : pd.DataFrame  columns = [n_neighbors, min_dist, silhouette]
              sorted by silhouette descending
    """
    from sklearn.metrics import silhouette_score

    rows = []
    for nn in n_neighbors_grid:
        for md in min_dist_grid:
            r = fit_umap(X, n_components=2, n_neighbors=nn,
                         min_dist=md, random_state=random_state)
            sil = float(silhouette_score(
                r.embedding, labels, sample_size=min(5000, len(labels)),
                random_state=random_state,
            ))
            rows.append({"n_neighbors": nn, "min_dist": md, "silhouette": sil})

    df = pd.DataFrame(rows).sort_values("silhouette", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Per-regime embedding density
# ---------------------------------------------------------------------------
def per_regime_density(
    embedding: np.ndarray,
    labels: np.ndarray,
    profile_names: Tuple[str, ...],
    grid_resolution: int = 100,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Estimate 2-D KDE density for each regime in UMAP space.

    Parameters
    ----------
    embedding      : np.ndarray  shape (N, 2)
    labels         : np.ndarray  shape (N,)  integer indices
    profile_names  : ordered tuple of regime names
    grid_resolution: KDE evaluation grid density

    Returns
    -------
    densities : dict  regime_name → (xx, yy, zz)
        where xx, yy are meshgrid arrays and zz is the KDE density.
    """
    from scipy.stats import gaussian_kde

    x_min, x_max = embedding[:, 0].min() - 0.5, embedding[:, 0].max() + 0.5
    y_min, y_max = embedding[:, 1].min() - 0.5, embedding[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid_pts = np.vstack([xx.ravel(), yy.ravel()])

    densities: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for idx, name in enumerate(profile_names):
        pts = embedding[labels == idx].T
        if pts.shape[1] < 4:
            continue
        kde = gaussian_kde(pts, bw_method=0.15)
        zz  = kde(grid_pts).reshape(xx.shape)
        densities[name] = (xx, yy, zz)

    return densities


# ---------------------------------------------------------------------------
# Regime connectivity (boundary porosity)
# ---------------------------------------------------------------------------
def regime_connectivity(
    embedding: np.ndarray,
    labels: np.ndarray,
    profile_names: Tuple[str, ...],
    k: int = 15,
) -> pd.DataFrame:
    """Compute fraction of k-NN edges crossing regime boundaries.

    A higher cross-regime fraction → more porous boundary → more overlap.

    Parameters
    ----------
    embedding     : np.ndarray  shape (N, 2)
    labels        : np.ndarray  shape (N,)
    profile_names : regime names in label order
    k             : neighbourhood size

    Returns
    -------
    pairwise_porosity : pd.DataFrame  shape (n_regimes, n_regimes)
        Entry [i,j] = fraction of regime-i points whose k-NN includes ≥1 regime-j point
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
    nn.fit(embedding)
    _, indices = nn.kneighbors(embedding)
    # Skip self (index 0)
    neighbor_labels = labels[indices[:, 1:]]  # shape (N, k)

    n_reg = len(profile_names)
    porosity = np.zeros((n_reg, n_reg))

    for i in range(n_reg):
        src_mask = labels == i
        src_neigh = neighbor_labels[src_mask]  # (n_i, k)
        n_i = src_mask.sum()
        if n_i == 0:
            continue
        for j in range(n_reg):
            cross = (src_neigh == j).any(axis=1).sum()
            porosity[i, j] = cross / n_i

    return pd.DataFrame(
        porosity,
        index   = list(profile_names),
        columns = list(profile_names),
    )