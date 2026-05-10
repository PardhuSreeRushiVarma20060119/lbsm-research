"""
tsne.py
=======
Latent Behavioral State Machine (LBSM) — Manifold Analysis
-----------------------------------------------------------
t-SNE (t-distributed Stochastic Neighbor Embedding) embeddings
of the LBSM telemetry feature space.

Role in Notebook 02
-------------------
t-SNE is the *validation* nonlinear method. It complements UMAP by:
  - Emphasising **local** cluster structure over global topology
  - Revealing within-regime sub-clusters that UMAP may compress
  - Exposing artefacts: if UMAP and t-SNE disagree strongly on
    cluster membership, the structure is fragile

Key difference from UMAP
------------------------
t-SNE distances in the embedding are NOT interpretable globally—only
local neighborhoods are preserved. Use it for cluster *shape*, not
inter-cluster *distance*.

Reference
---------
"Latent Behavioral State Machines: Manifold Geometry of Adaptive Agent Telemetry"
Section 5.3 — Comparative Validation: t-SNE
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class TSNEResult:
    """t-SNE embedding result.

    Attributes
    ----------
    embedding    : np.ndarray  shape (N_sample, 2)
    sample_idx   : np.ndarray  shape (N_sample,)  indices into the original X
    perplexity   : float
    kl_divergence: float  final KL divergence (lower = better fit)
    """

    embedding    : np.ndarray
    sample_idx   : np.ndarray
    perplexity   : float
    kl_divergence: float


# ---------------------------------------------------------------------------
# Core fit
# ---------------------------------------------------------------------------
def fit_tsne(
    X: np.ndarray,
    labels: Optional[np.ndarray] = None,
    n_sample: int = 5_000,
    perplexity: float = 50.0,
    max_iter: int = 1_000,
    pca_init_components: int = 50,
    random_state: int = 42,
    stratified: bool = True,
) -> TSNEResult:
    """Fit a 2-D t-SNE embedding on a stratified subsample of X.

    t-SNE is O(N²) and is practical only on ≤ 10k points; we subsample
    and optionally stratify to ensure all regime classes are represented.

    Parameters
    ----------
    X                   : np.ndarray  shape (N, d)  z-scored feature matrix
    labels              : np.ndarray  shape (N,)  integer labels (for stratified sampling)
    n_sample            : subsample size
    perplexity          : t-SNE perplexity — effective nearest-neighbor count
    max_iter            : optimisation iterations
    pca_init_components : PCA dimensionality before t-SNE (``"pca"`` init)
    random_state        : reproducibility seed
    stratified          : if True and labels given, sample equally per class

    Returns
    -------
    result : TSNEResult
    """
    rng = np.random.default_rng(random_state)
    N   = len(X)
    n_sample = min(n_sample, N)

    # ── Subsample
    if stratified and labels is not None:
        classes = np.unique(labels)
        per_class = n_sample // len(classes)
        idx_parts = []
        for c in classes:
            c_idx = np.where(labels == c)[0]
            chosen = rng.choice(c_idx, min(per_class, len(c_idx)), replace=False)
            idx_parts.append(chosen)
        sample_idx = np.concatenate(idx_parts)
        rng.shuffle(sample_idx)
    else:
        sample_idx = rng.choice(N, n_sample, replace=False)

    X_sub = X[sample_idx]

    # ── PCA pre-reduction (reduces noise, speeds up t-SNE)
    if pca_init_components is not None and X_sub.shape[1] > pca_init_components:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_init_components, random_state=random_state)
        X_sub = pca.fit_transform(X_sub)

    # ── t-SNE
    tsne = TSNE(
        n_components  = 2,
        perplexity    = perplexity,
        max_iter      = max(250, max_iter),   # sklearn requires ≥ 250
        init          = "pca",
        random_state  = random_state,
        n_jobs        = 1,         # determinism
    )
    embedding = tsne.fit_transform(X_sub)

    return TSNEResult(
        embedding     = embedding,
        sample_idx    = sample_idx,
        perplexity    = perplexity,
        kl_divergence = float(tsne.kl_divergence_),
    )


# ---------------------------------------------------------------------------
# Perplexity sensitivity analysis
# ---------------------------------------------------------------------------
def perplexity_sweep(
    X: np.ndarray,
    labels: np.ndarray,
    perplexity_grid: Sequence[float] = (10.0, 30.0, 50.0, 100.0),
    n_sample: int = 3_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate t-SNE silhouette score across perplexity values.

    Parameters
    ----------
    X               : feature matrix (z-scored)
    labels          : integer regime labels
    perplexity_grid : perplexity values to test
    n_sample        : subsample size (same across sweeps for comparability)
    random_state    : fixed seed

    Returns
    -------
    df : pd.DataFrame  columns = [perplexity, silhouette, kl_divergence]
    """
    from sklearn.metrics import silhouette_score

    rows = []
    for perp in perplexity_grid:
        result = fit_tsne(
            X, labels,
            n_sample     = n_sample,
            perplexity   = perp,
            max_iter     = 500,
            random_state = random_state,
            stratified   = True,
        )
        sil = float(silhouette_score(result.embedding, labels[result.sample_idx]))
        rows.append({
            "perplexity"   : perp,
            "silhouette"   : sil,
            "kl_divergence": result.kl_divergence,
        })

    return pd.DataFrame(rows).sort_values("silhouette", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Intra-regime compactness
# ---------------------------------------------------------------------------
def intra_regime_spread(
    embedding: np.ndarray,
    labels: np.ndarray,
    profile_names: Tuple[str, ...],
) -> pd.DataFrame:
    """Compute mean intra-regime distance from centroid in t-SNE space.

    A smaller spread indicates a tighter, more cohesive cluster.

    Returns
    -------
    df : pd.DataFrame  columns = [regime, centroid_x, centroid_y, mean_radius, std_radius]
    """
    rows = []
    for idx, name in enumerate(profile_names):
        pts = embedding[labels == idx]
        if len(pts) == 0:
            continue
        centroid = pts.mean(axis=0)
        radii    = np.linalg.norm(pts - centroid, axis=1)
        rows.append({
            "regime"      : name,
            "centroid_x"  : float(centroid[0]),
            "centroid_y"  : float(centroid[1]),
            "mean_radius" : float(radii.mean()),
            "std_radius"  : float(radii.std()),
        })
    return pd.DataFrame(rows).set_index("regime")