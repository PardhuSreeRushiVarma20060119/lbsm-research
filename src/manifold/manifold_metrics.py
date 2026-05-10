"""
manifold_metrics.py
===================
Latent Behavioral State Machine (LBSM) — Manifold Analysis
-----------------------------------------------------------
Quantitative evaluation of manifold embedding quality.

Without this module Notebook 02 is a gallery of pretty plots.
These metrics transform visual impressions into falsifiable claims:

  - **Silhouette coefficient**   — cluster separation in embedding space
  - **Davies-Bouldin index**     — cluster compactness / separation ratio
  - **Calinski-Harabasz score**  — between/within cluster variance ratio
  - **Trustworthiness**          — local neighborhood preservation
  - **Continuity**               — inverse: no local tears in the embedding
  - **Cross-method agreement**   — Procrustes-aligned embedding correlation

Reference
---------
"Latent Behavioral State Machines: Manifold Geometry of Adaptive Agent Telemetry"
Section 5.4 — Quantitative Manifold Evaluation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
    silhouette_samples,
)
from sklearn.manifold import trustworthiness


# ---------------------------------------------------------------------------
# Single embedding scorecard
# ---------------------------------------------------------------------------
def embedding_scorecard(
    X_high    : np.ndarray,
    X_embedded: np.ndarray,
    labels    : np.ndarray,
    method_name : str = "embedding",
    sample_size : int = 5_000,
    n_neighbors : int = 10,
    random_state: int = 42,
) -> Dict[str, float]:
    """Compute a full suite of quality metrics for one embedding.

    Parameters
    ----------
    X_high      : original high-dimensional feature matrix (N, d)
    X_embedded  : low-dimensional embedding (N, k)
    labels      : integer regime labels (N,)
    method_name : label for this embedding (used in summary tables)
    sample_size : subsample for silhouette (speeds up computation for N > 10k)
    n_neighbors : neighbourhood size for trustworthiness / continuity
    random_state: reproducibility seed

    Returns
    -------
    scores : dict  metric_name → float
    """
    # ── Cluster quality (embedding space)
    sil = float(silhouette_score(
        X_embedded, labels,
        sample_size  = min(sample_size, len(labels)),
        random_state = random_state,
    ))
    db  = float(davies_bouldin_score(X_embedded, labels))
    ch  = float(calinski_harabasz_score(X_embedded, labels))

    # ── Topology preservation
    n = min(3_000, len(X_high))
    tw  = float(trustworthiness(X_high[:n], X_embedded[:n], n_neighbors=n_neighbors))
    con = float(continuity(X_high[:n], X_embedded[:n], n_neighbors=n_neighbors))

    return {
        "method"         : method_name,
        "silhouette"     : sil,       # ↑ better  (range −1 to 1)
        "davies_bouldin" : db,        # ↓ better
        "calinski_harabasz": ch,      # ↑ better
        "trustworthiness": tw,        # ↑ better  (range 0 to 1)
        "continuity"     : con,       # ↑ better  (range 0 to 1)
    }


def compare_embeddings(
    scorecards: List[Dict[str, float]],
) -> pd.DataFrame:
    """Build a comparison DataFrame from a list of scorecards.

    Parameters
    ----------
    scorecards : list of dicts returned by :func:`embedding_scorecard`

    Returns
    -------
    df : pd.DataFrame  shape (n_methods, n_metrics)
         with method as the index
    """
    df = pd.DataFrame(scorecards).set_index("method")

    # Direction annotations for display
    df.attrs["higher_is_better"] = {
        "silhouette"       : True,
        "davies_bouldin"   : False,
        "calinski_harabasz": True,
        "trustworthiness"  : True,
        "continuity"       : True,
    }
    return df


# ---------------------------------------------------------------------------
# Continuity (complement of trustworthiness)
# ---------------------------------------------------------------------------
def continuity(
    X_high    : np.ndarray,
    X_embedded: np.ndarray,
    n_neighbors: int = 10,
) -> float:
    """Continuity score — measures whether neighbors in the embedding were
    also neighbors in the original space.

    Trustworthiness penalises false neighbors (points brought close that were
    far in original space). Continuity penalises missing neighbors (points
    that were close but are now far).

    Implementation mirrors sklearn's trustworthiness with transposed roles.

    Returns
    -------
    continuity : float ∈ (0, 1]  (higher is better)
    """
    # Continuity == trustworthiness with X_high and X_embedded swapped
    return float(trustworthiness(X_embedded, X_high, n_neighbors=n_neighbors))


# ---------------------------------------------------------------------------
# Per-regime silhouette
# ---------------------------------------------------------------------------
def per_regime_silhouette(
    X_embedded: np.ndarray,
    labels    : np.ndarray,
    profile_names: Tuple[str, ...],
) -> pd.DataFrame:
    """Compute mean silhouette coefficient per regime.

    A low silhouette for a specific regime (e.g. adaptive) while others
    are high confirms that *that* regime is the boundary case — motivating
    targeted analysis.

    Returns
    -------
    df : pd.DataFrame  columns = [regime, mean_silhouette, std_silhouette, n_samples]
    """
    sil_samples = silhouette_samples(X_embedded, labels)
    rows = []
    for idx, name in enumerate(profile_names):
        mask = labels == idx
        s = sil_samples[mask]
        rows.append({
            "regime"         : name,
            "mean_silhouette": float(s.mean()),
            "std_silhouette" : float(s.std()),
            "n_samples"      : int(mask.sum()),
        })
    return pd.DataFrame(rows).set_index("regime")


# ---------------------------------------------------------------------------
# Cross-method agreement (Procrustes)
# ---------------------------------------------------------------------------
def embedding_agreement(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 3_000,
    random_state: int = 42,
) -> float:
    """Measure structural agreement between two 2-D embeddings via Procrustes.

    Aligns emb_b to emb_a via optimal rotation + scaling, then computes
    the Pearson correlation of pairwise distances in both spaces.

    Higher correlation → the two methods agree on the global structure.

    Returns
    -------
    r : float  Pearson r of pairwise distance vectors
    """
    from scipy.spatial.distance import pdist
    from scipy.stats import pearsonr

    rng = np.random.default_rng(random_state)
    n   = min(sample_size, len(emb_a))
    idx = rng.choice(len(emb_a), n, replace=False)

    a = emb_a[idx]
    b = emb_b[idx]

    # Procrustes alignment
    from scipy.spatial import procrustes as _procrustes
    _, b_aligned, _ = _procrustes(a, b)

    da = pdist(a)
    db = pdist(b_aligned)

    r, _ = pearsonr(da, db)
    return float(r)


# ---------------------------------------------------------------------------
# Neighbourhood purity
# ---------------------------------------------------------------------------
def neighbourhood_purity(
    X_embedded  : np.ndarray,
    labels      : np.ndarray,
    profile_names: Tuple[str, ...],
    k           : int = 20,
) -> pd.DataFrame:
    """Compute per-regime k-NN purity in the embedding space.

    Purity(i) = fraction of k nearest neighbors sharing regime i.
    Averaged across all regime-i points.

    Returns
    -------
    df : pd.DataFrame  columns = [regime, purity, n_samples]
    """
    from sklearn.neighbors import NearestNeighbors

    nn  = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X_embedded)
    _, indices = nn.kneighbors(X_embedded)
    neighbor_labels = labels[indices[:, 1:]]  # drop self

    rows = []
    for idx, name in enumerate(profile_names):
        mask   = labels == idx
        neighs = neighbor_labels[mask]   # (n_i, k)
        purity = (neighs == idx).mean()
        rows.append({
            "regime"   : name,
            "purity"   : float(purity),
            "n_samples": int(mask.sum()),
        })
    return pd.DataFrame(rows).set_index("regime")