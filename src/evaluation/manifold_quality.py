"""
manifold_quality.py
===================
LBSM — Evaluation
------------------
Manifold embedding quality metrics: trustworthiness, continuity,
neighbourhood purity, and cross-method agreement (Procrustes).
Used in NB02 and NB04.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness


def embedding_trustworthiness(
    X_high    : np.ndarray,
    X_embedded: np.ndarray,
    n_neighbors: int = 10,
) -> float:
    """Trustworthiness of a low-dimensional embedding."""
    n = min(3000, len(X_high))
    return float(trustworthiness(X_high[:n], X_embedded[:n], n_neighbors=n_neighbors))


def embedding_continuity(
    X_high    : np.ndarray,
    X_embedded: np.ndarray,
    n_neighbors: int = 10,
) -> float:
    """Continuity score (roles of high/low swapped vs trustworthiness)."""
    n = min(3000, len(X_high))
    return float(trustworthiness(X_embedded[:n], X_high[:n], n_neighbors=n_neighbors))


def neighbourhood_purity(
    X_embedded   : np.ndarray,
    labels       : np.ndarray,
    class_names  : Tuple[str, ...],
    k            : int = 20,
) -> pd.DataFrame:
    """Fraction of k-NN sharing the same class label, per class."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_embedded)
    _, idx = nn.kneighbors(X_embedded)
    neigh_labels = labels[idx[:, 1:]]
    rows = []
    for i, name in enumerate(class_names):
        mask   = labels == i
        purity = float((neigh_labels[mask] == i).mean())
        rows.append({"class": name, "purity": purity, "n": int(mask.sum())})
    return pd.DataFrame(rows).set_index("class")


def procrustes_agreement(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    n    : int = 2000,
    seed : int = 42,
) -> float:
    """Pearson r of pairwise distances after Procrustes alignment."""
    from scipy.spatial.distance import pdist
    from scipy.spatial import procrustes as _proc
    from scipy.stats import pearsonr
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(emb_a), min(n, len(emb_a)), replace=False)
    _, b_al, _ = _proc(emb_a[idx], emb_b[idx])
    r, _ = pearsonr(pdist(emb_a[idx]), pdist(b_al))
    return float(r)