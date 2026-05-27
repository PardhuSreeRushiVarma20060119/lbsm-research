# tests/test_manifold.py

import numpy as np

from src.manifold.pca import fit_pca
from src.manifold.umap_projection import hyperparameter_sweep
from src.manifold.manifold_metrics import embedding_scorecard


# ---------------------------------------------------------------------------
# Synthetic test dataset
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)

X_synthetic = rng.normal(size=(500, 6))

labels = rng.integers(0, 4, size=500)

feature_names = [
    "latency",
    "entropy",
    "cpu",
    "memory",
    "throughput",
    "error_rate",
]

X_high = X_synthetic
X_embedded = X_synthetic[:, :2]


# ---------------------------------------------------------------------------
# PCA tests
# ---------------------------------------------------------------------------

def test_pca_explained_variance():

    result = fit_pca(X_synthetic, feature_names)

    print("\n=== PCA RESULT ===")

    print("Explained variance ratios:")
    print(result.explained_var)

    print("\nCumulative explained variance:")
    print(result.cumulative_var)

    print("\nTotal explained variance:")
    print(result.explained_var.sum())

    assert result.explained_var.sum() <= 1.0001
    assert result.cumulative_var[-1] <= 1.0001


# ---------------------------------------------------------------------------
# UMAP hyperparameter sweep
# ---------------------------------------------------------------------------

def test_umap_hyperparameter_sweep():

    sweep_result = hyperparameter_sweep(X_synthetic, labels)

    print("\n=== UMAP HYPERPARAMETER SWEEP ===")

    print(sweep_result.head())

    best_row = sweep_result.iloc[0]

    print("\nBest configuration:")
    print(best_row)

    assert "n_neighbors" in sweep_result.columns
    assert "min_dist" in sweep_result.columns
    assert "silhouette" in sweep_result.columns


# ---------------------------------------------------------------------------
# Embedding quality metrics
# ---------------------------------------------------------------------------

def test_embedding_scorecard_keys():

    scorecard = embedding_scorecard(
        X_high,
        X_embedded,
        labels,
    )

    print("\n=== EMBEDDING SCORECARD ===")

    for k, v in scorecard.items():
        print(f"{k:20s}: {v}")

    required_keys = [
        "method",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "trustworthiness",
        "continuity",
    ]

    assert all(k in scorecard for k in required_keys)