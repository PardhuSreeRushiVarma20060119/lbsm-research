# tests/test_manifold.py
def test_pca_explained_variance():
    result = fit_pca(X_synthetic, feature_names)
    assert result.explained_var.sum() <= 1.0001  # Allow floating-point error
    assert result.cumulative_var[-1] <= 1.0001


def test_umap_hyperparameter_sweep():
    sweep_result = hyperparameter_sweep(X_synthetic, labels, ...)
    assert "n_neighbors" in sweep_result.columns
    assert "min_dist" in sweep_result.columns
    assert "silhouette" in sweep_result.columns


def test_embedding_scorecard_keys():
    scorecard = embedding_scorecard(X_high, X_embedded, labels)
    required_keys = [
        "method",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "trustworthiness",
        "continuity",
    ]
    assert all(k in scorecard for k in required_keys)
