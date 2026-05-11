## tags: [lbsm, metrics, topology, trustworthiness, continuity] module: manifold_metrics.py notebook: 02_manifold_learning.ipynb §6

# Topological Preservation

## Two Complementary Metrics

### Trustworthiness

Penalises **false neighbors** — points that are close in the embedding but were far in the original space.

$$\text{Trustworthiness} \in (0, 1]$$

Higher is better.

### Continuity

Penalises **missing neighbors** — points that were close in the original space but are now far in the embedding.

```python
# Implementation in manifold_metrics.py:
# Continuity == trustworthiness with X_high and X_embedded swapped
continuity = trustworthiness(X_embedded, X_high, n_neighbors=n_neighbors)
```

$$\text{Continuity} \in (0, 1]$$

Higher is better.

## Full Scorecard (`embedding_scorecard`)

```python
scores = embedding_scorecard(X_high, X_embedded, labels,
    method_name  = "UMAP (2-D)",
    sample_size  = 5_000,
    n_neighbors  = 10,
    random_state = 42,
)
```

Returns dict with keys:

|Metric|Direction|Meaning|
|---|---|---|
|`silhouette`|↑ better|Cluster separation in embedding|
|`davies_bouldin`|↓ better|Compactness / separation ratio|
|`calinski_harabasz`|↑ better|Between/within cluster variance|
|`trustworthiness`|↑ better|No false neighbors|
|`continuity`|↑ better|No missing neighbors|

## Three-Way Comparison

```python
scorecard_df = compare_embeddings([sc_pca, sc_umap, sc_tsne])
```

Compared methods: PCA (PC1-2), UMAP (2-D), t-SNE (5k sample).

Expected ranking: UMAP > t-SNE > PCA on most metrics.

## Per-Regime Silhouette

```python
sil_df = per_regime_silhouette(X_embedded, labels, PROFILE_NAMES)
# Returns: DataFrame [regime, mean_silhouette, std_silhouette, n_samples]
```

Low adaptive silhouette = expected (bridge manifold). Low silhouette for _stable_ or _unstable_ = problem.

## Neighbourhood Purity

```python
purity_df = neighbourhood_purity(X_embedded, labels, PROFILE_NAMES, k=20)
# Entry: fraction of k-NN sharing the same regime
```

UMAP advantage = `purity_UMAP.mean() - purity_PCA.mean()`.

## Cross-Method Agreement

```python
r = embedding_agreement(E_umap_sample, E_tsne, y_tsne, sample_size=2000)
```

Procrustes-aligned Pearson r of pairwise distances.

## See Also

- [[Manifold Learning/PCA Analysis]]
- [[Manifold Learning/UMAP Projection]]
- [[Manifold Learning/tSNE Validation]]
- [[Metrics/Tortuosity Index]]
- [[Experiments/Embedding Validation]]
- [[Results/Regime Separability]]