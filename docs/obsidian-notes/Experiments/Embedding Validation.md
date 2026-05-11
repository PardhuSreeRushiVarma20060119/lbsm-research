## tags: [lbsm, experiments, embedding-validation] notebook: 02_manifold_learning.ipynb §6, §8

# Embedding Validation

## Purpose

Without quantitative validation, Notebook 02 would be a gallery of pretty plots. These metrics transform visual impressions into **falsifiable claims**.

## Three-Method Comparison

Methods validated: PCA (PC1-2), UMAP (2-D), t-SNE (5k stratified sample)

```python
sc_pca  = embedding_scorecard(X, E_pca,   y, method_name="PCA (PC1-2)")
sc_umap = embedding_scorecard(X, E_umap2, y, method_name="UMAP (2-D)")
sc_tsne = embedding_scorecard(X[tsne_idx], E_tsne, y_tsne, method_name="t-SNE (5k sample)")

scorecard_df = compare_embeddings([sc_pca, sc_umap, sc_tsne])
```

## Expected Ranking

|Metric|Expected Winner|
|---|---|
|Silhouette|UMAP|
|Davies-Bouldin|UMAP (lower is better)|
|Calinski-Harabasz|UMAP|
|Trustworthiness|UMAP|
|Continuity|UMAP|

PCA expected worst overall; t-SNE intermediate.

## Cross-Method Agreement (§8)

```python
r = embedding_agreement(E_umap_sample, E_tsne, y_tsne, sample_size=2000)
```

Procrustes-aligned Pearson r. If `r > 0.60` → structure is robust across methods.

## Perplexity Sensitivity (t-SNE)

```python
sweep = perplexity_sweep(X, y, perplexity_grid=(10, 30, 50, 100), n_sample=3000)
```

Tests if t-SNE cluster topology is stable across perplexity settings.

## Hyperparameter Sweep (UMAP)

```python
sweep_df = hyperparameter_sweep(X, y,
    n_neighbors_grid = (10, 30, 50, 100),
    min_dist_grid    = (0.05, 0.10, 0.25, 0.50),
)
```

Best (n_neighbors, min_dist) combination by silhouette — confirms robustness of chosen configuration (30, 0.10).

## See Also

- [[Metrics/Topological Preservation]]
- [[Manifold Learning/UMAP Projection]]
- [[Manifold Learning/tSNE Validation]]
- [[Results/Regime Separability]]