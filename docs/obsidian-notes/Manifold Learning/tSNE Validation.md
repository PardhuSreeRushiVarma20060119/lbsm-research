## tags: [lbsm, manifold-learning, tsne, validation] module: tsne.py notebook: 02_manifold_learning.ipynb §5

# t-SNE Validation

## Role

t-SNE is the **validation nonlinear method**. It complements UMAP by:

- Emphasising **local** cluster structure over global topology
- Revealing within-regime sub-clusters that UMAP may compress
- Exposing artifacts: if UMAP and t-SNE disagree strongly, the structure is fragile

> ⚠️ t-SNE distances in the embedding are **NOT interpretable globally** — only local neighborhoods are preserved. Use for cluster _shape_, not inter-cluster _distance_.

## Configuration Used

```python
tsne_result = fit_tsne(
    X, y,
    n_sample   = 5_000,    # stratified subsample (t-SNE is O(N²))
    perplexity = 50.0,
    max_iter   = 1_000,
    stratified = True,
    random_state = 42,
)
```

KL divergence reported as fit quality measure.

## `TSNEResult` Container

```python
@dataclass
class TSNEResult:
    embedding    : np.ndarray   # (N_sample, 2)
    sample_idx   : np.ndarray   # (N_sample,) — indices into original X
    perplexity   : float
    kl_divergence: float        # lower = better fit
```

## Key Functions (`tsne.py`)

|Function|Purpose|
|---|---|
|`fit_tsne(X, labels, ...)`|Stratified t-SNE with PCA pre-reduction|
|`perplexity_sweep(X, labels)`|Silhouette + KL divergence across perplexities|
|`intra_regime_spread(embedding, labels)`|Mean radius from centroid per regime|

## Perplexity Sensitivity

```python
sweep = perplexity_sweep(X, y, perplexity_grid=(10, 30, 50, 100), n_sample=3000)
```

Tests structural robustness: does the cluster topology depend strongly on perplexity? Stable structure → valid findings.

## Cross-Method Agreement

```python
r = embedding_agreement(E_umap_sample, E_tsne, y_tsne, sample_size=2000)
```

Procrustes-aligned pairwise distance Pearson r between UMAP and t-SNE:

- `r > 0.60` → STRONG agreement
- `r > 0.35` → MODERATE agreement
- `r < 0.35` → WEAK agreement (check for artifacts)

## See Also

- [[Manifold Learning/UMAP Projection]]
- [[Metrics/Topological Preservation]]
- [[Experiments/Embedding Validation]]
- [[Results/Regime Separability]]