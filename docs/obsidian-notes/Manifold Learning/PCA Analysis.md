## tags: [lbsm, manifold-learning, pca, linear] module: pca.py notebook: 02_manifold_learning.ipynb §3

# PCA Analysis

## Role in LBSM

PCA is the **linear baseline**. Its role is not to compete with UMAP/t-SNE but to:

1. Anchor the comparison — quantify what linear projection achieves
2. Reveal which features dominate PC1 (the healthy/unstable axis)
3. Show why higher PCs are needed for intra-healthy separation

## Key Results (Notebook 01 Full-Dataset PCA)

```
PC1 explains 80.3% of variance  ← healthy/unstable axis
PC2 explains X.X%
PC3 explains X.X%
Components for 90% variance: n_components_90
```

## Key Results (Notebook 02 Balanced Subsample)

```python
pca_result = fit_pca(X, TELEMETRY_FEATURES, n_components=6, random_state=42)
```

PCA scorecard (from `embedding_scorecard`):

- Lower silhouette than UMAP → confirms nonlinear structure exists

## `PCAResult` Container

```python
@dataclass
class PCAResult:
    embedding       : np.ndarray      # (N, n_components)
    explained_var   : np.ndarray      # per-PC fraction
    cumulative_var  : np.ndarray      # cumulative fraction
    loadings        : pd.DataFrame    # (n_features, n_components)
    pca_model       : PCA             # fitted sklearn object
    n_components_90 : int             # PCs needed for 90% variance
```

## Key Functions (`pca.py`)

|Function|Purpose|
|---|---|
|`fit_pca(X, feature_names)`|Fit PCA, return `PCAResult`|
|`regime_centroids_pca(embedding, labels)`|Per-regime centroids in PC space|
|`inter_regime_pc_distances(embedding, labels)`|Pairwise centroid distances|
|`loading_dominance(loadings, pc)`|Features ranked by|
|`print_pca_summary(result)`|Pretty-print diagnostics|

## Loading Interpretation

PC1 loadings reveal the dominant feature axis separating unstable from healthy:

- Features with high |PC1 loading| → primary discriminators
- Use `loading_dominance(loadings, "PC1")` to rank

## Scree Plot Data

`explained_var` and `cumulative_var` arrays → feed directly into scree + cumulative variance plots (Notebook 02 §3, Fig: scree).

## See Also

- [[Manifold Learning/UMAP Projection]]
- [[Manifold Learning/tSNE Validation]]
- [[Behavioral Regimes/Separability Theory]]
- [[Metrics/Topological Preservation]]
- [[Results/Regime Separability]]