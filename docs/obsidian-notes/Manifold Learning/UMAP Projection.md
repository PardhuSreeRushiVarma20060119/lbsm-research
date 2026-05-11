## tags: [lbsm, manifold-learning, umap] module: umap_projection.py notebook: 02_manifold_learning.ipynb §4

# UMAP Projection

## Role

UMAP is the **primary nonlinear embedding method** in LBSM. Unlike PCA, it preserves local topology, revealing the curvature in the latent behavioral manifold.

## Configuration Used

```python
umap_result_2d = fit_umap(X, n_components=2, n_neighbors=30, min_dist=0.10, random_state=42)
umap_result_3d = fit_umap(X, n_components=3, n_neighbors=30, min_dist=0.10, random_state=42)
```

## `UMAPResult` Container

```python
@dataclass
class UMAPResult:
    embedding    : np.ndarray   # (N, n_components)
    n_neighbors  : int
    min_dist     : float
    n_components : int
    reducer      : umap.UMAP    # fitted, for transform of new data
```

Outputs saved to `data/X_umap2.npy` and `data/X_umap3.npy`.

## Hyperparameter Sweep

```python
sweep_df = hyperparameter_sweep(X, y,
    n_neighbors_grid = (10, 30, 50, 100),
    min_dist_grid    = (0.05, 0.10, 0.25, 0.50),
)
```

Grid search over 16 (n_neighbors, min_dist) combinations, scored by silhouette coefficient. Sorted descending by silhouette.

## Key Functions (`umap_projection.py`)

|Function|Purpose|
|---|---|
|`fit_umap(X, ...)`|Core UMAP fit|
|`hyperparameter_sweep(X, labels)`|Grid search with silhouette scoring|
|`per_regime_density(embedding, labels)`|KDE density estimate per regime in 2-D|
|`regime_connectivity(embedding, labels, k)`|k-NN cross-regime edge fraction (boundary porosity)|

## Regime Connectivity (Boundary Porosity)

```python
connectivity = regime_connectivity(E_umap2, y, PROFILE_NAMES, k=20)
```

Entry `[i,j]` = fraction of regime-i points whose k-NN includes ≥1 regime-j point. Higher → more porous boundary.

From Notebook 02 §4.4: stable–adaptive and exploratory–adaptive boundaries are more porous than stable–unstable.

## Geometry Observed

- **Stable** → compact cluster
- **Exploratory** → dispersed cluster at opposite pole
- **Adaptive** → bridge manifold connecting stable and exploratory
- **Unstable** → geometrically separate diffuse region

## See Also

- [[Manifold Learning/PCA Analysis]]
- [[Manifold Learning/tSNE Validation]]
- [[Manifold Learning/Transitional Geometry]]
- [[Manifold Learning/Regime Boundaries]]
- [[Metrics/Topological Preservation]]
- [[Trajectory Geometry/Trajectory Overview]]