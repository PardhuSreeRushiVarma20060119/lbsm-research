## tags: [lbsm, manifold-learning, hypothesis]

# Latent Geometry Hypothesis

## Statement

> Adaptive agent telemetry forms **structured low-dimensional manifolds** embedded in high-dimensional telemetry space.

This is the geometric foundation of the entire LBSM framework.

## Why Manifolds Emerge

Four reasons behavioral manifolds form (rather than random volume-filling):

1. **Regimes constrain feature combinations** — not all (latency, entropy, reward, …) combinations are reachable; each regime permits only a local region
2. **Transitions are smooth** — AR(1) smoothing prevents discontinuous jumps
3. **Temporal continuity restricts motion** — trajectories are continuous curves, not random walks
4. **Adaptive dynamics are intrinsically low-dimensional** — policy space is high-dimensional but behavioral dynamics evolve along low-dimensional manifolds

## Evidence

|Evidence Type|Source|
|---|---|
|PCA 3-PCs captures ~90% variance|`pca.py`, Notebook 01 §4.6|
|UMAP reveals curved surfaces and bridges|`umap_projection.py`, Notebook 02 §4|
|t-SNE confirms local cluster structure|`tsne.py`, Notebook 02 §5|
|Trustworthiness + continuity scores > threshold|`manifold_metrics.py`|
|Silhouette improves PCA→UMAP|`manifold_metrics.py`|

## Linear vs Nonlinear Structure

- **PCA** captures the dominant linear axis (healthy vs unstable) at PC1
- **UMAP** reveals the curved intra-healthy manifold connecting stable → adaptive → exploratory
- The gap between PCA and UMAP silhouette scores quantifies the **nonlinear component**

## See Also

- [[Manifold Learning/UMAP Projection]]
- [[Manifold Learning/PCA Analysis]]
- [[Manifold Learning/tSNE Validation]]
- [[Manifold Learning/Transitional Geometry]]
- [[Theory/Dynamical Systems Interpretation]]
- [[Experiments/Hypothesis Testing]]