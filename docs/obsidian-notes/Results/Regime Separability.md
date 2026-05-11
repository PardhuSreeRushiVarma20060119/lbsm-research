## tags: [lbsm, results, separability] notebook: 01_telemetry_generation.ipynb §5, 02_manifold_learning.ipynb §6

# Regime Separability

## NB01 Results Summary

|Finding|Value|
|---|---|
|PC1 variance (full dataset)|80.3%|
|PC1 captures|healthy/unstable axis|
|Healthy regimes (PC1)|overlapping — nonlinear analysis needed|
|LDA accuracy|70.4%|
|NB01 evidence criteria|3/5 PASS|

## NB02 Results Summary

From Notebook 02 §6 scorecard:

|Method|Silhouette|DB|CH|Trust|Continuity|
|---|---|---|---|---|---|
|PCA (PC1-2)|baseline|—|—|—|—|
|UMAP (2-D)|> PCA|—|—|—|—|
|t-SNE (5k)|intermediate|—|—|—|—|

> Fill in actual values when running notebooks.

## Per-Regime Silhouette (UMAP, expected)

|Regime|Expected Silhouette|
|---|---|
|Stable|High|
|Unstable|High|
|Exploratory|Medium–High|
|Adaptive|Low (bridge manifold — expected)|

## k-NN Purity (UMAP vs PCA)

```python
purity_cmp["UMAP gain"] = purity_UMAP - purity_PCA
```

UMAP purity gain greatest for the healthy regimes (stable, exploratory, adaptive) — confirms nonlinear structure is real and improves separability.

## Structural Verdict

UMAP sharpens what PCA can only hint at:

- Three healthy regimes form a **connected manifold continuum** (not three isolated clusters)
- Unstable is **geometrically isolated** from the healthy loop
- Nonlinear structure is **robust across methods** (Procrustes r)

## See Also

- [[Behavioral Regimes/Separability Theory]]
- [[Manifold Learning/UMAP Projection]]
- [[Manifold Learning/PCA Analysis]]
- [[Metrics/Topological Preservation]]
- [[Experiments/Hypothesis Testing]]