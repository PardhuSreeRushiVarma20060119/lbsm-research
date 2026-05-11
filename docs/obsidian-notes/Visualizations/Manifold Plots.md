---

## tags: [lbsm, visualizations, manifold] notebook: 01_telemetry_generation.ipynb §4, 02_manifold_learning.ipynb §4–§7

---
# Manifold Plots

## Notebook 01 Visualizations

| §   | Plot                                        | Key Purpose                         |
| --- | ------------------------------------------- | ----------------------------------- |
| 4.1 | State frequency bar chart                   | Empirical time in each regime       |
| 4.2 | Feature violin + strip plots                | Per-regime distribution per feature |
| 4.3 | Latency vs Entropy scatter                  | Primary behavioral plane            |
| 4.4 | Feature separability heatmap                | Fisher ratio per feature            |
| 4.5 | Single-agent temporal trajectory            | AR(1) smoothing, regime dwell times |
| 4.6 | PCA 2-D scatter (PC1–PC2, PC1–PC3, PC2–PC3) | Linear baseline geometry            |
| 4.7 | Pair-plot (6-way feature scatter)           | Full cross-feature relationships    |

## Notebook 02 Visualizations

|§|Plot|Key Purpose|
|---|---|---|
|3|Scree plot + cumulative variance|PCA baseline variance|
|3|PCA 2-D scatter|Baseline visual|
|4.2|UMAP 2-D scatter|**Primary behavioral manifold**|
|4.3|UMAP 3-D surface|Surface curvature|
|4.4|Regime connectivity heatmap|Boundary porosity|
|5.2|t-SNE scatter + intra-regime spread|Local cluster validation|
|6.1|Radar/bar scorecard|Method comparison|
|6.2|Per-regime silhouette grouped bar|Regime-level quality|
|7|Multi-agent trajectory overlay|Behavioral paths in UMAP space|
|7.2|Transition point scatter on manifold|Boundary localization|
|7.3|Manifold speed timeline|Velocity + regime shading|

## Key Visual Design Choices

- **Color palette** consistent across all plots: one color per regime
- **Background scatter** faded (alpha 0.07–0.12) when overlaying trajectories/transitions
- **Centroids** marked with `*` markers on all embedding plots
- **30% sampling** used for UMAP scatter to avoid overplotting

## See Also

- [[Manifold Learning/UMAP Projection]]
- [[Manifold Learning/Regime Boundaries]]
- [[Trajectory Geometry/Trajectory Overview]]
- [[Trajectory Geometry/Manifold Velocity]]