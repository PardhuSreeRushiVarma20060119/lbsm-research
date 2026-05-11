---

## tags: [lbsm, experiments, hypotheses] notebook: 02_manifold_learning.ipynb §9

# Hypothesis Testing

## Five Core Hypotheses

### H1 — Regime Manifold Geometry

> Behavioral regimes produce identifiable manifold geometry.

**Test:** UMAP silhouette coefficient significantly exceeds PCA baseline.

**NB02 Criterion C1:**

```python
criterion1 = sil_umap_v > sil_pca_v
# UMAP: X.XXXX   PCA: X.XXXX   Delta: +X.XXXX
```

---

### H2 — Adaptive Bridge Manifold

> Adaptive behavior forms transitional latent bridges.

**Test:** Adaptive regime points occupy the region between stable and exploratory in UMAP space. Boundary porosity (regime connectivity) is highest between stable–adaptive and exploratory–adaptive.

**Evidence:** `regime_connectivity(E_umap2, y, PROFILE_NAMES, k=20)` — off-diagonal entries for adaptive row/column.

---

### H3 — Trajectory > Static Clustering

> Trajectory geometry contains richer information than static clustering.

**Test:** Trajectory-based metrics (path length, tortuosity, velocity) explain variance in behavioral outcomes that static cluster assignments cannot.

**Evidence:** `compute_trajectory_stats()` + `regime_arc_statistics()` — differential speed per regime.

---

### H4 — Instability → Velocity + Tortuosity

> Instability manifests as increased manifold velocity and tortuosity.

**Test:** Manifold speed is significantly higher during unstable regime segments vs stable.

**Evidence:** `manifold_velocity()` per hidden_state + `regime_arc_statistics()` mean_speed comparison.

**Prediction:** `mean_speed(unstable) > mean_speed(exploratory) > mean_speed(adaptive) > mean_speed(stable)`

---

### H5 — Boundary Transition Localization

> Behavioral transitions localize near geometric boundaries.

**Test:** KDE of transition coordinates (from `transition_coords.csv`) peaks in the bridge region, not uniformly distributed across embedding space.

**Evidence:** `transition_embedding_coords()` visualization (Notebook 02 §7.2).

---

## NB01 Evidence Checklist (5 Criteria)

```
[PASS/FAIL] C1. PCA (3 PCs) captures >60% variance
[PASS/FAIL] C2. Best feature separability ratio > 5.0
[PASS/FAIL] C3. LDA cross-val accuracy > 60%
[PASS/FAIL] C4. ...
[PASS/FAIL] C5. ...
```

NB01 verdict: **3/5 criteria** → motivates nonlinear analysis in NB02.

## NB02 Evidence Checklist

Extended with nonlinear-specific criteria (Notebook 02 §9).

## See Also

- [[Results/Regime Separability]]
- [[Metrics/Topological Preservation]]
- [[Manifold Learning/Regime Boundaries]]
- [[Trajectory Geometry/Manifold Velocity]]
