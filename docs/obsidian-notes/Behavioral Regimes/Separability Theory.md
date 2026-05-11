## tags: [lbsm, behavioral-regimes, separability]

# Separability Theory

## Core Claim

> Regimes are **statistically overlapping** yet **geometrically separable**.

This distinction is fundamental to LBSM.

## Why Not Perfect Separability?

Perfect separability would imply:

- Unrealistic discrete behavioral switching
- Trivial classification (any linear classifier would suffice)
- No transitional ambiguity

LBSM deliberately models **soft boundaries** because real adaptive agents blur regime boundaries continuously.

## Linear Separability (PCA Results)

From Notebook 01 + 02:

- **PC1 captures 80.3% of variance** (full dataset) — primarily the healthy/unstable axis
- **Unstable is linearly separable** from the three healthy regimes on PC1
- Three healthy regimes overlap linearly — PC1 cannot separate them
- LDA accuracy: **70.4%** — real but moderate linear separability

## Nonlinear Separability (UMAP Results)

UMAP sharply improves regime separation:

- Silhouette score improves significantly over PCA baseline
- Unstable achieves high k-NN purity (already linearly separate)
- Adaptive retains low silhouette — expected for a bridge manifold

## Per-Regime Silhouette Expectation

|Regime|Expected Silhouette|Reason|
|---|---|---|
|Stable|High|Compact, distinct attractor|
|Unstable|High|Geometrically isolated|
|Exploratory|Medium–High|Dispersed but bounded|
|Adaptive|Low|Bridge manifold, intentional overlap|

## See Also

- [[Manifold Learning/PCA Analysis]]
- [[Manifold Learning/UMAP Projection]]
- [[Metrics/Topological Preservation]]
- [[Behavioral Regimes/Adaptive Regime]]
- [[Results/Regime Separability]]