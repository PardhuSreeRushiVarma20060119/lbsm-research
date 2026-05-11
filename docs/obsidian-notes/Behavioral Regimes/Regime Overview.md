## tags: [lbsm, behavioral-regimes, overview]

# Behavioral Regime Overview

## The Four Regimes

$$\mathcal{S} = { \text{stable},\ \text{exploratory},\ \text{adaptive},\ \text{unstable} }$$

Each regime acts as a **behavioral attractor basin** — a region of latent space toward which behavioral dynamics are drawn.

## Regime Summary

|Regime|Attractor Type|Entropy|Velocity|Stability|
|---|---|---|---|---|
|[[Behavioral Regimes/Stable Regime\|Stable]]|Compact convergent|Low|Low|High|
|[[Behavioral Regimes/Exploratory Regime\|Exploratory]]|Dispersed, high-energy|High|Medium|Medium|
|[[Behavioral Regimes/Adaptive Regime\|Adaptive]]|Bridge manifold|Medium|Medium|Medium|
|[[Behavioral Regimes/Unstable Regime\|Unstable]]|Collapsed / diffuse|High|High|Low|

## Dominant Transitions (empirical)

From `transition_coords.csv` (14,161 transitions):

```
stable      → exploratory   2,201  (primary exploration trigger)
exploratory → adaptive      2,572  (convergence begins)
adaptive    → stable        2,522  (convergence completes)
```

These three form the **core adaptation loop**: stable → exploratory → adaptive → stable.

## Geometric Interpretation

- **Stable** and **Exploratory** are opposing attractor poles in UMAP space
- **Adaptive** sits geometrically _between_ them — a bridge manifold
- **Unstable** is a diffuse outlier region, geometrically separate from the healthy loop

## See Also

- [[Behavioral Regimes/Stable Regime]]
- [[Behavioral Regimes/Exploratory Regime]]
- [[Behavioral Regimes/Adaptive Regime]]
- [[Behavioral Regimes/Unstable Regime]]
- [[Behavioral Regimes/Attractor Basins]]
- [[Behavioral Regimes/Separability Theory]]
- [[Mathematics/State Space Definition]]
- [[Mathematics/Markov Transition Model]]