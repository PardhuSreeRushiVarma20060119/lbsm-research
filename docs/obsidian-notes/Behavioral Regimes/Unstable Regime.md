## tags: [lbsm, behavioral-regimes, unstable]

# Unstable Regime

## Definition

Attractor collapse. The agent's policy has destabilized — chaotic behavioral dispersion with no coherent strategy.

## Attractor Properties

- **High entropy** — maximum behavioral uncertainty
- **High variance** — wide telemetry fluctuations
- **High velocity** — rapid, erratic movement in manifold space
- **High tortuosity** — inefficient, wandering trajectories
- **Weak attractor structure** — diffuse, non-compact region in UMAP

## Telemetry Signature

|Feature|Expected Level|
|---|---|
|latency|High|
|entropy|High|
|reward|Low|
|error_rate|High|
|action_freq|Erratic|

## Geometric Form

Unlike the healthy regime loop (stable → exploratory → adaptive → stable), unstable is a **diffuse outlier region** in UMAP space. It has high k-NN purity because it is well-separated from healthy regimes — but this separation indicates pathological behavior, not healthy clustering.

## PC1 Dominance

PCA analysis from Notebook 01 shows that **PC1 (80.3% of variance on full dataset)** primarily captures the healthy/unstable axis. The unstable regime is **linearly separable** from the three healthy regimes, while healthy-regime separation requires nonlinear methods.

## Behavioral Shock Link

Large manifold velocity spikes (detected by `manifold_velocity()`) correspond to **entry into or exit from** the unstable regime — these are the behavioral shocks.

## Transition Behavior

Roughly symmetric — unstable transitions to exploratory, stable, and adaptive at approximately equal rates (~800–830 each), confirming no preferred exit path.

## See Also

- [[Behavioral Regimes/Regime Overview]]
- [[Theory/Instability Theory]]
- [[Theory/Behavioral Shock Theory]]
- [[Trajectory Geometry/Manifold Velocity]]
- [[Manifold Learning/PCA Analysis]]