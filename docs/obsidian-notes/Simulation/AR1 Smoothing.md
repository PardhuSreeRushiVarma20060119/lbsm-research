## tags: [lbsm, simulation, temporal-coherence, ar1] module: behavior_profiles.py

# AR(1) Smoothing

## Purpose

Enforces **temporal coherence** in emitted telemetry. Without it, each timestep would be an independent Gaussian draw — agents would teleport behaviorally.

## Equation

$$x_t = \alpha \cdot x_{t-1} + (1 - \alpha) \cdot \tilde{x}_t$$

where $\tilde{x}_t \sim \mathcal{N}(\mu_{s_t}, \Sigma_{s_t})$ is the raw Gaussian emission for the current regime.

- $\alpha$ close to 1 → strong temporal memory, slow behavioral drift
- $\alpha$ close to 0 → fast regime response, weaker smoothing

## Effect on Geometry

AR(1) smoothing is the mechanism that creates:

- **Smooth trajectories** in UMAP/t-SNE space
- **Manifold continuity** — no discontinuous jumps between embedding coordinates
- **Meaningful behavioral motion** — velocity and tortuosity become interpretable

## Effect on Regime Transitions

Transitions are not instantaneous. When the hidden state switches from regime $i$ to regime $j$, the telemetry $x_t$ drifts toward $\mu_j$ rather than jumping there. This creates the **transition manifold bridges** visible in UMAP embeddings.

## Defined In

`behavior_profiles.py` — `BehaviorProfile.sample()` method

## See Also

- [[Mathematics/Emission Model]]
- [[Theory/Temporal Coherence]]
- [[Manifold Learning/Transitional Geometry]]
- [[Trajectory Geometry/Manifold Velocity]]