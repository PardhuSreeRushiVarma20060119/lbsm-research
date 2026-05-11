## tags: [lbsm, mathematics, emission, gaussian] module: behavior_profiles.py

# Emission Model

## Emission Equation

$$x_t \sim \mathcal{N}(\mu_{s_t},\ \Sigma_{s_t})$$

Each hidden regime $s_t$ has its own **Gaussian emission distribution**. Observed telemetry $x_t$ is a noisy draw from that regime's distribution.

## Telemetry Feature Vector

$$x_t = [\text{latency},\ \text{entropy},\ \text{reward},\ \text{memory_usage},\ \text{error_rate},\ \text{action_freq}]$$

Six features. Z-scored versions (`latency_z`, `entropy_z`, …) used for all downstream analysis.

## Why Gaussian?

Gaussian emissions provide:

- Continuous behavioral variation
- Realistic noise
- **Probabilistic regime overlap** — critical for soft boundary modeling
- Smooth manifold formation

> Without overlap, regimes become trivial clusters and geometry loses realism.

## AR(1) Temporal Smoothing

Raw Gaussian samples are temporally smoothed via an AR(1) process (defined in `behavior_profiles.py`):

$$x_t = \alpha \cdot x_{t-1} + (1 - \alpha) \cdot \tilde{x}_t, \quad \tilde{x}_t \sim \mathcal{N}(\mu_{s_t}, \Sigma_{s_t})$$

This enforces **temporal coherence** — agents do not teleport behaviorally between timesteps.

## Regime Statistical Signatures

Each regime has distinct $(\mu, \Sigma)$ parameters. Empirical vs theoretical means are compared in Notebook 01 §3.1 to validate the emission model.

## Feature Separability (Notebook 01 §3.3)

Fisher separability ratios computed per feature. Most discriminative features drive PC1.

## See Also

- [[Mathematics/State Space Definition]]
- [[Mathematics/Markov Transition Model]]
- [[Simulation/AR1 Smoothing]]
- [[Simulation/Telemetry Vector]]
- [[Theory/Emission Theory]]
- [[Behavioral Regimes/Separability Theory]]