## tags: [lbsm, simulation, telemetry, features] module: behavior_profiles.py data: telemetry_n20_t2000.csv

## Feature Definition

$$x_t = [\text{latency},\ \text{entropy},\ \text{reward},\ \text{memory_usage},\ \text{error_rate},\ \text{action_freq}]$$

Six observable features emitted per agent per timestep.

## Feature Semantics

|Feature|Meaning|Regime Signal|
|---|---|---|
|`latency`|Computational / reactive load|High in unstable|
|`entropy`|Policy stochasticity|High in exploratory|
|`reward`|Task performance signal|High in stable|
|`memory_usage`|Memory consumption|Varies|
|`error_rate`|Execution error frequency|High in unstable|
|`action_freq`|Actions per unit time|High in exploratory|

## Z-Scored Columns

All downstream analysis uses z-scored versions:

```
latency_z, entropy_z, reward_z, memory_usage_z, error_rate_z, action_freq_z
```

Z-scoring ensures PCA/UMAP/t-SNE are not dominated by scale differences.

## Data File

`telemetry_n20_t2000.csv` — 40,000 rows × 17 columns

Sample row:

```
agent_0000, t=0, hidden_state=stable,
latency=50.73, entropy=0.763, reward=8.635,
memory_usage=124.23, error_rate=0.015, action_freq=14.22
```

## Feature Separability

Entropy and latency are the most theoretically orthogonal features:

- **Latency** → computational/reactive load
- **Entropy** → policy stochasticity

Their joint scatter (Notebook 01 §4.3) is the **primary behavioral plane** for visual separation.

## See Also

- [[Mathematics/Emission Model]]
- [[Simulation/Agent Model]]
- [[Simulation/AR1 Smoothing]]
- [[Manifold Learning/PCA Analysis]]