## tags: [lbsm, simulation, agent] module: agent.py notebook: 01_telemetry_generation.ipynb

# Agent Model

## Configuration

```python
N_AGENTS    = 20
N_TIMESTEPS = 2_000
SEED        = 42
```

Total observations: **40,000** (20 agents × 2,000 timesteps)

## TelemetryGenerator

```python
gen = TelemetryGenerator(
    n_agents    = N_AGENTS,
    n_timesteps = N_TIMESTEPS,
    seed        = SEED,
    verbose     = False,
)
df = gen.run()
```

## Dataset Shape

```
Dataset shape  : (40000, 17)
Columns        : agent_id, timestep, hidden_state, latency, entropy, reward,
                 memory_usage, error_rate, action_freq, state_label, is_anomaly,
                 latency_z, entropy_z, reward_z, memory_usage_z, error_rate_z, action_freq_z
```

## Agent Heterogeneity

20 agents with **heterogeneous** behavioral profiles. Each agent has its own transition matrix and draws from shared regime emission parameters.

## Per-Agent Summary

Available via `gen.per_agent_statistics()` — breakdown of time spent per regime, per-agent transition counts, etc.

## Stationary Distribution

```python
agent0 = gen._agents[0]
pi_theory = agent0.stationary_distribution()
```

Verified against empirical state frequencies.

## Exports

```python
gen.save("data/telemetry_n20_t2000.csv", include_z_scores=True)
np.save("data/X_telemetry.npy", gen.feature_matrix(z_scored=True))
np.save("data/y_labels.npy", gen.labels())
```

## See Also

- [[Mathematics/State Space Definition]]
- [[Simulation/Telemetry Vector]]
- [[Simulation/AR1 Smoothing]]
- [[Experiments/Hypothesis Testing]]