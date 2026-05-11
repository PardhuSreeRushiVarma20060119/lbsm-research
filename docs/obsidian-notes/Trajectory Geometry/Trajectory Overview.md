## tags: [lbsm, trajectory-geometry, overview] module: trajectory_geometry.py notebook: 02_manifold_learning.ipynb §7 data: trajectory_stats.csv

# Trajectory Overview

## What Makes LBSM Distinct

> Traditional clustering analyzes **locations**. LBSM analyzes **motion**.

Because telemetry is temporal, the UMAP embedding is not merely a scatter of points — it is a family of **continuous curves**, one per agent, tracing behavioral trajectories through latent space.

## Formal Definition

Each agent $i$ generates:

$$\tau_i = (x_1, x_2, x_3, \ldots, x_T)$$

→ a continuous path through latent manifold space.

## `extract_agent_trajectories()`

```python
trajectories = extract_agent_trajectories(E_umap2, df)
# Returns: dict[agent_id → np.ndarray shape (T, 2)]
# Sorted by timestep for each agent
```

## Dataset: `trajectory_stats.csv`

20 agents, columns: `path_length, displacement, tortuosity, mean_speed, max_speed, n_transitions`

Summary statistics:

|Metric|Mean|Std|Min|Max|
|---|---|---|---|---|
|path_length|5874.1|71.0|5708.5|5982.7|
|displacement|6.89|3.71|1.81|13.01|
|tortuosity|~800|—|—|~2054|
|mean_speed|2.972|0.024|—|—|
|max_speed|15.04|0.23|14.63|15.62|
|n_transitions|708.1|22.9|668|770|

## Key Observations

- **Path lengths** are tightly clustered (~5708–5983) — similar total behavioral travel across agents
- **Displacement** varies widely (1.81–13.01) — net start→end movement differs greatly
- **Tortuosity** varies enormously (reflecting displacement variance with similar path length)
- **n_transitions** (~708 per agent on average) — agents transition regime ~every 2.8 timesteps on average

## See Also

- [[Trajectory Geometry/Path Length]]
- [[Trajectory Geometry/Tortuosity]]
- [[Trajectory Geometry/Manifold Velocity]]
- [[Trajectory Geometry/Displacement]]
- [[Manifold Learning/UMAP Projection]]
- [[Visualizations/Manifold Plots]]