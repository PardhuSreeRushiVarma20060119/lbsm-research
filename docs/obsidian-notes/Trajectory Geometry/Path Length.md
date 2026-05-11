## tags: [lbsm, trajectory-geometry, path-length] module: trajectory_geometry.py data: trajectory_stats.csv

# Path Length

## Definition

Total arc length of an agent's trajectory in UMAP embedding space:

$$L_i = \sum_{t=1}^{T-1} | x_{t+1} - x_t |_2$$

The sum of all step-to-step Euclidean distances over the agent's lifetime.

## Behavioral Interpretation

Path length measures **total behavioral travel distance** — how much ground the agent covered in latent space, regardless of direction.

- Long path + small displacement → high tortuosity (behavioral wandering)
- Long path + large displacement → efficient directional adaptation

## Empirical Values

From `trajectory_stats.csv` (20 agents, T=2000):

|Statistic|Value|
|---|---|
|Mean|5874.1|
|Std|71.0|
|Min|5708.5|
|Max|5982.7|

Very low variance across agents — all agents travel similar total distances. The **mean_speed** (~2.97 per step) × 1999 steps ≈ 5935, consistent.

## Regime Arc Statistics

`regime_arc_statistics()` computes mean path length _within_ each regime, allowing comparison:

> Does the unstable regime produce faster-moving (longer arc per time) trajectories?

## See Also

- [[Trajectory Geometry/Trajectory Overview]]
- [[Trajectory Geometry/Tortuosity]]
- [[Trajectory Geometry/Displacement]]
- [[Trajectory Geometry/Manifold Velocity]]