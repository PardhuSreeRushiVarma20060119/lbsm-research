## tags: [lbsm, trajectory-geometry, displacement] module: trajectory_geometry.py data: trajectory_stats.csv

# Displacement

## Definition

Straight-line Euclidean distance from the agent's **starting position** to its **ending position** in UMAP embedding space:

$$D_i = | x_T - x_1 |_2$$

## Behavioral Interpretation

Displacement measures **net adaptation movement** — whether the agent ended up in a meaningfully different behavioral location than where it started.

- High displacement → significant behavioral drift over the run
- Low displacement → agent returned near its start (cyclical behavior, equilibrium)

## Empirical Values

From `trajectory_stats.csv`:

|Statistic|Value|
|---|---|
|Mean|6.89|
|Std|3.71|
|Min|1.81|
|Max|13.01|

High variance relative to mean (CV ≈ 0.54) — displacement differs substantially between agents even though path lengths are nearly identical. This reflects differences in which regimes agents start/end in.

## Contrast with Path Length

Path length is nearly constant across agents (~5874 ± 71). Displacement is highly variable (1.81–13.01). This divergence is what creates the extreme tortuosity values observed.

## See Also

- [[Trajectory Geometry/Tortuosity]]
- [[Trajectory Geometry/Path Length]]
- [[Trajectory Geometry/Trajectory Overview]]