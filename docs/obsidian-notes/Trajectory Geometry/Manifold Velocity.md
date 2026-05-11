## tags: [lbsm, trajectory-geometry, velocity, shock] module: trajectory_geometry.py — manifold_velocity() notebook: 02_manifold_learning.ipynb §7.3

# Manifold Velocity

## Definition

Instantaneous speed of an agent's trajectory in UMAP embedding space:

$$v_t = | x_{t+1} - x_t |_2$$

Step-to-step Euclidean displacement. A rolling mean is applied to smooth AR(1) noise.

## `manifold_velocity()` Function

```python
vel_df = manifold_velocity(trajectories, df, window=5)
# Returns DataFrame: [agent_id, timestep, speed, speed_smooth, hidden_state]
```

Rolling mean window = 5 (center=True, min_periods=1).

## Behavioral Interpretation

|Speed|Behavioral Meaning|
|---|---|
|Low|Stable, converged behavior — minimal manifold movement|
|Medium|Normal regime traversal|
|High spike|Behavioral shock — instability, regime collapse, abrupt policy change|

## Behavioral Shock Detection

Large manifold velocity spikes indicate:

- Entry into the **unstable** regime
- Abrupt policy changes
- Reward collapse
- Attractor escape

From `trajectory_stats.csv`:

- Mean max_speed across agents: **15.04** (σ = 0.23)
- Mean mean_speed: **2.97**

The ratio max/mean ≈ 5× indicates all agents experience at least one velocity spike ~5× their average speed.

## Per-Regime Speed

`regime_arc_statistics()` computes mean speed _within_ each regime. Prediction:

```
unstable > exploratory > adaptive > stable
```

This ordering, if confirmed, directly validates H4 (instability → increased velocity).

## Visualization (Notebook 02 §7.3)

Single-agent timeline: top panel shows regime shading (hidden state), bottom panel shows speed and speed_smooth overlaid. Spikes align with unstable regime entry.

## See Also

- [[Trajectory Geometry/Trajectory Overview]]
- [[Theory/Behavioral Shock Theory]]
- [[Behavioral Regimes/Unstable Regime]]
- [[Experiments/Hypothesis Testing]]