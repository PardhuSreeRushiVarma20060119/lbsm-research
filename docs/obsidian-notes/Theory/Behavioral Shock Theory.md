---

## tags: [lbsm, theory, behavioral-shock] module: trajectory_geometry.py — manifold_velocity()
---
# Behavioral Shock Theory

## Definition

Large manifold velocity spikes indicate **behavioral shocks**:

- Instability events
- Abrupt policy changes
- Reward collapse
- Attractor escape

## Detection

```python
vel_df = manifold_velocity(trajectories, df, window=5)
# Shocks: timesteps where speed >> mean_speed
```

Max speed / mean speed ratio ≈ 5× across all agents — every agent experiences at least one shock.

## See Also

- [[Trajectory Geometry/Manifold Velocity]]
- [[Behavioral Regimes/Unstable Regime]]
- [[Theory/Instability Theory]]