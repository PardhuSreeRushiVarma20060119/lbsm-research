---

## tags: [lbsm, theory, temporal-coherence] module: behavior_profiles.py

---

# Temporal Coherence

## Core Principle

> Agents do not teleport behaviorally.

Behavior evolves continuously. This creates smooth trajectories, manifold continuity, and meaningful behavioral motion.

## Mechanism

AR(1) smoothing in `behavior_profiles.py` enforces temporal correlation between successive telemetry observations. See [[Simulation/AR1 Smoothing]].

## Consequences for Geometry

- Trajectories are smooth curves (not random point clouds)
- Manifold is continuous (no disconnected islands)
- Velocity and tortuosity are interpretable (not artifacts of noise)
- The adaptive bridge manifold exists (AR(1) creates the transition zone)

## See Also

- [[Simulation/AR1 Smoothing]]
- [[Theory/Emission Theory]]
- [[Manifold Learning/Transitional Geometry]]