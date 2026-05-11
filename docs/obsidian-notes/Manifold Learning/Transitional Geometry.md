## tags: [lbsm, manifold-learning, transitional-geometry, bridge]

# Transitional Geometry

## Core Claim

> The adaptive regime functions as a **geometric bridge manifold**.

Adaptation is not discrete state-switching. It is **continuous geometric traversal** between the stable and exploratory attractor basins.

## Evidence

- In UMAP 2-D space: adaptive points occupy the region _between_ stable and exploratory clusters
- In UMAP 3-D surface: the bridge is visible as an elongated curved surface
- Transition point coordinates (`transition_coords.csv`) cluster in the bridge region — not randomly distributed

## Implication for Theory

This is one of the strongest conceptual aspects of LBSM:

- Traditional models: adaptation = discrete state change
- LBSM: adaptation = smooth manifold traversal (measurable, continuous, geometric)

This allows adaptation to be quantified via path metrics (length, velocity, tortuosity) rather than just transition counts.

## AR(1) Role

The AR(1) smoothing in `behavior_profiles.py` is the mechanism that _creates_ the bridge. Without temporal smoothing, there would be no bridge region — only hard jumps between regime clusters.

## See Also

- [[Behavioral Regimes/Adaptive Regime]]
- [[Manifold Learning/UMAP Projection]]
- [[Manifold Learning/Regime Boundaries]]
- [[Simulation/AR1 Smoothing]]
- [[Theory/Geometric Interpretation of Adaptation]]