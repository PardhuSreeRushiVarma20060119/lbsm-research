
## tags: [lbsm, behavioral-regimes, adaptive]

# Adaptive Regime

## Definition

Policy integration dynamics. The agent is in transition — neither fully exploring nor fully converged. This is the **most theoretically critical** regime in LBSM.

## Geometric Role

> The adaptive regime acts as a **geometric bridge manifold**.

Adaptation is not a discrete state switch. It is a **continuous geometric traversal** between the exploratory and stable attractor basins.

In UMAP space, adaptive points occupy the region _between_ exploratory and stable — confirming the bridge hypothesis.

## Attractor Properties

- **Medium entropy** — partial commitment to policy
- **Medium velocity** — directional movement toward stable
- **Bridge geometry** — elongated manifold region, not a compact cluster

## Telemetry Signature

|Feature|Expected Level|
|---|---|
|entropy|Medium (decreasing)|
|reward|Medium (increasing)|
|latency|Medium|
|error_rate|Medium (decreasing)|

## Why It Has Low Silhouette

The adaptive regime is expected to have the **lowest per-regime silhouette coefficient** — because it genuinely overlaps with both stable and exploratory. This is not a failure; it is the correct behavior for a bridge manifold.

## Transition Behavior

- Exits to **stable** most frequently (2,522) — convergence succeeds
- Exits to **exploratory** (819) — backsliding
- Exits to **unstable** (796) — convergence failure

## See Also

- [[Behavioral Regimes/Regime Overview]]
- [[Manifold Learning/Transitional Geometry]]
- [[Theory/Geometric Interpretation of Adaptation]]
- [[Behavioral Regimes/Separability Theory]]