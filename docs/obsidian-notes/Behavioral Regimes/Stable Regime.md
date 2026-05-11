## tags: [lbsm, behavioral-regimes, stable]

# Stable Regime

## Definition

Converged policy equilibrium. The agent has found a reliable behavioral strategy and is executing it deterministically.

## Attractor Properties

- **Low entropy** — deterministic policy execution
- **Low velocity** — minimal movement in manifold space
- **Low manifold dispersion** — compact, dense cluster in UMAP
- **High local density** — many nearby points in embedding

## Telemetry Signature

|Feature|Expected Level|
|---|---|
|latency|Low-moderate|
|entropy|Low|
|reward|High|
|error_rate|Low|
|action_freq|Low-moderate|

## Geometric Form

Compact attractor. In UMAP space, stable forms a **tight dense cluster** — the most cohesive of all four regimes.

## Transition Behavior

- Exits most commonly to **exploratory** (2,201 transitions)
- Returns from **adaptive** (2,522 transitions)
- Rarely transitions directly to **unstable**

## See Also

- [[Behavioral Regimes/Regime Overview]]
- [[Behavioral Regimes/Attractor Basins]]
- [[Theory/Stability Theory]]
- [[Metrics/Topological Preservation]]