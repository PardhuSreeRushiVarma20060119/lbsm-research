## tags: [lbsm, metrics, tortuosity] data: trajectory_stats.csv

# Tortuosity Index

## Definition

$$\text{Tortuosity} = \frac{\text{Path Length}}{\text{Displacement} + \epsilon}$$

$\epsilon = 10^{-9}$ to prevent division by zero.

## Empirical Range (N=20 agents)

From `trajectory_stats.csv`:

|Agent|Tortuosity|Displacement|Path Length|
|---|---|---|---|
|Lowest|~532|10.98|5840|
|Median|~871|7.35|5871|
|Highest|~2054|2.88|5916|

## Use in Hypothesis Testing

Directly tests **H4**: Instability manifests as increased manifold velocity and tortuosity.

To test: compare tortuosity computed _within_ each regime segment vs across the full trajectory. Unstable regime segments should have the highest within-segment tortuosity.

## Relationship to Behavioral Efficiency

- Tortuosity = 1 → ideal directed convergence (never observed in practice)
- Tortuosity = 500–2000 → typical LBSM range for agents cycling between regimes
- Very high tortuosity with low displacement → agent in dynamic equilibrium (not converging)

## See Also

- [[Trajectory Geometry/Tortuosity]]
- [[Trajectory Geometry/Path Length]]
- [[Trajectory Geometry/Displacement]]
- [[Experiments/Hypothesis Testing]]