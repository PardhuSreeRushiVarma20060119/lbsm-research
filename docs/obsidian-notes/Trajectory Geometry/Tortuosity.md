## tags: [lbsm, trajectory-geometry, tortuosity] module: trajectory_geometry.py data: trajectory_stats.csv

# Tortuosity

## Formula

$$\text{Tortuosity} = \frac{\text{Path Length}}{\text{Displacement}}$$

Where:

- **Path Length** = total arc length (sum of step-to-step distances in embedding space)
- **Displacement** = straight-line Euclidean distance from start point to end point

A small epsilon is added to displacement to avoid division by zero: `displacement + 1e-9`

## Interpretation

|Value|Behavioral Meaning|
|---|---|
|Tortuosity = 1|Perfectly straight path — idealized directed convergence|
|Low (1–10)|Efficient, directed adaptation|
|Medium|Normal behavioral wandering|
|High (100+)|Chaotic exploration or unstable wandering|
|Very high (1000+)|Near-zero net displacement — agent returns to start|

## Empirical Values (from `trajectory_stats.csv`)

- Mean tortuosity: very high (driven by small displacement values ~6.89 with large path lengths ~5874)
- Agent range: 532 to 2054
- High tortuosity is expected given constant regime cycling with no net drift

## Note on Interpretation

The very high tortuosity values in this dataset reflect that agents cycle through regimes continuously without systematic drift from their starting point — they are in a **dynamic behavioral equilibrium**, not converging to a fixed point.

## Code

```python
tortuosity = path_length / (displacement + 1e-9)
```

In `compute_trajectory_stats()` from `trajectory_geometry.py`.

## See Also

- [[Trajectory Geometry/Trajectory Overview]]
- [[Trajectory Geometry/Path Length]]
- [[Trajectory Geometry/Displacement]]
- [[Metrics/Tortuosity Index]]
- [[Theory/Instability Theory]]