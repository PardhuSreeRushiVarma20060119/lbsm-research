## tags: [lbsm, manifold-learning, boundaries, transitions] module: trajectory_geometry.py — transition_embedding_coords() data: transition_coords.csv

# Regime Boundaries

## Core Finding

Behavioral transitions **do not occur randomly** across the manifold. They localize near:

- Manifold boundaries
- Bridge regions
- Geometric bottlenecks

## `transition_embedding_coords()` Function

```python
transitions = transition_embedding_coords(E_umap2, df)
# Returns: DataFrame with columns:
# [agent_id, timestep, from_state, to_state, emb_x, emb_y]
```

Each row is one regime-change event, with its UMAP coordinates at the moment of transition.

## Data: `transition_coords.csv`

**14,161 total transitions** across 20 agents × 2000 timesteps.

|from_state|to_state|count|
|---|---|---|
|adaptive|stable|2,522|
|exploratory|adaptive|2,572|
|stable|exploratory|2,201|
|exploratory|unstable|885|
|unstable|exploratory|830|
|unstable|stable|807|
|stable|unstable|768|
|adaptive|exploratory|819|
|unstable|adaptive|816|
|adaptive|unstable|796|
|stable|adaptive|750|
|exploratory|stable|395|

## Geometric Observation

Plotting transition points on the UMAP manifold (Notebook 02 §7.2) reveals that:

- stable→exploratory and exploratory→adaptive transitions cluster in the **bridge region**
- unstable transitions cluster near the **outer boundary** of the healthy manifold
- transitions do not occur uniformly across embedding space

## See Also

- [[Mathematics/Markov Transition Model]]
- [[Manifold Learning/Transitional Geometry]]
- [[Manifold Learning/UMAP Projection]]
- [[Theory/Regime Boundary Theory]]
- [[Visualizations/Manifold Plots]]