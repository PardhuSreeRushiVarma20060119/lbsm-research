## tags: [lbsm, behavioral-regimes, attractors]

# Attractor Basins

## Core Concept

Each behavioral regime acts as a **behavioral attractor basin** — a region of latent space toward which the agent's dynamics are drawn when in that regime.

|Regime|Basin Type|
|---|---|
|Stable|Deep, narrow well — strong convergent attractor|
|Exploratory|Wide, shallow basin — weak attractor, high diffusion|
|Adaptive|Saddle / ridge — transitional manifold|
|Unstable|Collapsed basin — no coherent attractor|

## Dynamical Systems Framing

From a dynamical systems perspective:

- **Stable** = fixed point attractor
- **Exploratory** = limit cycle / diffuse attractor
- **Adaptive** = slow manifold / heteroclinic connection
- **Unstable** = repeller / chaotic region

## Manifestation in UMAP

The attractor basin structure is directly visible in UMAP embeddings:

- Stable → compact dense cluster
- Exploratory → dispersed but bounded region
- Adaptive → elongated bridge between stable and exploratory
- Unstable → geometrically separated diffuse cloud

## See Also

- [[Behavioral Regimes/Regime Overview]]
- [[Theory/Dynamical Systems Interpretation]]
- [[Manifold Learning/Latent Geometry Hypothesis]]
- [[Connections/Nonlinear Dynamics]]