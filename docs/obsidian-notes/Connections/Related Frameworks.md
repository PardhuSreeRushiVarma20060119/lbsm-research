---

## tags: [lbsm, connections, related-work]
---
# Related Frameworks

## Hidden Markov Models (HMMs)

LBSM shares the Markov hidden-state assumption with HMMs. Key difference: HMMs infer regimes from observations. LBSM starts with known ground-truth regimes (simulation) and studies the _geometry_ of their telemetry projections.

## Manifold Learning

LBSM applies standard manifold learning tools (PCA, UMAP, t-SNE) but in a new context: agent behavioral telemetry with explicit temporal structure and ground-truth regime labels.

## Trajectory Analysis / Movement Ecology

Tortuosity, path length, and displacement metrics are borrowed from movement ecology (animal trajectory analysis). LBSM applies these to behavioral manifold trajectories.

## Topological Data Analysis (TDA)

Manifold metrics (trustworthiness, continuity) connect to TDA. Future work: persistent homology of behavioral manifolds could characterize topological changes across training.

## Anomaly Detection

The `is_anomaly` column in the telemetry data suggests LBSM connects to anomaly detection — unstable regime transitions or behavioral shocks could be anomaly signals.

## See Also

- [[Paper Draft/Novelty Statement]]
- [[Connections/Nonlinear Dynamics]]
- [[Connections/Entropy and Exploration]]