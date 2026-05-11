---

## tags: [lbsm, connections, entropy, exploration]
---
# Entropy and Exploration

## Entropy as Exploration Indicator

In reinforcement learning, entropy regularization encourages exploration. LBSM's entropy feature directly measures this:

- High policy entropy → agent sampling broadly from action space (exploratory regime)
- Low policy entropy → agent committed to specific actions (stable regime)
- Entropy collapse → possibly entering unstable regime (brittle over-committed policy)

## Connection to MaxEnt RL

Maximum entropy RL (e.g. SAC) explicitly maximizes entropy as an objective. In LBSM terms, MaxEnt agents would spend more time in the exploratory regime and have higher mean entropy.

## See Also

- [[Theory/Information Theoretic Interpretation]]
- [[Behavioral Regimes/Exploratory Regime]]
- [[Simulation/Telemetry Vector]]