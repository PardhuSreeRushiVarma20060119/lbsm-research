## tags: [lbsm, behavioral-regimes, exploratory]

# Exploratory Regime

## Definition

Entropy expansion phase. The agent is actively traversing policy-space, sampling new actions rather than exploiting known strategies.

## Attractor Properties

- **High entropy** — maximum policy stochasticity
- **High action frequency** — rapid action sampling
- **Medium velocity** — active movement in manifold space
- **Dispersed cluster** — wide spread in UMAP

## Telemetry Signature

|Feature|Expected Level|
|---|---|
|latency|Medium-high|
|entropy|High|
|reward|Medium|
|error_rate|Medium|
|action_freq|High|

## Information-Theoretic Interpretation

Entropy is the primary discriminator for exploratory behavior. High entropy = the agent's policy is maximally uncertain — it is not committed to any single action.

## Geometric Form

More dispersed than stable in UMAP space. Sits at the opposite pole from stable along the adaptation axis.

## Transition Behavior

- Exits most commonly to **adaptive** (2,572 transitions) — exploration leads to convergence
- Occasionally drops to **unstable** (885 transitions)
- Rarely jumps directly back to **stable** (395 — the rarest transition)

## See Also

- [[Behavioral Regimes/Regime Overview]]
- [[Behavioral Regimes/Adaptive Regime]]
- [[Theory/Information Theoretic Interpretation]]
- [[Connections/Entropy and Exploration]]