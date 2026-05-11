---

## tags: [lbsm, mathematics, state-space] module: agent.py

# State Space Definition

## Hidden State Space $\mathcal{S}$

$$\mathcal{S} = { \text{stable},\ \text{exploratory},\ \text{adaptive},\ \text{unstable} }$$

Four discrete latent regimes. Agents cannot be observed directly in these states — only their telemetry emissions are visible.

## Integer Label Encoding

|Label|Regime|
|---|---|
|0|stable|
|1|exploratory|
|2|adaptive|
|3|unstable|

Used in `y_labels.npy` and throughout sklearn metric calls.

## Class Balance (N=20 agents, T=2000)

From `telemetry_n20_t2000.csv` (40,000 observations total):

```
States observed: ['stable', 'exploratory', 'adaptive', 'unstable']
```

Each agent's hidden state evolves as a **first-order Markov chain**. Observable telemetry is emitted from the corresponding regime profile via an AR(1) process.

## Source

Defined in → `agent.py`

---

## See Also

- [[Mathematics/Markov Transition Model]]
- [[Mathematics/Emission Model]]
- [[Behavioral Regimes/Regime Overview]]
- [[Simulation/Agent Model]]