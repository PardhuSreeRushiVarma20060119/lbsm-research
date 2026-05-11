## tags: [lbsm, mathematics, markov]

# Markov Transition Model

## Transition Equation

$$P(s_{t+1} = j \mid s_t = i)$$

The hidden behavioral regime evolves according to a **stochastic Markov transition process**.

## Key Properties

- Future regime depends **only on the current regime** — not the full history
- This is the Markov (memoryless) assumption
- Transition matrix $\mathbf{A}$ has entries $A_{ij} = P(s_{t+1}=j \mid s_t=i)$
- Each row sums to 1

## Empirical Transition Counts (from `transition_coords.csv`)

14,161 total transitions observed across 20 agents × 2000 timesteps.

|From → To|Count|
|---|---|
|adaptive → stable|2,522|
|exploratory → adaptive|2,572|
|stable → exploratory|2,201|
|exploratory → unstable|885|
|unstable → exploratory|830|
|stable → unstable|768|
|unstable → stable|807|
|adaptive → exploratory|819|
|adaptive → unstable|796|
|unstable → adaptive|816|
|stable → adaptive|750|
|exploratory → stable|395|

## Observations

- **stable → exploratory** is the dominant healthy transition (2,201)
- **exploratory → adaptive** and **adaptive → stable** form the core adaptation loop
- **unstable** transitions are roughly symmetric — it enters and exits from all regimes
- **exploratory → stable** is the rarest transition (395) — agents rarely jump directly from exploration to stability

## Stationary Distribution

Derived from the transition matrix via eigenvector method:

$$\pi \mathbf{A} = \pi, \quad \sum_i \pi_i = 1$$

Verified against empirical state frequencies in Notebook 01.

## See Also

- [[Mathematics/State Space Definition]]
- [[Mathematics/Emission Model]]
- [[Behavioral Regimes/Regime Overview]]
- [[Experiments/Hypothesis Testing]]
- [[Results/Regime Separability]]
