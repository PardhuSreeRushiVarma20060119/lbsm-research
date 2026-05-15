
# **SECTION A — Markov Dynamics**
#### *Problem A1 : Stationary Distribution*

*Given transition matrix:*
$$ T = \begin{bmatrix} 0.80 & 0.15 & 0.03 & 0.02 \\ 0.20 & 0.50 & 0.20 & 0.10 \\ 0.10 & 0.20 & 0.60 & 0.10 \\ 0.25 & 0.15 & 0.10 & 0.50 \end{bmatrix} $$
*Find stationary distribution:*
$$\pi T = \pi$$ *subject to:* $$\sum_i \pi_i = 1$$
-------------------------------------------------------------------------------------------------------------

#### Problem A2 — Expected Regime Occupancy

*Using stationary distribution $\pi$ from A1, derive the expected percentage
of time spent in each regime:*

- *Stable*
- *Adaptive*
- *Exploratory*
- *Unstable*

*Interpret geometrically.*

------------------------------------------------------------------------

#### Problem A3 — Multi-Step Transition Probability

*Compute:*

$$T^2, \quad T^3, \quad T^{10}$$

*Interpret:*

- *Long-term mixing*
- *Regime persistence*
- *Transition smoothing*

------------------------------------------------------------------------

# **Section B — AR(1) Telemetry Mathematics**

#### Problem B1 — Mean Derivation

Given:

$$x_t = (1-\rho)\mu + \rho x_{t-1} + \epsilon_t$$

Derive:

$$E[x_t] = \mu$$

under stationarity.

---

#### Problem B2 — Variance Derivation

Derive:

$$\text{Var}(x_t) = \frac{\sigma_\epsilon^2}{1 - \rho^2}$$

Interpret:

- Smooth behavior
- Volatility
- Trajectory coherence

---

#### Problem B3 — Autocorrelation

Show:

$$\text{Corr}(x_t, x_{t+k}) = \rho^k$$

Interpret:

- Behavioral memory
- Persistence
- Temporal smoothness

-----------------

# **Section C — PCA Mathematics**

#### Problem C1 — Derive PCA Optimization

Show that PCA solves:

$$\max_{\|v\|=1} \text{Var}(Xv)$$

and derive:

$$\Sigma v = \lambda v$$

> This is one of the **most important** derivations.

---

#### Problem C2 — Explained Variance

Given eigenvalues:

$$\lambda = [4.82,\ 0.31,\ 0.28,\ 0.23,\ 0.20,\ 0.16]$$

Compute explained variance ratios:

$$\frac{\lambda_i}{\sum_j \lambda_j}$$

Verify:

$$\text{PC1} \approx 80\%$$

---

#### Problem C3 — Why Does PC1 Dominate?

Using covariance intuition, explain mathematically why the
unstable regime drives most variance.

---

#### Problem C4 — Low-Dimensionality Proof

Given:

$$\text{PC1} + \text{PC2} + \text{PC3} \approx 90\%$$

argue mathematically that the behavior manifold is
intrinsically low-dimensional.

---

# **Section D — Fisher & LDA**

#### Problem D1 — Fisher Ratio

Derive:

$$F = \frac{(\mu_1 - \mu_2)^2}{\sigma_1^2 + \sigma_2^2}$$

Compute for:

- Latency
- Entropy
- Error rate

Compare discriminative strength.

---

#### Problem D2 — LDA Derivation

Derive:

$$w \propto S_W^{-1}(\mu_1 - \mu_2)$$

Explain:

- Why LDA separates linearly
- Why overlap still exists

---

# **Section E — Geometry**

#### Problem E1 — Regime Centroid Distances

Given centroid vectors, compute pairwise Euclidean distances:

$$d(c_i, c_j) = \|c_i - c_j\|_2 = \sqrt{\sum_k (c_{ik} - c_{jk})^2}$$

Interpret: unstable regime isolation.

---

#### Problem E2 — Why Healthy Regimes Overlap

Use covariance ellipsoid reasoning to explain:

$$\text{stable} / \text{adaptive} / \text{exploratory continuity}$$

---

#### Problem E3 — Covariance Ellipsoids

Show how:

- Covariance eigenvectors
- Eigenvalues

define geometric ellipsoids in feature space:

$$\mathcal{E} = \{ x \mid (x - \mu)^\top \Sigma^{-1} (x - \mu) \leq 1 \}$$

---

# **Section F — Trajectory Geometry**

#### Problem F1 — Path Length

Derive:

$$L = \sum_t \|x_{t+1} - x_t\|$$

Interpret:

- Exploration
- Instability
- Behavioral motion

---

#### Problem F2 — Tortuosity

Derive:

$$\tau = \frac{L}{\|x_T - x_0\|}$$

Interpret:

- Erratic behavior
- Smooth adaptation
- Looping trajectories

---

# **Section G — Advanced / Research-Level**

#### Problem G1 — Why Subsampling Barely Changed PCA

> This is **GOLD**.

Analyze:

- Covariance perturbation
- Eigenvalue stability
- Sampling robustness

Explain mathematically why:

| | Full | Sample |
|---|------|--------|
| PC1 Variance | 0.803 | 0.788 |

is still very stable.

---

#### Problem G2 — Spectral Stability

Suppose:

$$\Sigma' = \Sigma + E$$

Analyze how eigenvalues shift under perturbation.

By **Weyl's Theorem**:

$$|\lambda_i(\Sigma') - \lambda_i(\Sigma)| \leq \|E\|_2$$

This enters: matrix perturbation theory.

---

#### Problem G3 — Behavioral Manifold Hypothesis

Formalize mathematically that healthy regimes form a
connected manifold $\mathcal{M} \subset \mathbb{R}^d$.

Possible directions:

- Continuity
- Overlap
- Neighborhood graphs
- Transition topology

---
---
---

[[Notebook01-Solutions]]
