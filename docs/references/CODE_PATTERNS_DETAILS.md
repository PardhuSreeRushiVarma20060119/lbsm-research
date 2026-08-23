# LBSM Code Patterns & Implementation Details

## Key Design Patterns

### 1. Result Container Pattern (Dataclasses)
The codebase uses frozen dataclasses to encapsulate complex results, now spanning every analysis branch:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class PCAResult:
    embedding: np.ndarray
    explained_var: np.ndarray
    cumulative_var: np.ndarray
    loadings: pd.DataFrame
    pca_model: PCA
    n_components_90: int

@dataclass
class HMMResult:                # src/hmm/hidden_state_model.py
    model: object
    pred_raw: np.ndarray
    pred_aligned: np.ndarray     # after Hungarian label alignment
    posteriors_all: np.ndarray
    mapping: dict                # {learned_label: gt_label}
    ari: float
    accuracy: float
    log_likelihood: float
    confusion: np.ndarray
    convergence_ll: list
    n_iter_actual: int

@dataclass
class MahalanobisResult:        # src/drift/drift_detection.py
    scores: np.ndarray
    flags: np.ndarray
    threshold: float
```

**Rationale**:
- Encapsulates related results together; self-documenting attribute lists
- `frozen=True` where the result should never mutate after fitting (PCA/UMAP/t-SNE); plain `@dataclass`
  where a training loop's result legitimately gets appended to over time (HMM/EWMA/KL results are built once
  and returned, so either works — the codebase isn't fully consistent about `frozen` and that's fine)

---

### 2. Factory Function Pattern
Convenience wrappers for object creation, now extended into RL (`make_env_pool`, `train_agent_pool`):

```python
def make_agent_pool(n_agents, initial_states=None, base_seed=42):
    if initial_states is None:
        initial_states = list(PROFILE_NAMES)
    return [
        AdaptiveAgent(agent_id=f"agent_{i:04d}", initial_state=initial_states[i % len(initial_states)], rng_seed=base_seed + i)
        for i in range(n_agents)
    ]

def make_env_pool(n_envs, base_seed=42, delta=..., n_steps=...):
    """src/rl/environment.py — same base_seed+offset idiom, applied to BehavioralEnv."""

def train_agent_pool(envs, config, healthy_envelope, verbose=False):
    """src/rl/q_learning.py — one QLearningAgent per env, same fan-out shape as make_agent_pool."""
```

**Rationale**: simplifies common creation scenarios; `base_seed + offset` keeps agents/envs reproducible
but non-identical. The same shape (`n_x -> List[X]`, deterministic per-item seed) now appears in three
different modules (`simulation`, `rl.environment`, `rl.q_learning`) — recognize it as one idiom, not three.

---

### 3. Validation Pattern
Explicit validation with informative error messages — unchanged from the original simulation code:

```python
@staticmethod
def _validate_transition_matrix(T: np.ndarray) -> None:
    k = len(PROFILE_NAMES)
    if T.shape != (k, k):
        raise ValueError(f"Transition matrix must be ({k},{k}), got {T.shape}.")
    row_sums = T.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(f"All rows must sum to 1. Got row sums: {row_sums}.")
    if (T < 0).any():
        raise ValueError("Transition probabilities must be non-negative.")
```

---

### 4. AR-1 Temporal Correlation Pattern
```python
def sample(self, n=1, rng=None, prev=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, 1.0, size=(n, N_FEATURES))
    samples = np.empty((n, N_FEATURES))
    x_prev = prev if prev is not None else self.means.copy()
    for t in range(n):
        x_t = self.autocorr * x_prev + np.sqrt(1.0 - self.autocorr**2) * (self.means + self.stds * noise[t])
        samples[t] = x_t
        x_prev = x_t
    return samples
```
ρ ≈ 0.3; normalization by √(1-ρ²) keeps variance stationary.

---

### 5. Hidden State Management Pattern
```python
def _transition(self) -> None:
    probs = self._T[self._state_idx]
    self._state_idx = int(self._rng.choice(len(PROFILE_NAMES), p=probs))

def step(self, timestep: int) -> dict:
    telemetry_vec = self._emit()
    record = {"agent_id": self.agent_id, "timestep": timestep, "hidden_state": self.current_state}
    record.update(zip(TELEMETRY_FEATURES, telemetry_vec.astype(float)))
    self._history.append(record)
    self._transition()
    return record
```

---

### 6. Index Mapping Pattern
```python
_STATE_INDEX = {name: i for i, name in enumerate(PROFILE_NAMES)}
_INDEX_STATE = {i: name for i, name in enumerate(PROFILE_NAMES)}
```
This same bidirectional-mapping idea reappears in `src/rl/environment.py` as `obs_to_grid()` /
`grid_to_coords()` — a discrete `(latency_bin, entropy_bin)` grid position ↔ flat state index, used by
`BehavioralEnv` exactly the way `_STATE_INDEX` is used by `AdaptiveAgent`.

---

### 7. Hungarian-Alignment Pattern (new — `src/hmm`)
Unsupervised clustering/HMM state labels are arbitrary permutations of the true regime indices; before
computing accuracy you must find the best label permutation:

```python
from scipy.optimize import linear_sum_assignment

def align_labels(pred, y_gt, n_states):
    confusion = np.zeros((n_states, n_states))
    for p, g in zip(pred, y_gt):
        confusion[p, g] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)   # maximize overlap
    mapping = dict(zip(row_ind, col_ind))
    return np.array([mapping[p] for p in pred]), mapping
```
Used by `HMMResult.mapping` / `pred_aligned` (`hidden_state_model.fit_hmm`) and again independently per-agent
in `latent_state_metrics.per_agent_metrics` — every agent's HMM decode can settle on a *different* label
permutation, so alignment has to happen per-agent, not once globally.

---

### 8. Sweep-Function Pattern (new — pervasive across hmm/drift/manifold/rl)
The dominant "model selection" idiom in this codebase is a function that loops over a hyperparameter grid
and returns a scored `DataFrame`, rather than an optimizer object:

```python
def model_selection_sweep(X_concat, lengths, n_comp_grid, ...):
    """src/hmm/sequence_inference.py — BIC/AIC per candidate n_components."""

def alpha_sweep(X, y_gt, alpha_grid):
    """src/drift/ewma.py — AUC per candidate smoothing factor."""

def window_size_sweep(X, y_gt, mu_ref, var_ref, window_grid):
    """src/drift/kl_divergence.py — AUC per candidate window size."""

def threshold_sweep(scores, y_gt, n_pts):
    """src/drift/drift_detection.py — precision/recall/F1/FPR per threshold."""

def hyperparameter_sweep(X, labels, n_neighbors_grid, min_dist_grid, ...):
    """src/manifold/umap_projection.py — silhouette per (n_neighbors, min_dist)."""
```
If you're adding a new hyperparameter search anywhere in this codebase, match this shape: take a grid
(or several), return a tidy DataFrame with one row per configuration and the relevant score column(s) —
don't introduce a different sweep abstraction.

---

### 9. Potential-Based Reward Shaping (new — `src/rl/reward_dynamics.py`)
```python
class ManifoldPotential:
    """Φ(x) = -||(x - μ_healthy) / σ_healthy||_2"""
    def potential(self, telemetry_vec): ...
    def shaping_bonus(self, prev_telemetry, next_telemetry, gamma):
        return gamma * self.potential(next_telemetry) - self.potential(prev_telemetry)
```
Standard Ng-Harada-Russell shaping: adding `F(s,a,s') = γΦ(s') − Φ(s)` to the reward doesn't change the
optimal policy, but densifies the reward signal toward the healthy-regime manifold centroid computed by
`drift.fit_healthy_envelope`. This is the concrete link between the RL branch and the drift-detection branch.

### 10. Reward Curriculum Pattern (new — `src/rl/reward_dynamics.py`)
```python
@dataclass
class RewardCurriculum:
    r_unstable_start: float
    r_unstable_final: float
    n_warmup_episodes: int
    def unstable_penalty(self, episode):
        frac = min(episode / self.n_warmup_episodes, 1.0)
        return self.r_unstable_start + frac * (self.r_unstable_final - self.r_unstable_start)
```
Linearly ramps the "unstable regime" penalty over training instead of fixing it, so early ε-greedy
exploration into `unstable` isn't punished as harshly as it would be once the policy has matured.

### 11. Count-Based Curiosity Bonus (new — `src/rl/exploration.py`)
```python
class CuriosityBonus:
    """β / sqrt(N(s,a) + 1), β decayed geometrically each episode."""
    def bonus(self, state, action): ...
    def update(self, state, action): ...   # increments N(s,a)
```
Standard count-based exploration bonus (MBIE-EB style), decoupled from the ε-greedy schedule
(`EpsilonSchedule`) — the two exploration mechanisms are composed, not merged into one class.

---

## Statistical Concepts

### 1. Manifold Hypothesis
High-dimensional telemetry (6D) lies on a low-dimensional manifold (≤3D). PCA: PC1 explains ~60% of
variance; 2–3 PCs sufficient for 90% variance. UMAP preserves local topology better than PCA; t-SNE is
fit on a stratified subsample for stability/speed (`fit_tsne(..., stratified=True)`).

### 2–6. Cluster/embedding quality metrics
Silhouette [−1,1]↑, Davies-Bouldin [0,∞)↓, Calinski-Harabasz [0,∞)↑, Trustworthiness [0,1]↑, Continuity
[0,1]↑ — formulas and interpretation unchanged from the original manifold-only version of this doc; see
`src/manifold/manifold_metrics.py` / `src/evaluation/manifold_quality.py` for the implementations.

### 7. Regime Connectivity (LBSM-specific, `src/manifold/umap_projection.py`)
Fraction of k-NN edges in the embedding that cross a regime boundary — high → regimes blend smoothly, low →
sharp boundaries.

### 8. HMM Model Selection: BIC / AIC (new — `src/hmm/sequence_inference.py`)
```
BIC = -2·log L + k·log(N)        AIC = -2·log L + 2k
```
where `k` is the number of free parameters (grows with `n_components`) and `N` the number of observations.
`model_selection_sweep()` fits a Gaussian HMM at each candidate `n_components` and reports both — used to
justify choosing 4 hidden states rather than assuming it.

### 9. Spectral Gap (new — `src/hmm/transition_analysis.py`)
```
spectral_gap(T) = 1 - |λ_2|
```
where λ_2 is the second-largest eigenvalue magnitude of the transition matrix. Larger gap → faster mixing /
shorter memory of past states; used to compare the learned vs. ground-truth transition matrices' dynamics,
not just their entries.

### 10. Mahalanobis Distance / Healthy Envelope (new — `src/drift/drift_detection.py`)
```
d_M(x) = sqrt( (x - μ)ᵀ Σ⁻¹ (x - μ) )
```
fit only on `healthy_regimes` observations (e.g. `stable`). This is also exactly what
`BehaviorProfile.mahalanobis()` and `rl.reward_dynamics.ManifoldPotential` use — one distance concept
reused across simulation validation, drift detection, and RL reward shaping.

### 11. EWMA Residual Scoring (new — `src/drift/ewma.py`)
```
S_t = α·x_t + (1-α)·S_{t-1}          score_t = |x_t - S_t|
```
Adaptive threshold computed after a `warmup` period (so the filter has stabilized before flagging begins).

### 12. Gaussian KL Divergence Drift (new — `src/drift/kl_divergence.py`)
For diagonal-covariance Gaussians (per-feature independent):
```
KL(p‖q) = Σ_i [ log(σ_qi/σ_pi) + (σ_pi² + (μ_pi-μ_qi)²)/(2σ_qi²) - 1/2 ]
```
computed between each sliding window's empirical distribution and a fixed healthy `fit_reference()`
distribution — `window_size` trades detection latency against noise.

### 13. Detection Latency & Shift Magnitude (new — `src/drift/regime_shift_analysis.py`)
Detectors are evaluated by how many timesteps after a `ground_truth_changepoints()` event they first flag
(`detection_latency`), not just by pointwise precision/recall — a detector that's accurate but slow is a
different failure mode than one that's fast but noisy, and this codebase measures both.

---

## Data Flow in Key Functions

### AdaptiveAgent.step() Flow — unchanged, see previous versions of this doc / `src/simulation/agent.py`.

### fit_hmm() Flow (new)
```
INPUT: X_concat (all agents stacked), lengths (per-agent sequence lengths), y_gt (for scoring only)

STEP 1: hmmlearn.GaussianHMM(n_components, covariance_type).fit(X_concat, lengths)  — Baum-Welch/EM
STEP 2: model.predict(X_concat, lengths)  — Viterbi decode -> pred_raw
STEP 3: Hungarian-align pred_raw to y_gt (per confusion matrix) -> pred_aligned, mapping
STEP 4: score: ARI(pred_raw, y_gt) [permutation-invariant], accuracy(pred_aligned, y_gt), confusion matrix
STEP 5: posteriors_all = model.predict_proba(X_concat, lengths)  — for posterior_entropy() downstream

OUTPUT: HMMResult
```

### QLearningAgent.train() Flow (new)
```
INPUT: BehavioralEnv, QLearningConfig, HealthyEnvelope (for reward shaping / Mahalanobis tracking)

PER EPISODE:
  1. env.reset() -> obs
  2. PER STEP: ε-greedy _select_action(obs) using EpsilonSchedule (+ optional CuriosityBonus)
     -> env.step(action) -> StepResult(obs', reward, done, info)
     -> reward may be shaped: RewardCurriculum.compute_reward() + ManifoldPotential.shaping_bonus()
     -> tabular Q-update: Q[s,a] += alpha * (r + gamma * max_a' Q[s',a'] - Q[s,a])
  3. Track EpisodeStats: total_reward, regime_fractions (dwell time per regime this episode),
     mean_mah_score (via _mahalanobis_from_envelope against HealthyEnvelope), epsilon, n_steps
  4. EpsilonSchedule.step() -> decay epsilon for next episode

OUTPUT: training_dataframe() (tidy per-episode log) feeding reward_tracking.py and adaptation_dynamics.py
```

### embedding_scorecard() Flow — unchanged, see `src/manifold/manifold_metrics.py`.

---

## Key Constants & Enumerations

### Hidden States / Telemetry Features / Default Transition Matrix
Unchanged — see `src/simulation/behavior_profiles.py` (`PROFILE_NAMES`, `TELEMETRY_FEATURES`) and
`src/simulation/agent.py` (`DEFAULT_TRANSITION_MATRIX`).

### RL Grid Constants (new — `src/rl/environment.py`, re-exported via `src/rl/__init__.py`)
```python
N_STATES, N_ACTIONS, N_GRID_LATENCY, N_GRID_ENTROPY
ACTION_PUSH_STABLE, ACTION_PUSH_EXPLORATORY, ACTION_DO_NOTHING
DELTA_BASE, N_STEPS_PER_EPISODE
```
`N_STATES = N_GRID_LATENCY * N_GRID_ENTROPY` — the same "index ↔ 2-tuple" mapping idea as `_STATE_INDEX`,
just over a 2-D discretization grid instead of the 4 named regimes.

---

## Error Handling & Validation
Unchanged from the original doc — `_validate_transition_matrix`, unknown-state checks. No new validation
idioms were introduced in `hmm`/`drift`/`rl`; they mostly trust well-formed inputs (arrays of the expected
shape from upstream `manifold`/`telemetry` calls) rather than re-validating.

---

## Reproducibility Practices
Unchanged: `rng_seed` / `base_seed` threaded through every factory (`make_agent_pool`, `make_env_pool`,
`train_agent_pool`), `random_state=42` for sklearn/UMAP, deterministic `PROFILE_NAMES` /
`TELEMETRY_FEATURES` ordering, `agent_{i:04d}` IDs for lexicographic sort stability.

---

## Type Hints & Documentation
NumPy-style docstrings, type hints on parameters, shape annotations. This style is consistent across the
newer `hmm`/`drift`/`rl` modules too — the extraction script in `CODEBASE_UNDERSTANDING.md`'s verification
note pulls the first docstring line for every function if you want to spot-check style compliance.

---

## Testing Patterns

**Only 2 of 6 test files have real assertions** — `tests/test_simulation.py` and `tests/test_manifold.py`.
The other four (`test_drift.py`, `test_hmm.py`, `test_metrics.py`, `test_projection.py`, `test_rl.py`) are
empty despite `hmm`, `drift`, `rl`, and `evaluation` all being fully implemented — this is the single
biggest coverage gap in the repo.

```python
# tests/test_simulation.py (real)
def test_agent_initialization():
    agent = AdaptiveAgent(agent_id="test_agent", initial_state="stable")
    assert agent.current_state == "stable"
    assert len(agent.history) == 0

def test_agent_step():
    agent = AdaptiveAgent(rng_seed=42)
    record = agent.step(timestep=0)
    assert record["timestep"] == 0
    assert record["hidden_state"] in PROFILE_NAMES
    assert all(feat in record for feat in TELEMETRY_FEATURES)

def test_transition_matrix_validation():
    invalid_T = np.array([[0.5, 0.5], [0.5, 0.5]])  # wrong shape
    with pytest.raises(ValueError):
        AdaptiveAgent._validate_transition_matrix(invalid_T)

def test_stationary_distribution():
    pi = AdaptiveAgent().stationary_distribution()
    assert np.allclose(pi.sum(), 1.0)
    assert (pi >= 0).all()

# tests/test_manifold.py (real)
def test_pca_explained_variance():
    result = fit_pca(X_synthetic, feature_names)
    assert result.explained_var.sum() <= 1.0001

def test_umap_hyperparameter_sweep():
    sweep_result = hyperparameter_sweep(X_synthetic, labels, ...)
    assert {"n_neighbors", "min_dist", "silhouette"} <= set(sweep_result.columns)

def test_embedding_scorecard_keys():
    scorecard = embedding_scorecard(X_high, X_embedded, labels)
    required = {"method", "silhouette", "davies_bouldin", "calinski_harabasz", "trustworthiness", "continuity"}
    assert required <= set(scorecard)
```

If you're picking up coverage work, `test_hmm.py`/`test_drift.py`/`test_rl.py` are the highest-value gaps —
each covers a fully-implemented module with real scientific claims (ARI/accuracy, detection latency, reward
convergence) that currently have zero automated verification.

---

This document captures the key design decisions, statistical foundations, and implementation patterns
throughout the LBSM codebase — including the hmm/drift/rl branches added since the previous version of this
doc, which described them as "planned."
