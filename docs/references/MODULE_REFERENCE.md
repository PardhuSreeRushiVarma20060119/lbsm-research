# LBSM Module Reference - Detailed Function Catalog

Every signature below was extracted directly from the source (via `ast`), not hand-transcribed — see
`CODEBASE_UNDERSTANDING.md`'s verification note for the extraction command. Files with no entry under a
module heading are empty stubs (called out explicitly).

## src/simulation/

### `agent.py` - AdaptiveAgent Class
**Purpose**: Core agent-based simulator with hidden Markov behavioral dynamics

- `AdaptiveAgent(agent_id, initial_state, transition_matrix, rng_seed)`
  - Properties: `.current_state`, `.current_profile`, `.history` (DataFrame)
  - `.step(timestep)` / `.simulate(n_steps, start_timestep)` / `.reset(initial_state, clear_history)`
  - `._transition()` / `._emit()` (private)
  - `.state_distribution()`, `.transition_counts()`, `.stationary_distribution()`
  - `._validate_transition_matrix(T)` (static): shape, row-stochasticity, non-negativity checks

**Constants**: `DEFAULT_TRANSITION_MATRIX` (4×4), `_STATE_INDEX`, `_INDEX_STATE`

**Factory Functions**: `make_agent(agent_id, initial_state, rng_seed)`, `make_agent_pool(n_agents, initial_states, base_seed)`

```python
from src.simulation import make_agent_pool
agents = make_agent_pool(n_agents=20, base_seed=42)
for agent in agents:
    df = agent.simulate(n_steps=2000)
```

---

### `behavior_profiles.py` - Behavioral Regime Definitions

**Constants**: `TELEMETRY_FEATURES` (6-tuple), `N_FEATURES = 6`, `PROFILE_NAMES`, `BEHAVIOR_PROFILES`

- `BehaviorProfile` (frozen dataclass): `name`, `means`, `stds`, `autocorr`, `description`, `color`
  - `.sample(n, rng, prev)` — AR-1 correlated draws: x_t = ρ·x_{t-1} + √(1-ρ²)·(mean + std·noise)
  - `.mahalanobis(x)` — distance from `x` to this profile's centroid
- `get_profile(name)`, `profile_distance_matrix()`, `regime_separability_ratio()`

---

### `telemetry_generator.py` - Simulation Orchestrator

- `TelemetryGenerator(n_agents, n_timesteps, initial_states, seed, verbose)`
  - `.run(reset_agents)` — execute simulation for all agents
  - `.save(path, include_z_scores)` / `.load(cls, path)` — CSV round-trip
  - `.data` (property), `.summary_statistics()`, `.state_frequencies()`, `.per_agent_statistics()`
  - `.feature_matrix(z_scored)`, `.labels()`
- `generate_and_save(output_path, n_agents, n_timesteps, seed)` — one-shot helper

### `environment.py`, `reward_dynamics.py`
Present but not the ones that matter for RL — the real environment/reward logic lives in `src/rl/environment.py` and `src/rl/reward_dynamics.py`.

---

## src/telemetry/

### `preprocessing.py`
- `clip_features(df, feature_cols, bounds)` — clip to physically valid ranges (uses `FEATURE_BOUNDS`)
- `enforce_dtypes(df, feature_cols)` — cast to float32
- `drop_incomplete(df, min_steps, agent_col)` — remove short agent sequences
- `temporal_train_test_split(df, test_frac, agent_col, time_col)` — early/late split per agent
- `to_feature_matrix(df, feature_cols, z_scored, dtype)` — DataFrame → 2-D array

### `normalization.py`
- `ZScoreParams` / `MinMaxParams` (dataclasses)
- `fit_zscore(X)` / `apply_zscore(X, params)` / `zscore_matrix(X)`
- `fit_minmax(X)` / `apply_minmax(X, params)`
- `normalize_scores(scores)` — rescale a 1-D anomaly-score array to [0,1]
- `zscore_dataframe(df, feature_cols, suffix)`

### `feature_extraction.py`
- `rolling_mean(df, feature_cols, window, agent_col, time_col, suffix)`
- `rolling_std(df, feature_cols, window, agent_col, time_col, suffix)`
- `temporal_diff(df, feature_cols, lag, agent_col, time_col, suffix)` — first-order difference (velocity)
- `composite_health_score(df, latency_col, error_col, reward_col, entropy_col)` — heuristic score in [0,1]
- `augment_phase_space(df, feature_cols, agent_col, time_col)` — adds temporal-diff columns

### `statistics.py`
- `regime_summary(df, feature_cols, regime_col, regime_order)` — mean ± std per regime
- `fisher_separability(df, feature_cols, regime_col)` — univariate Fisher ratio per feature
- `anomaly_rate_by_regime(df, anomaly_col, regime_col)`
- `per_agent_summary(df, feature_cols, agent_col, regime_col)` — dominant regime, mean reward/error/latency
- `bhattacharyya_distance(df, feature_cols, regime_col)` — pairwise regime-distribution distance

### `windowing.py`
- `sliding_windows(X, window_size, step)` — yields `(start_index, window_array)`
- `window_statistics(X, window_size, step, feature_names)` — per-window mean/std/min/max
- `reference_test_split(X, reference_size, test_size, step)` — yields `(reference_window, test_window, test_start)`
- `per_agent_windows(df, feature_cols, window_size, step, agent_col, time_col)`

---

## src/manifold/

### `pca.py` - Principal Component Analysis (Linear Baseline)
- `PCAResult` (dataclass): `embedding`, `explained_var`, `cumulative_var`, `loadings`, `pca_model`, `n_components_90`
- `fit_pca(X, feature_names, n_components, random_state) -> PCAResult`
- `regime_centroids_pca(embedding, labels, profile_names, n_pcs)`
- `inter_regime_pc_distances(embedding, labels, profile_names, n_pcs)`
- `loading_dominance(loadings, pc="PC1")`
- `print_pca_summary(result, top_n)`

```python
from src.manifold.pca import fit_pca, loading_dominance
result = fit_pca(X_normalized, feature_names, n_components=10)
pc1_importance = loading_dominance(result.loadings, "PC1")
```

---

### `umap_projection.py` - UMAP (Primary Nonlinear Embedding)
- `UMAPResult` (dataclass): `embedding`, `n_neighbors`, `min_dist`, `n_components`, `reducer`
- `fit_umap(X, n_components, n_neighbors, min_dist, metric, random_state, verbose) -> UMAPResult`
- `hyperparameter_sweep(X, labels, n_neighbors_grid, min_dist_grid, random_state)` — grid search, scored by silhouette
- `per_regime_density(embedding, labels, profile_names, grid_resolution)` — 2-D KDE per regime
- `regime_connectivity(embedding, labels, profile_names, k)` — fraction of k-NN edges crossing regime boundaries

```python
from src.manifold.umap_projection import fit_umap, regime_connectivity
result = fit_umap(X_normalized, n_components=2, n_neighbors=30)
conn = regime_connectivity(result.embedding, labels, profile_names, k=10)
```

---

### `tsne.py` - t-SNE (Alternative Nonlinear Embedding)
- `TSNEResult` (dataclass): `embedding`, `sample_idx`, `perplexity`, `kl_divergence`
- `fit_tsne(X, labels, n_sample, perplexity, max_iter, pca_init_components, random_state, stratified)` — fits on a **stratified subsample**, not the full set
- `perplexity_sweep(X, labels, perplexity_grid, n_sample, random_state)` — silhouette across perplexities
- `intra_regime_spread(embedding, labels, profile_names)` — mean intra-regime distance from centroid

---

### `manifold_metrics.py` - Quantitative Embedding Quality
- `embedding_scorecard(X_high, X_embedded, labels, method_name, sample_size, n_neighbors, random_state) -> Dict`
  - keys: `method`, `silhouette` [-1,1]↑, `davies_bouldin` [0,∞)↓, `calinski_harabasz` [0,∞)↑, `trustworthiness` [0,1]↑, `continuity` [0,1]↑
- `compare_embeddings(scorecards) -> pd.DataFrame` — sets `.attrs["higher_is_better"]`
- `continuity(X_high, X_embedded, n_neighbors)` — inverse of trustworthiness
- `per_regime_silhouette(X_embedded, labels, profile_names)`
- `embedding_agreement(emb_a, emb_b, labels, sample_size, random_state)` — Procrustes-based structural agreement
- `neighbourhood_purity(X_embedded, labels, profile_names, k)`

```python
from src.manifold.manifold_metrics import embedding_scorecard, compare_embeddings
pca_score = embedding_scorecard(X_normalized, pca_embedding, labels, "PCA")
umap_score = embedding_scorecard(X_normalized, umap_embedding, labels, "UMAP")
comparison = compare_embeddings([pca_score, umap_score])
```

---

### `trajectory_geometry.py`
- `TrajectoryStats` (dataclass): `agent_id`, `path_length`, `displacement`, `tortuosity`, `mean_speed`, `max_speed`, `n_transitions`
- `extract_agent_trajectories(embedding, df_full, agent_ids)` — maps embedding coords back to per-agent sequences
- `compute_trajectory_stats(trajectories, df_full) -> List[TrajectoryStats]`
- `transition_embedding_coords(embedding, df_full)` — coords at each regime transition
- `manifold_velocity(trajectories, df_full, window)` — instantaneous speed per timestep per agent
- `regime_arc_statistics(trajectories, df_full, profile_names)` — mean speed/length *within* each regime

### `covariance_analysis.py`
**Empty file — no implementation.**

---

## src/hmm/

Unsupervised hidden-state recovery via a Gaussian-emission HMM (`hmmlearn`). Ground truth is used only to
score the result (ARI, accuracy), never to fit the model.

### `hidden_state_model.py`
- `HMMResult` (dataclass): `model`, `pred_raw`, `pred_aligned`, `posteriors_all`, `mapping: Dict[int,int]` (Hungarian-aligned label mapping), `ari`, `accuracy`, `log_likelihood`, `confusion`, `convergence_ll: List[float]`, `n_iter_actual`
- `prepare_sequences(df, feature_cols, agent_col, time_col, z_scored)` — stacks per-agent sequences into the concatenated `(X_concat, lengths)` form `hmmlearn` expects
- `fit_hmm(X_concat, lengths, y_gt, n_components, covariance_type, n_iter, tol, random_state, profile_names) -> HMMResult` — Baum-Welch/EM fit + Viterbi decode

### `sequence_inference.py`
- `model_selection_sweep(X_concat, lengths, n_comp_grid, covariance_type, n_iter, tol, random_state)` — BIC/AIC across `n_components` (model-order selection)
- `stationary_distribution(transmat)` — limiting distribution of a row-stochastic matrix

### `transition_analysis.py`
- `transition_matrix_error(result, T_gt, profile_names)` — element-wise abs error, learned vs. ground truth
- `spectral_gap(transmat)` — 1 − |second-largest eigenvalue|
- `expected_dwell_times(transmat, profile_names)` — expected consecutive timesteps per state
- `empirical_transition_counts(state_seq, n_states, profile_names)`

### `latent_state_metrics.py`
- `per_regime_accuracy(result, profile_names)` — precision/recall/accuracy per regime
- `per_agent_metrics(result, y_gt, lengths, agent_ids, profile_names)` — Hungarian-aligned ARI/accuracy **per agent**
- `posterior_entropy(posteriors)` — Shannon entropy (nats) of the forward-backward posterior at each timestep

---

## src/drift/

Three independent detectors (Mahalanobis, EWMA, KL-divergence) plus tooling to characterize detected shifts
against HMM-derived ground-truth changepoints.

### `drift_detection.py`
- `HealthyEnvelope` (dataclass): `mu`, `cov_inv`, `cov`, `regime_params: Dict[str, Tuple[mean, cov]]`
- `MahalanobisResult` (dataclass): `scores`, `flags`, `threshold`
- `fit_healthy_envelope(df, feature_cols, regime_col, healthy_regimes, regularize)` — Gaussian fit on healthy-only data
- `mahalanobis_scores(X, envelope, mode)`
- `fit_mahalanobis(X, envelope, mode, threshold_pct, y_healthy)` — percentile-threshold flagging
- `combined_anomaly_score(mah_scores, ewma_scores, w_mah, w_ewma)` — weighted fusion of two detectors
- `threshold_sweep(scores, y_gt, n_pts)` — precision/recall/F1/FPR across a threshold grid

### `ewma.py`
- `EWMAResult` (dataclass): `scores`, `ewma_path`, `flags`, `threshold`, `alpha`
- `ewma_scores(X, alpha)` — EWMA residual scores for one agent sequence
- `fit_ewma(X, alpha, threshold_k, warmup)` — adaptive threshold, ignoring a warmup period
- `ewma_all_agents(df, feature_cols, alpha, threshold_k, warmup, agent_col, time_col)`
- `alpha_sweep(X, y_gt, alpha_grid)` — AUC across smoothing factors

### `kl_divergence.py`
- `KLDriftResult` (dataclass): `kl_scores`, `t_starts`, `flags`, `threshold`, `window_size`
- `gaussian_kl(mu_p, var_p, mu_q, var_q)` — KL(p‖q) for diagonal-covariance Gaussians
- `fit_reference(X_reference)` — fit a diagonal Gaussian reference distribution
- `kl_drift_scores(X, mu_ref, var_ref, window_size, step)` — KL(window‖reference) per sliding window
- `fit_kl_detector(X, mu_ref, var_ref, window_size, step, threshold_k, warmup_frac)`
- `kl_all_agents(df, feature_cols, mu_ref, var_ref, window_size, step, agent_col, time_col)`
- `window_size_sweep(X, y_gt, mu_ref, var_ref, window_grid)` — AUC across window sizes

### `regime_shift_analysis.py`
- `ground_truth_changepoints(df, regime_col, agent_col, time_col)` — every regime-transition timestep, from labels
- `detection_latency(flags, changepoints, max_lag)` — timesteps between a changepoint and its first flag
- `detection_latency_summary(df_cp, df_flags, flag_col, agent_col, time_col, max_lag)` — per-agent summary
- `shift_magnitude(df, feature_cols, df_cp, window, agent_col, time_col)` — feature-space magnitude of each transition
- `transition_shift_summary(df_mag)` — mean shift magnitude by (from_regime → to_regime)

---

## src/rl/

Tabular Q-learning over a discretized `(latency, entropy)` state grid, with reward shaping tied to the
"healthy manifold" concept from `drift/`, and analysis tools that map training dynamics back onto manifold
geometry (`manifold/`) and transition structure (`hmm/`).

### `environment.py`
- Constants (re-exported from `__init__`): `N_STATES`, `N_ACTIONS`, `N_GRID_LATENCY`, `N_GRID_ENTROPY`, `ACTION_PUSH_STABLE`, `ACTION_PUSH_EXPLORATORY`, `ACTION_DO_NOTHING`, `DELTA_BASE`, `N_STEPS_PER_EPISODE`
- `StepResult` (dataclass): `obs`, `reward`, `done`, `info`
- `obs_to_grid(latency, entropy)` / `grid_to_coords(state_idx)` — discretization ↔ inverse
- `_digitise(value, lo, hi, n_bins)` — clip-and-bin helper
- `BehavioralEnv(agent_id, rng_seed, delta, n_steps, record_traj)` — the MDP:
  - `.reset(rng_seed)`, `.step(action) -> StepResult`, `.trajectory()`, `.n_states()`, `.n_actions()`
- `_nudge_transition(T_default, from_state, action, delta)` — nudges the underlying agent's transition-matrix row toward stable/exploratory (this is what an "action" actually does)
- `_compute_reward(hidden_state, was_in_unstable)` — returns `(base_reward, exit_bonus_triggered)`
- `make_env_pool(n_envs, base_seed, delta, n_steps)`

### `q_learning.py`
- `QLearningConfig` (dataclass): `alpha`, `gamma`, `epsilon_start`, `epsilon_end`, `epsilon_decay`, `n_episodes`, `seed`
- `EpisodeStats` (dataclass): `episode`, `total_reward`, `regime_fractions`, `mean_mah_score`, `epsilon`, `n_steps`; `.unstable_frac`
- `QLearningAgent(env, config)`:
  - `.train(healthy_envelope, verbose)`, `._run_episode(ep_idx)`, `._select_action(obs)` (ε-greedy)
  - `.greedy_action(obs)`, `.evaluate(n_episodes, healthy_envelope)`
  - `.policy_map()`, `.value_map()`, `.training_dataframe()`
- `train_agent_pool(envs, config, healthy_envelope, verbose)` — one `QLearningAgent` per env
- `_mahalanobis_from_envelope(x, envelope)` — links back to `drift.HealthyEnvelope`

### `policy.py`
- `greedy_policy(Q)`, `policy_action_grid(policy)`, `value_grid(Q)`
- `policy_entropy(Q, temperature)` / `policy_entropy_grid(Q, temperature)` — softmax policy entropy, reshaped to grid
- `action_frequency_from_trajectory(trajectory)`, `action_state_heatmap(trajectory)`
- `policy_agreement(policy_a, policy_b)` — fraction of states where two policies agree
- `policy_summary_table(agents, agent_ids)`

### `exploration.py`
- `EpsilonSchedule` (dataclass): `epsilon_start`, `epsilon_end`, `decay_mode: "geometric"|"linear"`, `decay_param`
  - `.step()`, `.epsilon`, `.reset()`
- `geometric_epsilon(episode, epsilon_start, epsilon_end, decay)` / `linear_epsilon(episode, n_episodes, epsilon_start, epsilon_end)` — functional/stateless forms
- `CuriosityBonus(beta, decay_factor)` — count-based exploration bonus β/√(N(s,a)+1)
  - `.bonus(state, action)`, `.update(state, action)`, `.step_episode()`, `.reset()`, `.visit_counts()`, `.state_visit_counts()`
- `exploration_coverage(visit_counts, threshold)`, `state_coverage(visit_counts, threshold)`

### `reward_dynamics.py`
- `ManifoldPotential(shaping_coeff)` — Φ(x) = −‖(x−μ_healthy)/σ_healthy‖₂
  - `.potential(telemetry_vec)`, `.shaping_bonus(prev_telemetry, next_telemetry, gamma)` = γΦ(s′) − Φ(s)
- `RewardCurriculum` (dataclass): `r_unstable_start`, `r_unstable_final`, `n_warmup_episodes`, `r_healthy`, `r_exit`
  - `.unstable_penalty(episode)` — linearly ramped penalty
  - `.compute_reward(hidden_state, was_in_unstable, episode)` -> `(shaped_reward, exit_bonus_triggered)`
- `decompose_episode_rewards(trajectory)`, `reward_by_regime(trajectory)`

### `reward_tracking.py`
- `smooth(values, window, mode)` — moving-average
- `learning_curve_df(train_df, smooth_window)`, `pool_learning_curves(train_dfs, smooth_window)`
- `convergence_episode(unstable_fracs, threshold, n_consecutive)` — first episode where unstable_frac stays below threshold
- `convergence_table(train_dfs, threshold)`
- `dwell_evolution_df(train_df)` — episode-indexed dwell fractions for all 4 regimes
- `regime_delta_table(train_df, n_window)` / `pool_regime_summary(train_dfs, n_window)` — early vs. late episode comparison

### `adaptation_dynamics.py`
- `manifold_trajectory_stats(episode_trajectories, umap_embedding, feature_cols)`
- `cluster_migration_table(episode_trajectories, phase_boundaries)` — regime dwell fractions split into early/mid/late training phases
- `anomaly_score_evolution(episode_mah_means, smooth_window)` — mean Mahalanobis score across training
- `transition_entropy_series(episode_trajectories)` — empirical H(s′|s) per episode, a proxy for HMM transition complexity
- `umap_episode_centroids(episode_trajectories, X_umap, global_df, agent_id)`
- `regime_novelty_score(episode_trajectories, n_bins, feature_pair)` — fraction of newly-visited (latency, entropy) grid cells per episode

---

## src/evaluation/

Cross-notebook metrics shared by the manifold, HMM, and drift analyses.

### `clustering_metrics.py`
- `clustering_scorecard(X, labels, name, sample, seed)`, `per_class_silhouette(X, labels, class_names)`, `ari_score(y_true, y_pred)`

### `manifold_quality.py`
- `embedding_trustworthiness(X_high, X_embedded, n_neighbors)`, `embedding_continuity(X_high, X_embedded, n_neighbors)`
- `neighbourhood_purity(X_embedded, labels, class_names, k)`, `procrustes_agreement(emb_a, emb_b, n, seed)` — Pearson r of pairwise distances after Procrustes alignment

### `explained_variance.py`
- `pca_explained_variance(X, n_components, random_state)`, `n_components_for_threshold(explained_var, threshold)`, `intrinsic_dimensionality_estimate(X, seed)` — participation-ratio estimate

### `stability_metrics.py`
- `bootstrap_auc(scores, y_gt, n_boot, seed)` — bootstrap mean ± std of ROC-AUC
- `detector_stability_table(score_dict, y_gt, n_boot, seed)` — used to compare drift detectors' stability

### `trajectory_metrics.py`
- `path_length(traj)`, `displacement(traj)`, `tortuosity(traj)` (path_length/displacement), `mean_speed(traj)`, `trajectory_summary(trajectories)`

---

## src/visualization/ — mostly empty

- `manifold_plots.py` — **not a function library**. It's a standalone script: loads `data/raw/nb02/X_umap3.npy` /
  `y_labels.npy` via hardcoded relative paths, builds one `plotly.express.scatter_3d` figure, and calls
  `fig.write_html("lbsm_umap3d.html")` as an import-time side effect.
- `trajectory_plots.py`, `heatmaps.py`, `temporal_dynamics.py`, `state_transitions.py`, `dashboard.py` — **empty files**.

All real plotting in this project happens inline inside the notebooks.

---

## src/utils/ — all empty

`logging_utils.py`, `experiment_tracking.py`, `io.py`, `random_seed.py` have no implementation. Seed handling
is inlined everywhere else (`rng_seed=...`, `random_state=42`); there's no shared logger or experiment tracker.

---

## Configuration Files (`configs/`)

### `simulation.yaml` (populated)
```yaml
n_agents: 20
n_timesteps: 2000
initial_states: [stable, exploratory, adaptive, unstable]
transition_matrix: null  # Use DEFAULT_TRANSITION_MATRIX
random_seed: 42
```

### `projection.yaml` (populated)
```yaml
pca:
  n_components: 10
  random_state: 42
umap:
  n_components: 2
  n_neighbors: 30
  min_dist: 0.1
tsne:
  n_components: 2
  perplexity: 30
```

### `telemetry.yaml` (populated)
```yaml
preprocessing:
  remove_outliers: true
  outlier_threshold: 3  # sigma units
normalization:
  method: zscore
  feature_ranges: null
```

### `rl.yaml`, `experiments.yaml`
**Both empty files.** RL hyperparameters live in `QLearningConfig` defaults instead; there's no
multi-stage orchestration config yet.

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Python files (`src/`) | 55 (34 non-empty, 21 empty) |
| Python LOC (`src/`) | ~7,500 |
| Top-level function definitions | ~209 |
| Classes / dataclasses | ~24 |
| Jupyter Notebooks | 7 (01–05 populated, 06–07 empty) |
| Config Files | 5 (3 populated, 2 empty) |
| Experiment Scripts | 4 (`experiments/*/run_*.py`, all 0 lines) |
| Test Files | 6 (2 real: simulation, manifold; 4 empty: drift, hmm, metrics, projection, rl) |

---

## Typical Workflow

```python
# 1. SIMULATION: Generate ground-truth behavioral telemetry
from src.simulation import make_agent_pool
agents = make_agent_pool(n_agents=20, base_seed=42)
telemetry_data = [agent.simulate(n_steps=2000) for agent in agents]

# 2. PREPROCESSING: Clean and normalize
from src.telemetry.preprocessing import drop_incomplete, to_feature_matrix
from src.telemetry.normalization import zscore_matrix
df = drop_incomplete(pd.concat(telemetry_data))
X_raw = to_feature_matrix(df, feature_cols)
X_normalized, zparams = zscore_matrix(X_raw)

# 3. MANIFOLD LEARNING: Fit embeddings
from src.manifold.pca import fit_pca
from src.manifold.umap_projection import fit_umap
pca_result = fit_pca(X_normalized, feature_names)
umap_result = fit_umap(X_normalized, n_components=2)

# 4. EVALUATION: Quantify quality
from src.manifold.manifold_metrics import embedding_scorecard, compare_embeddings
pca_scores = embedding_scorecard(X_normalized, pca_result.embedding, labels, "PCA")
umap_scores = embedding_scorecard(X_normalized, umap_result.embedding, labels, "UMAP")
comparison = compare_embeddings([pca_scores, umap_scores])

# 5. HMM: Recover hidden states unsupervised
from src.hmm import prepare_sequences, fit_hmm
X_concat, lengths = prepare_sequences(df, feature_cols, agent_col="agent_id", time_col="timestep")
hmm_result = fit_hmm(X_concat, lengths, y_gt=df["hidden_state_idx"].values, n_components=4)

# 6. DRIFT: Score anomalies against a healthy envelope
from src.drift import fit_healthy_envelope, fit_mahalanobis, fit_ewma
envelope = fit_healthy_envelope(df, feature_cols, regime_col="hidden_state", healthy_regimes=["stable"])
mah_result = fit_mahalanobis(X_normalized, envelope)

# 7. RL: Train Q-learning agents and track manifold displacement
from src.rl import make_env_pool, QLearningConfig, train_agent_pool, manifold_trajectory_stats
envs = make_env_pool(n_envs=20)
agents_rl = train_agent_pool(envs, QLearningConfig(n_episodes=500), healthy_envelope=envelope)

# 8. VISUALIZATION: no reusable module — plot inline (matplotlib/seaborn/plotly) as the notebooks do
```

This is the complete module reference. Notebooks 01–05 instantiate this workflow end-to-end; 06–07 are not started.
