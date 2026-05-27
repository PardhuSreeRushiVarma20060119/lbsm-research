# LBSM Module Reference - Detailed Function Catalog

## src/simulation/

### `agent.py` - AdaptiveAgent Class
**Purpose**: Core agent-based simulator with hidden Markov behavioral dynamics

**Key Classes**:
- `AdaptiveAgent`: Main agent simulator
  - **Init Parameters**:
    - `agent_id: str | None` - Unique ID (auto-generated if None)
    - `initial_state: str` - Starting regime (default: "stable")
    - `transition_matrix: np.ndarray | None` - 4×4 row-stochastic matrix
    - `rng_seed: int | None` - Random seed for reproducibility

  - **Properties**:
    - `current_state: str` - Current hidden behavioral regime
    - `current_profile: BehaviorProfile` - Statistical profile of current state
    - `history: pd.DataFrame` - Full telemetry buffer as DataFrame

  - **Core Methods**:
    - `step(timestep: int) -> Dict` - Advance one timestep, return record
    - `simulate(n_steps: int, start_timestep: int = 0) -> pd.DataFrame` - Run n steps
    - `reset(initial_state: str, clear_history: bool)` - Reset to clean state
    - `_transition() -> None` - Sample next hidden state (private)
    - `_emit() -> np.ndarray` - Sample telemetry from current profile (private)

  - **Introspection Methods**:
    - `state_distribution() -> Dict[str, float]` - Empirical time in each regime
    - `transition_counts() -> np.ndarray` - Empirical (4×4) transition matrix
    - `stationary_distribution() -> np.ndarray` - Theoretical stationary π
    - `_validate_transition_matrix(T: np.ndarray)` - Row-sum & non-neg checks

**Constants**:
- `DEFAULT_TRANSITION_MATRIX`: Pre-configured 4×4 transition probabilities
- `_STATE_INDEX`: Dict mapping state names → indices
- `_INDEX_STATE`: Dict mapping indices → state names

**Factory Functions**:
- `make_agent(agent_id, initial_state, rng_seed, **kwargs) -> AdaptiveAgent`
- `make_agent_pool(n_agents, initial_states, base_seed) -> List[AdaptiveAgent]`

**Usage Example**:
```python
from src.simulation import make_agent_pool
agents = make_agent_pool(n_agents=20, base_seed=42)
for agent in agents:
    df = agent.simulate(n_steps=2000)  # DataFrame with telemetry
```

---

### `behavior_profiles.py` - Behavioral Regime Definitions
**Purpose**: Immutable statistical profiles for each hidden state

**Constants**:
- `TELEMETRY_FEATURES`: Tuple of 6 feature names (latency, entropy, reward, memory_usage, error_rate, action_freq)
- `N_FEATURES`: int = 6
- `PROFILE_NAMES`: Regime names (stable, exploratory, adaptive, unstable)
- `BEHAVIOR_PROFILES`: Dict mapping regime names → BehaviorProfile instances

**Key Classes**:
- `BehaviorProfile`: Dataclass (frozen)
  - **Attributes**:
    - `name: str` - Regime label
    - `means: np.ndarray` - Feature means (shape 6)
    - `stds: np.ndarray` - Feature standard deviations (shape 6)
    - `autocorr: float` - AR-1 coefficient ∈ [0, 1)
    - `description: str` - Plain-language description
    - `color: str` - Matplotlib color for visualization

  - **Methods**:
    - `sample(n: int, rng, prev: np.ndarray | None) -> np.ndarray` - Draw n samples (shape n×6)
      - Implements AR-1 temporal correlation: x_t = ρ·x_{t-1} + √(1-ρ²)·noise

---

### `environment.py`, `reward_dynamics.py`, `telemetry_generator.py`
(Details: Placeholder modules for extensibility; agent.py contains core logic)

---

## src/telemetry/

### `preprocessing.py`
**Purpose**: Raw telemetry cleaning and validation

**Expected Functions**:
- `remove_nulls(X: np.ndarray) -> np.ndarray` - Handle missing values
- `remove_outliers(X: np.ndarray, threshold: float) -> np.ndarray` - IQR or z-score based

---

### `normalization.py`
**Purpose**: Standardization and scaling

**Expected Functions**:
- `zscore_normalize(X: np.ndarray) -> np.ndarray` - Mean=0, Std=1
- `minmax_scale(X: np.ndarray) -> np.ndarray` - Scale to [0, 1]

---

### `feature_extraction.py`
**Purpose**: Derived feature engineering

**Expected Functions**:
- `extract_features(X_raw: np.ndarray) -> np.ndarray` - Augment with derived features

---

### `statistics.py`
**Purpose**: Descriptive statistics

**Expected Functions**:
- `compute_stats(X: np.ndarray) -> Dict` - Mean, std, min, max, percentiles

---

### `windowing.py`
**Purpose**: Temporal windowing and rolling statistics

**Expected Functions**:
- `rolling_window(X: np.ndarray, window_size: int, step: int) -> np.ndarray`
- `apply_rolling_stats(X: np.ndarray, window_size: int) -> np.ndarray` - Rolling mean, std, etc.

---

## src/manifold/

### `pca.py` - Principal Component Analysis (Linear Baseline)
**Purpose**: Linear dimensionality reduction for comparison baseline

**Key Classes**:
- `PCAResult`: Dataclass
  - **Attributes**:
    - `embedding: np.ndarray` - Projected points (N, n_components)
    - `explained_var: np.ndarray` - Variance ratio per PC
    - `cumulative_var: np.ndarray` - Cumulative variance
    - `loadings: pd.DataFrame` - Feature loadings (d, n_components)
    - `pca_model: sklearn.decomposition.PCA` - Fitted model
    - `n_components_90: int` - Min PCs for 90% variance

**Key Functions**:
- `fit_pca(X: np.ndarray, feature_names: Tuple[str, ...], n_components: int | None, random_state: int) -> PCAResult`
  - Fits PCA, computes explained variance, constructs loading matrix
  - Default: keeps min(N, d) components

- `regime_centroids_pca(embedding: np.ndarray, labels: np.ndarray, profile_names: Tuple, n_pcs: int) -> pd.DataFrame`
  - Compute per-regime centroids in PC space (shape: n_regimes × n_pcs)

- `inter_regime_pc_distances(embedding: np.ndarray, labels: np.ndarray, profile_names: Tuple, n_pcs: int) -> pd.DataFrame`
  - Pairwise Euclidean distances between regime centroids (symmetric distance matrix)

- `loading_dominance(loadings: pd.DataFrame, pc: str = "PC1") -> pd.Series`
  - Rank features by absolute loading on a given PC (descending)

- `print_pca_summary(result: PCAResult, top_n: int) -> None`
  - Pretty-print key diagnostics to stdout (component count to 90% variance, top loadings)

**Usage Example**:
```python
from src.manifold.pca import fit_pca, loading_dominance
result = fit_pca(X_normalized, feature_names, n_components=10)
print(f"PCs to 90% var: {result.n_components_90}")
pc1_importance = loading_dominance(result.loadings, "PC1")
```

---

### `umap_projection.py` - UMAP (Primary Nonlinear Embedding)
**Purpose**: Nonlinear dimensionality reduction preserving local topology

**Key Classes**:
- `UMAPResult`: Dataclass
  - **Attributes**:
    - `embedding: np.ndarray` - Projected points (N, n_components)
    - `n_neighbors: int` - Hyperparameter used
    - `min_dist: float` - Hyperparameter used
    - `n_components: int` - Embedding dimension
    - `reducer: umap.UMAP` - Fitted UMAP transformer

**Key Functions**:
- `fit_umap(X: np.ndarray, n_components: int, n_neighbors: int, min_dist: float, metric: str, random_state: int, verbose: bool) -> UMAPResult`
  - Fits UMAP embedding
  - Default: n_neighbors=30, min_dist=0.10, metric="euclidean"

- `hyperparameter_sweep(X: np.ndarray, labels: np.ndarray, n_neighbors_range: List[int], min_dist_range: List[float], ...) -> pd.DataFrame`
  - Grid search over (n_neighbors, min_dist) combinations
  - Returns silhouette scores for each setting
  - Identifies optimal hyperparameters

- `umap_per_regime_density(embedding: np.ndarray, labels: np.ndarray, profile_names: Tuple, bw: float) -> Dict`
  - KDE density estimation in 2D UMAP space per regime
  - Returns smoothed density for each regime

- `regime_connectivity(embedding: np.ndarray, labels: np.ndarray, profile_names: Tuple, k: int) -> pd.DataFrame`
  - Compute fraction of k-NN edges crossing regime boundaries
  - Quantifies manifold boundary "porosity" / smoothness
  - High value → regimes blend smoothly; Low → sharp boundaries

**Usage Example**:
```python
from src.manifold.umap_projection import fit_umap, regime_connectivity
result = fit_umap(X_normalized, n_components=2, n_neighbors=30)
conn = regime_connectivity(result.embedding, labels, profile_names, k=10)
print(f"Cross-regime 10-NN edges: {conn.values.mean():.2%}")  # Boundary porosity
```

---

### `tsne.py` - t-SNE (Alternative Nonlinear Embedding)
**Purpose**: t-SNE embedding for comparison with UMAP
(Similar interface to UMAP, differences in hyperparameters: perplexity, learning_rate)

---

### `manifold_metrics.py` - Quantitative Embedding Quality
**Purpose**: Compute suite of quality metrics for embeddings

**Key Functions**:
- `embedding_scorecard(X_high: np.ndarray, X_embedded: np.ndarray, labels: np.ndarray, method_name: str, sample_size: int, n_neighbors: int, random_state: int) -> Dict[str, float]`
  - Returns dict with keys:
    - `"method"`: str (embedding name)
    - `"silhouette"`: float in [-1, 1] (↑ better)
    - `"davies_bouldin"`: float (↓ better)
    - `"calinski_harabasz"`: float (↑ better)
    - `"trustworthiness"`: float in [0, 1] (↑ better)
    - `"continuity"`: float in [0, 1] (↑ better)

- `compare_embeddings(scorecards: List[Dict]) -> pd.DataFrame`
  - Builds comparison table (rows = methods, cols = metrics)
  - Sets `.attrs["higher_is_better"]` annotation for interpretation

- `continuity(X_high: np.ndarray, X_embedded: np.ndarray, n_neighbors: int) -> float`
  - Inverse of trustworthiness; detects local "tears" in embedding

**Interpretation**:
- **Silhouette**: [−1, 1] — cluster separation in embedding space
  - +1: perfect clusters; 0: random; −1: overlapping/inverted
- **Davies-Bouldin**: ≥0 — average compactness/separation ratio
  - Lower is better (well-separated, tight clusters)
- **Calinski-Harabasz**: ≥0 — ratio of between/within variance
  - Higher is better (distinct clusters)
- **Trustworthiness**: [0, 1] — local neighborhood preservation from X_high to X_embedded
  - 1: perfect; 0: worst-case
- **Continuity**: [0, 1] — inverse (neighborhood preservation from X_embedded to X_high)
  - 1: perfect; 0: worst-case

**Usage Example**:
```python
from src.manifold.manifold_metrics import embedding_scorecard, compare_embeddings

pca_score = embedding_scorecard(X_normalized, pca_embedding, labels, "PCA")
umap_score = embedding_scorecard(X_normalized, umap_embedding, labels, "UMAP")
comparison = compare_embeddings([pca_score, umap_score])
print(comparison)  # Side-by-side metric comparison
```

---

### `trajectory_geometry.py`, `covariance_analysis.py`
(Module structure: Geometric properties of agent trajectories and covariance structure analysis)

---

## src/evaluation/

### `manifold_quality.py`
(Delegates to `manifold_metrics.py`; aggregation/pipeline functions)

---

### `clustering_metrics.py`
**Purpose**: Cluster validation independent of embedding

**Expected Functions**:
- `silhouette_per_sample(X_embedded: np.ndarray, labels: np.ndarray) -> np.ndarray` - Per-point silhouette
- `davies_bouldin_index(X_embedded: np.ndarray, labels: np.ndarray) -> float`
- `calinski_harabasz_index(X_embedded: np.ndarray, labels: np.ndarray) -> float`

---

### `trajectory_metrics.py`
**Purpose**: Properties of trajectories in feature/manifold space

**Expected Functions**:
- `trajectory_length(X: np.ndarray) -> float` - Total distance traversed
- `trajectory_velocity(X: np.ndarray) -> np.ndarray` - Per-timestep speed
- `trajectory_acceleration(X: np.ndarray) -> np.ndarray` - Per-timestep acceleration

---

### `stability_metrics.py`
**Purpose**: Temporal stability of behavioral regimes

**Expected Functions**:
- `regime_persistence(labels: np.ndarray, regime: str) -> float` - Average time in regime before transition
- `regime_volatility(X: np.ndarray, labels: np.ndarray, regime: str) -> float` - Within-regime variance

---

### `explained_variance.py`
**Purpose**: Dimensionality and variance diagnostics

**Expected Functions**:
- `cumulative_explained_variance(explained_var: np.ndarray) -> np.ndarray`
- `dimensionality_estimate(X: np.ndarray, threshold: float = 0.95) -> int` - Min dims for threshold% variance

---

## src/hmm/ (Planned)

### `hidden_state_model.py` - HMM Inference
**Purpose**: Recover hidden state sequence from observed telemetry using HMM

**Expected Classes**:
- `HiddenStateModel`: HMM wrapper
  - Methods: `fit()`, `decode()` (Viterbi), `forward_backward()`

**Expected Functions**:
- `decode_sequence(X_obs: np.ndarray, transition_matrix: np.ndarray, emission_params: Dict) -> np.ndarray` - Infer state sequence

---

### `sequence_inference.py`
- Viterbi algorithm, Forward-Backward algorithm
- `viterbi_decode()`, `forward_backward_algorithm()`

---

### `transition_analysis.py`
- Extract empirical transition matrix from inferred sequences
- `empirical_transition_matrix(states: np.ndarray) -> np.ndarray`

---

### `latent_state_metrics.py`
- State duration, state entropy, etc.
- `average_state_duration()`, `state_entropy()`

---

## src/drift/ (Planned)

### `drift_detection.py`
**Purpose**: Detect behavioral regime shifts

**Expected Functions**:
- `adwin_detector(X: np.ndarray, delta: float) -> List[int]` - Adaptive Windowing drift detector
- `ddm_detector(X: np.ndarray, lambda_val: float) -> List[int]` - Drift Detection Method
- `eddm_detector(X: np.ndarray) -> List[int]` - Early DDM variant

---

### `regime_shift_analysis.py`
- Characterize detected shifts
- `shift_magnitude()`, `shift_duration()`, `shift_smoothness()`

---

### `kl_divergence.py`
- KL divergence-based drift metric
- `kl_divergence_time_series()`, `kl_based_drift_score()`

---

### `ewma.py`
- Exponentially weighted moving average
- `ewma_track()`, `ewma_anomaly_score()`

---

## src/rl/ (Planned)

### `q_learning.py`
**Purpose**: Q-learning agent for policy optimization

**Expected Classes**:
- `QLearningAgent`: Q-learning implementation
  - Methods: `choose_action()`, `learn()`, `get_q_table()`

---

### `policy.py`
- Policy abstraction: π(a|s)
- `GreedyPolicy`, `EpsilonGreedyPolicy`, `SoftmaxPolicy`

---

### `exploration.py`
- Exploration strategies
- `epsilon_greedy()`, `softmax_exploration()`, `ucb_exploration()`

---

### `reward_tracking.py`
- Track cumulative rewards, learning curves
- `cumulative_reward()`, `running_average_reward()`

---

### `adaptation_dynamics.py`
- Map policy learning to manifold geometry
- `track_policy_trajectory()`, `learning_curve_geometry()`

---

## src/visualization/

### `manifold_plots.py`
**Purpose**: 2D/3D scatterplots of embeddings

**Expected Functions**:
- `scatter_2d(embedding: np.ndarray, labels: np.ndarray, ...)` - 2D scatterplot
- `scatter_3d(embedding: np.ndarray, labels: np.ndarray, ...)` - 3D scatterplot
- `plot_embeddings_comparison(embeddings_dict: Dict, labels: np.ndarray)` - Multi-panel comparison

---

### `trajectory_plots.py`
**Purpose**: Trajectory visualization

**Expected Functions**:
- `plot_trajectory_2d()`, `plot_trajectory_3d()`
- `plot_regime_trajectories()` - Separate by regime

---

### `heatmaps.py`
**Purpose**: Matrix visualizations

**Expected Functions**:
- `plot_transition_matrix_heatmap(T: np.ndarray, profile_names: Tuple)`
- `plot_covariance_heatmap(Σ: np.ndarray, feature_names: Tuple)`

---

### `temporal_dynamics.py`
**Purpose**: Time series plots

**Expected Functions**:
- `plot_feature_timeseries()` - Line plot over timesteps
- `plot_state_sequence()` - Timeline of regime transitions

---

### `state_transitions.py`
**Purpose**: Regime flow diagrams

**Expected Functions**:
- `plot_transition_flow()` - Sankey diagram of state transitions
- `plot_regime_duration_histogram()` - Distribution of time in each regime

---

### `dashboard.py`
**Purpose**: Multi-panel analysis dashboard

**Expected Functions**:
- `create_analysis_dashboard()` - Composite figure with all key plots

---

## src/utils/

### `logging_utils.py`
- Logging configuration and helpers

### `experiment_tracking.py`
- Experiment metadata, results storage, run tracking

### `io.py`
- File I/O (CSV, pickle, JSON, etc.)

### `random_seed.py`
- Seed management for reproducibility

---

## Configuration Files (`configs/`)

### `simulation.yaml`
```yaml
n_agents: 20
n_timesteps: 2000
initial_states: [stable, exploratory, adaptive, unstable]
transition_matrix: null  # Use DEFAULT_TRANSITION_MATRIX
random_seed: 42
```

### `projection.yaml`
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

### `telemetry.yaml`
```yaml
preprocessing:
  remove_outliers: true
  outlier_threshold: 3  # sigma units
normalization:
  method: zscore
  feature_ranges: null
```

### `experiments.yaml`
- Orchestration of multi-stage pipelines

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Python Modules | 51 |
| Python LOC | ~2,191 |
| Function Definitions | ~58 |
| Data Classes | ~5 (PCAResult, UMAPResult, BehaviorProfile, HiddenStateModel?, etc.) |
| Jupyter Notebooks | 7 (2 populated, 5 planned) |
| Config Files | 5 |
| Experiment Scripts | 4 |
| Test Files | 6 (all empty) |
| Data Files | 3 |

---

## Typical Workflow

```python
# 1. SIMULATION: Generate ground-truth behavioral telemetry
from src.simulation import make_agent_pool
agents = make_agent_pool(n_agents=20, base_seed=42)
telemetry_data = []
for agent in agents:
    df = agent.simulate(n_steps=2000)
    telemetry_data.append(df)

# 2. PREPROCESSING: Clean and normalize
from src.telemetry.preprocessing import remove_outliers
from src.telemetry.normalization import zscore_normalize
X_raw = preprocess(telemetry_data)
X_normalized = zscore_normalize(X_raw)

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

# 5. VISUALIZATION: Present results
from src.visualization.manifold_plots import scatter_2d
scatter_2d(umap_result.embedding, labels, title="UMAP Embedding of Behavioral Regimes")
```

This is the complete module reference. Notebooks 01-02 instantiate this workflow with data generation and manifold learning.
