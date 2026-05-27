# LBSM Code Patterns & Implementation Details

## Key Design Patterns

### 1. Result Container Pattern (Dataclasses)
The codebase uses frozen dataclasses to encapsulate complex results:

```python
from dataclasses import dataclass

@dataclass(frozen=True)  # Immutable
class PCAResult:
    embedding: np.ndarray
    explained_var: np.ndarray
    cumulative_var: np.ndarray
    loadings: pd.DataFrame
    pca_model: PCA
    n_components_90: int

@dataclass(frozen=True)
class UMAPResult:
    embedding: np.ndarray
    n_neighbors: int
    min_dist: float
    n_components: int
    reducer: object  # umap.UMAP instance
```

**Rationale**: 
- Encapsulates related results together
- Immutable (frozen=True) prevents accidental modification
- Type-safe: IDE completion and type checking
- Self-documenting: attributes list all computed values

---

### 2. Factory Function Pattern
Convenience wrappers for object creation:

```python
def make_agent(
    agent_id: Optional[str] = None,
    initial_state: str = "stable",
    rng_seed: Optional[int] = None,
    **kwargs,
) -> AdaptiveAgent:
    """Convenience factory wrapping AdaptiveAgent."""
    return AdaptiveAgent(
        agent_id=agent_id,
        initial_state=initial_state,
        rng_seed=rng_seed,
        **kwargs,
    )

def make_agent_pool(
    n_agents: int,
    initial_states: Optional[Sequence[str]] = None,
    base_seed: int = 42,
) -> List[AdaptiveAgent]:
    """Create a heterogeneous pool of agents."""
    if initial_states is None:
        initial_states = list(PROFILE_NAMES)  # Cycle through all regimes
    
    agents = []
    for i in range(n_agents):
        state = initial_states[i % len(initial_states)]
        agent = AdaptiveAgent(
            agent_id=f"agent_{i:04d}",
            initial_state=state,
            rng_seed=base_seed + i,
        )
        agents.append(agent)
    return agents
```

**Rationale**:
- Simplifies common creation scenarios
- Enforces sensible defaults (cycling through regimes)
- Reproducible: base_seed + offset ensures different but deterministic seeds

---

### 3. Validation Pattern
Explicit validation with informative error messages:

```python
@staticmethod
def _validate_transition_matrix(T: np.ndarray) -> None:
    """Validate transition matrix properties."""
    k = len(PROFILE_NAMES)
    
    # Shape check
    if T.shape != (k, k):
        raise ValueError(f"Transition matrix must be ({k},{k}), got {T.shape}.")
    
    # Row-stochasticity check (all rows sum to 1)
    row_sums = T.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(
            f"All rows must sum to 1. Got row sums: {row_sums}."
        )
    
    # Non-negativity check
    if (T < 0).any():
        raise ValueError("Transition probabilities must be non-negative.")
```

**Rationale**:
- Fails fast with clear error messages
- Prevents silent data corruption
- Helps with debugging

---

### 4. AR-1 Temporal Correlation Pattern
Implementing realistic time-series autocorrelation:

```python
def sample(
    self,
    n: int = 1,
    rng: np.random.Generator | None = None,
    prev: np.ndarray | None = None,
) -> np.ndarray:
    """Draw samples with AR-1 correlation."""
    
    # Initialize RNG if not provided
    if rng is None:
        rng = np.random.default_rng()

    # Generate Gaussian noise
    noise = rng.normal(loc=0.0, scale=1.0, size=(n, N_FEATURES))
    samples = np.empty((n, N_FEATURES))

    # AR-1 process: x_t = ρ·x_{t-1} + √(1-ρ²)·noise
    x_prev = prev if prev is not None else self.means.copy()
    
    for t in range(n):
        x_t = self.autocorr * x_prev + np.sqrt(1.0 - self.autocorr**2) * (self.means + self.stds * noise[t])
        samples[t] = x_t
        x_prev = x_t

    return samples
```

**Rationale**:
- AR-1 model: x_t = ρ·x_{t-1} + (1-ρ)·mean + noise
- ρ ≈ 0.3: temporal smoothing (realistic behavioral persistence)
- Normalization by √(1-ρ²) maintains variance stationarity

---

### 5. Hidden State Management Pattern
Encapsulating state transitions:

```python
def _transition(self) -> None:
    """Sample next hidden state from Markov chain."""
    # Get transition probabilities from current state row
    probs = self._T[self._state_idx]
    
    # Sample next state
    self._state_idx = int(
        self._rng.choice(len(PROFILE_NAMES), p=probs)
    )

def step(self, timestep: int) -> Dict:
    """One timestep: emit → record → transition."""
    
    # 1. Emit telemetry under current state
    telemetry_vec = self._emit()
    
    # 2. Build record with metadata
    record: Dict = {
        "agent_id": self.agent_id,
        "timestep": timestep,
        "hidden_state": self.current_state,  # Ground truth (hidden to analysis)
    }
    for feat, val in zip(TELEMETRY_FEATURES, telemetry_vec):
        record[feat] = float(val)
    
    self._history.append(record)
    
    # 3. Stochastic transition (for *next* step)
    self._transition()
    
    return record
```

**Rationale**:
- Separation of concerns: emission vs. transition
- Hidden state remains opaque to external code (accessed via `current_state` property)
- History tracks both hidden states (for validation) and features (for analysis)

---

### 6. Index Mapping Pattern
Bidirectional mapping between states and indices:

```python
# Map regime name → integer index
_STATE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(PROFILE_NAMES)}
# Map integer index → regime name
_INDEX_STATE: Dict[int, str] = {i: name for i, name in enumerate(PROFILE_NAMES)}

@property
def current_state(self) -> str:
    """Current hidden behavioral regime name."""
    return _INDEX_STATE[self._state_idx]

# Usage in Markov chain
probs = self._T[_STATE_INDEX[from_state], _STATE_INDEX[to_state]]
```

**Rationale**:
- Numerical indices for efficient matrix operations
- String names for human readability
- Bidirectional mapping ensures consistency

---

## Statistical Concepts

### 1. Manifold Hypothesis
**Core Claim**: High-dimensional telemetry (6D) lies on a low-dimensional manifold (≤3D)

**Evidence**:
- PCA: PC1 explains ~60% of variance (healthy/unstable axis)
- PC2, PC3 capture intra-healthy separation
- Total of 2-3 PCs sufficient for 90% variance

**Manifold Learning Methods**:
- **PCA**: Linear projection (baseline)
- **UMAP**: Local topology preservation (primary)
  - Preserves neighborhood structure better than PCA
  - Reveals true manifold curvature
  - Hyperparameters: n_neighbors (30), min_dist (0.1)
- **t-SNE**: Global structure (alternative)
  - Good for visualization but less stable for analysis

---

### 2. Silhouette Coefficient
Cluster separation metric in embedding space:

```
silhouette(i) = (b(i) - a(i)) / max(a(i), b(i))

where:
  a(i) = average distance to points in same cluster
  b(i) = minimum average distance to other clusters
```

**Range**: [−1, 1]
- +1: perfect clustering
- 0: random/overlapping
- −1: inverted/wrong cluster assignment

**Interpretation for LBSM**:
- High silhouette → regimes form distinct clusters in embedding
- Validates manifold structure captures behavioral separation

---

### 3. Davies-Bouldin Index
Average similarity between each cluster and its most similar neighbor:

```
DB = (1/k) Σ max_{j≠i} (σ_i + σ_j) / d_{ij}

where:
  k = number of clusters
  σ_i = avg distance within cluster i
  d_ij = distance between cluster centroids
```

**Interpretation**:
- Lower is better (well-separated, compact clusters)
- 0 = perfect; high = overlapping/loose clusters

---

### 4. Calinski-Harabasz Score
Ratio of between-cluster to within-cluster variance:

```
CH = (Tr(S_B) / (k-1)) / (Tr(S_W) / (N-k))

where:
  S_B = between-cluster scatter
  S_W = within-cluster scatter
```

**Interpretation**:
- Higher is better (tight, well-separated clusters)
- Dimensionless ratio; scale-invariant

---

### 5. Trustworthiness
Measures neighborhood preservation from high-D to low-D:

```
T = 1 - (2 / (N·k·(2m - 3k - 1))) Σ_i Σ_u∈U(i) r(i, u)
```

**Meaning**:
- For each point in embedding, check if k-NN match high-D k-NN
- High T → local structure well-preserved
- Used to validate embedding quality

---

### 6. Continuity
Inverse of trustworthiness (embedding to high-D):

```
Validates that no "tears" introduced in embedding
```

---

### 7. Regime Connectivity
Custom metric (LBSM-specific):

```
Measures fraction of k-NN edges crossing regime boundaries

High connectivity → regimes blend smoothly in manifold
Low connectivity → sharp boundaries between regimes
```

**Formula**:
```python
def regime_connectivity(embedding, labels, k=10):
    """Fraction of k-NN cross-regime edges."""
    n_cross = 0
    n_total = 0
    
    for i in range(len(labels)):
        # Find k nearest neighbors
        distances = np.linalg.norm(embedding - embedding[i], axis=1)
        knn_indices = np.argsort(distances)[1:k+1]  # Skip self
        
        # Count cross-regime edges
        for j in knn_indices:
            n_total += 1
            if labels[i] != labels[j]:
                n_cross += 1
    
    return n_cross / n_total
```

---

## Data Flow in Key Functions

### AdaptiveAgent.step() Flow

```
BEFORE step(t):
  - _state_idx: current hidden state index
  - _prev_telemetry: last emission (for AR-1)

STEP 1: Emit telemetry
  ├─ Get current profile from _state_idx
  ├─ Sample from profile.sample() with AR-1 correlation
  ├─ Use _prev_telemetry as AR-1 seed
  └─ Update _prev_telemetry for next step

STEP 2: Build and store record
  ├─ Create dict with agent_id, timestep, hidden_state (ground truth!)
  ├─ Add features (latency, entropy, reward, memory_usage, error_rate, action_freq)
  ├─ Append to _history
  └─ Return record

STEP 3: Transition (for next step)
  ├─ Get transition probabilities from row _state_idx
  ├─ Sample next _state_idx from multinomial
  └─ (Hidden states updated but not visible in record)

AFTER step(t):
  - _state_idx: updated for next step
  - _history: extended with new record
  - _prev_telemetry: ready for next emission
```

### fit_umap() Flow

```
INPUT: X (N×6 feature matrix, z-scored)

STEP 1: Fit UMAP
  ├─ Create UMAP(n_neighbors=30, min_dist=0.1, ...)
  ├─ fit_transform(X) → embedding (N×2 or N×3)
  └─ Store fitted reducer for future transform()

STEP 2: Extract hyperparameters
  └─ Store n_neighbors, min_dist in result

OUTPUT: UMAPResult with embedding + metadata

POST-FIT ANALYSIS:
  ├─ regime_connectivity(embedding, labels) → Boundary porosity
  ├─ silhouette_score(embedding, labels) → Cluster separation
  └─ regime_centroids_umap(embedding, labels) → Cluster positions
```

### embedding_scorecard() Flow

```
INPUT: X_high (N×6), X_embedded (N×2), labels (N,), method_name

STEP 1: Cluster quality (in embedding space)
  ├─ silhouette_score(X_embedded, labels) → [-1, 1]
  ├─ davies_bouldin_score(X_embedded, labels) → [0, ∞)
  └─ calinski_harabasz_score(X_embedded, labels) → [0, ∞)

STEP 2: Topology preservation
  ├─ Sub-sample to 3000 points (speedup)
  ├─ trustworthiness(X_high[:3k], X_embedded[:3k]) → [0, 1]
  └─ continuity(X_high[:3k], X_embedded[:3k]) → [0, 1]

STEP 3: Build scorecard
  └─ Return dict {
        "method": "PCA" / "UMAP" / ...,
        "silhouette": float,
        "davies_bouldin": float,
        "calinski_harabasz": float,
        "trustworthiness": float,
        "continuity": float,
      }

COMPARE EMBEDDINGS:
  └─ compare_embeddings([scorecard1, scorecard2]) → pd.DataFrame
```

---

## Key Constants & Enumerations

### Hidden States (PROFILE_NAMES)
```python
PROFILE_NAMES = ("stable", "exploratory", "adaptive", "unstable")
```

### Telemetry Features (TELEMETRY_FEATURES)
```python
TELEMETRY_FEATURES = (
    "latency",        # ms   — response / action latency
    "entropy",        # bits — policy / action-distribution entropy
    "reward",         # a.u. — instantaneous reward signal
    "memory_usage",   # MB   — working-set memory footprint
    "error_rate",     # [0,1]— proportion of erroneous actions
    "action_freq",    # Hz   — actions per second
)
```

### Default Transition Matrix (4×4)
```python
DEFAULT_TRANSITION_MATRIX = np.array(
    #  stable  explor  adapt   unstab
    [[ 0.75,   0.15,   0.05,   0.05 ],   # from stable
     [ 0.05,   0.55,   0.30,   0.10 ],   # from exploratory
     [ 0.30,   0.10,   0.50,   0.10 ],   # from adaptive
     [ 0.10,   0.10,   0.10,   0.70 ]],  # from unstable
    dtype=np.float64,
)
```

**Interpretation**:
- Stable agents mostly stay stable (0.75) but explore (0.15)
- Exploratory agents learn and adapt (0.30)
- Adaptive agents maintain strategy (0.50) or stabilize (0.30)
- Unstable agents persist (0.70) or recover randomly

---

## Error Handling & Validation

### State Validation
```python
if initial_state not in _STATE_INDEX:
    raise ValueError(
        f"Unknown initial_state {initial_state!r}. "
        f"Valid: {list(_STATE_INDEX.keys())}"
    )
```

### Matrix Validation
```python
# Check shape
if T.shape != (4, 4):
    raise ValueError(f"Expected (4,4), got {T.shape}")

# Check row-stochasticity
if not np.allclose(T.sum(axis=1), 1.0, atol=1e-6):
    raise ValueError("Rows must sum to 1")

# Check non-negativity
if (T < 0).any():
    raise ValueError("Probabilities must be non-negative")
```

---

## Reproducibility Practices

### Random Seed Usage
```python
# Simulation
agent = AdaptiveAgent(rng_seed=42)
agents = make_agent_pool(base_seed=42)  # agent_i gets seed 42+i

# Manifold learning
pca = PCA(random_state=42)
umap = UMAP(random_state=42)

# Metrics
silhouette_score(..., random_state=42)
```

### Deterministic Ordering
```python
# Consistent feature ordering
TELEMETRY_FEATURES = ("latency", "entropy", "reward", ...)

# Consistent regime ordering
PROFILE_NAMES = ("stable", "exploratory", "adaptive", "unstable")

# Consistent agent IDs
f"agent_{i:04d}"  # Ensures lexicographic sorting: agent_0000, agent_0001, ...
```

---

## Type Hints & Documentation

### Example Function Signature
```python
def fit_pca(
    X: np.ndarray,                           # Input array with type hint
    feature_names: Tuple[str, ...],          # Tuple of strings
    n_components: Optional[int] = None,      # Optional with default
    random_state: int = 42,                  # Keyword with default
) -> PCAResult:                               # Return type hint
    """Fit PCA on a z-scored feature matrix.
    
    Parameters
    ----------
    X : np.ndarray  shape (N, d)
        Input feature matrix (should be z-scored).
    feature_names : tuple of str
        Column names (length d).
    n_components : int | None
        Number of components to retain. If None, keeps min(N, d).
    random_state : int
        Reproducibility seed for sklearn PCA.
    
    Returns
    -------
    result : PCAResult
        Object containing embedding, loadings, explained variance, etc.
        
    Reference
    ---------
    "Latent Behavioral State Machines: ..."
    Section 5.1 — Linear Baseline: Principal Component Analysis
    """
```

**Style**:
- NumPy docstring format
- Type hints on all parameters
- Clear shape annotations (e.g., "shape (N, d)")
- References to paper sections

---

## Testing Patterns (Planned)

```python
# tests/test_simulation.py
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
    assert len(agent.history) == 1

def test_transition_matrix_validation():
    invalid_T = np.array([[0.5, 0.5], [0.5, 0.5]])  # Wrong shape
    with pytest.raises(ValueError):
        AdaptiveAgent._validate_transition_matrix(invalid_T)

def test_stationary_distribution():
    agent = AdaptiveAgent()
    pi = agent.stationary_distribution()
    assert np.allclose(pi.sum(), 1.0)  # Probability distribution
    assert (pi >= 0).all()  # Non-negative

# tests/test_manifold.py
def test_pca_explained_variance():
    result = fit_pca(X_synthetic, feature_names)
    assert result.explained_var.sum() <= 1.0001  # Allow floating-point error
    assert result.cumulative_var[-1] <= 1.0001

def test_umap_hyperparameter_sweep():
    sweep_result = hyperparameter_sweep(X_synthetic, labels, ...)
    assert "n_neighbors" in sweep_result.columns
    assert "min_dist" in sweep_result.columns
    assert "silhouette" in sweep_result.columns

def test_embedding_scorecard_keys():
    scorecard = embedding_scorecard(X_high, X_embedded, labels)
    required_keys = ["method", "silhouette", "davies_bouldin", 
                     "calinski_harabasz", "trustworthiness", "continuity"]
    assert all(k in scorecard for k in required_keys)
```

---

This document captures the key design decisions, statistical foundations, and implementation patterns throughout the LBSM codebase.
