# LBSM Architecture Diagram

## Data Flow & Module Interactions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SIMULATION LAYER                                 │
│  (Generates ground-truth behavioral telemetry with hidden state)        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   BehaviorProfile  ────→  AdaptiveAgent (4 hidden states)              │
│   (μ, Σ, ρ)              Markov chain with AR-1 emission              │
│                          │                                              │
│                          ├─→ step() ──→ Telemetry record (6 features) │
│                          ├─→ simulate() ──→ DataFrame history          │
│                          └─→ reset() ──→ Clean slate                   │
│                                                                          │
│   Factory: make_agent() / make_agent_pool()                             │
│                                                                          │
└────────────┬────────────────────────────────────────────────────────────┘
             │
             ↓ (Raw telemetry: N agents × T timesteps × 6 features)
┌─────────────────────────────────────────────────────────────────────────┐
│                    TELEMETRY PROCESSING LAYER                            │
│              (Cleaning, normalization, feature extraction)              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  preprocessing.py  ──→  normalization.py  ──→  feature_extraction.py  │
│  (Remove nulls,         (Z-score)               (Derive features)      │
│   outliers)             windowing.py            statistics.py          │
│                                                                          │
└────────────┬────────────────────────────────────────────────────────────┘
             │
             ↓ (Processed feature matrix: N samples × 6 dimensions)
┌─────────────────────────────────────────────────────────────────────────┐
│                    MANIFOLD LEARNING LAYER                               │
│        (Compress high-dimensional telemetry to low-D structure)         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─ fit_pca() ──────┐                                                   │
│  │ (Linear          │                                                   │
│  │  baseline)       │                                                   │
│  └──────────────────┘  PCAResult(embedding, loadings, explained_var)  │
│         │                                                               │
│         ├─ regime_centroids_pca()  ──→ Centroid positions             │
│         ├─ inter_regime_pc_distances() ──→ Cluster geometry           │
│         └─ loading_dominance() ──→ Feature importance                 │
│                                                                          │
│  ┌─ fit_umap() ──────┐                                                  │
│  │ (Primary          │                                                  │
│  │  nonlinear        │                                                  │
│  │  method)          │                                                  │
│  └──────────────────┘  UMAPResult(embedding, n_neighbors, min_dist)   │
│         │                                                               │
│         ├─ hyperparameter_sweep() ──→ Grid search + Silhouette scores │
│         ├─ umap_per_regime_density() ──→ Regime KDE estimates        │
│         └─ regime_connectivity() ──→ Boundary porosity metrics         │
│                                                                          │
│  ┌─ fit_tsne() ──────┐                                                  │
│  │ (Alternative)     │                                                  │
│  └──────────────────┘  (Similar interface)                             │
│                                                                          │
└────────────┬────────────────────────────────────────────────────────────┘
             │
             ├─────────────────────────────────────────┐
             ↓ (2D/3D embeddings with labels)         ↓ (Original X + embedded X)
┌─────────────────────────────────────────────────┐  ┌──────────────────────┐
│      VISUALIZATION LAYER                        │  │  EVALUATION LAYER    │
├─────────────────────────────────────────────────┤  ├──────────────────────┤
│                                                 │  │                      │
│ manifold_plots.py  ──→ 2D/3D scatterplots     │  │ embedding_scorecard()│
│ trajectory_plots.py ──→ Trajectories           │  │ ├─ silhouette       │
│ heatmaps.py ──→ Transition matrices            │  │ ├─ davies_bouldin   │
│ temporal_dynamics.py ──→ Time series           │  │ ├─ calinski_harabasz│
│ state_transitions.py ──→ Regime flows          │  │ ├─ trustworthiness  │
│ dashboard.py ──→ Multi-panel views             │  │ └─ continuity       │
│                                                 │  │                      │
│ (Jupyter + matplotlib/plotly)                  │  │ → Comparison table  │
│                                                 │  │    (PCA vs UMAP)    │
└─────────────────────────────────────────────────┘  └──────────────────────┘
             ↑
             └─ Notebooks 01-02: Research pipeline


PARALLEL ANALYSIS BRANCHES
──────────────────────────

┌──────────────────────────────────────────────────────────────────────────┐
│                           HMM LAYER (Planned)                            │
│          (Infer hidden state sequence from observed telemetry)          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  hidden_state_model.py ──→ Forward/Backward/Viterbi algorithms        │
│  sequence_inference.py ──→ State recovery                              │
│  transition_analysis.py ──→ Empirical transition matrix recovery       │
│  latent_state_metrics.py ──→ State-space diagnostics                   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                          DRIFT LAYER (Planned)                           │
│        (Detect behavioral regime shifts over time)                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  drift_detection.py ──→ Primary detection algorithms                   │
│  regime_shift_analysis.py ──→ Characterize shift events                │
│  kl_divergence.py ──→ KL divergence-based metrics                      │
│  ewma.py ──→ Exponentially weighted moving average tracking            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                           RL LAYER (Planned)                             │
│        (Adaptive behavioral evolution through learning)                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  q_learning.py ──→ Q-learning policy optimization                      │
│  policy.py ──→ Policy abstraction layer                                │
│  exploration.py ──→ Exploration strategies (ε-greedy, softmax, etc.)   │
│  reward_tracking.py ──→ Cumulative rewards and learning curves         │
│  adaptation_dynamics.py ──→ How learning maps to manifold geometry    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘


KEY DATA STRUCTURES
───────────────────

AdaptiveAgent:
  - agent_id: str
  - _T: np.ndarray (4×4 transition matrix)
  - _state_idx: int (current hidden state index)
  - _history: List[Dict] (telemetry records)
  - _prev_telemetry: np.ndarray (for AR-1 correlation)

BehaviorProfile:
  - name: str
  - means: np.ndarray (6 features)
  - stds: np.ndarray (6 features)
  - autocorr: float (AR-1 coefficient)
  - description: str
  - color: str

PCAResult:
  - embedding: np.ndarray (N, n_components)
  - explained_var: np.ndarray
  - cumulative_var: np.ndarray
  - loadings: pd.DataFrame (feature × PC loadings)
  - pca_model: sklearn.PCA
  - n_components_90: int

UMAPResult:
  - embedding: np.ndarray (N, n_components)
  - n_neighbors: int
  - min_dist: float
  - n_components: int
  - reducer: umap.UMAP


EXPERIMENT PIPELINE
───────────────────

experiments/baseline/run_baseline.py
  └─→ Generate agents → Simulate telemetry → Save CSV

experiments/manifold/run_projection_experiment.py
  └─→ Load telemetry → PCA/UMAP/t-SNE → Evaluate metrics → Visualize

experiments/drift/run_drift_experiment.py
  └─→ Detect regime shifts → Analyze drift characteristics

experiments/rl_adaptive/run_rl_experiment.py
  └─→ Agent learning → Track manifold trajectory → Measure adaptation


JUPYTER NOTEBOOKS (Active Research Documents)
──────────────────────────────────────────────

01_telemetry_generation.ipynb
   └─ Simulation → Telemetry generation → State dynamics visualization

02_manifold_learning.ipynb
   └─ PCA/UMAP/t-SNE comparison → Manifold quality metrics → Interpretation

03_hmm_inference.ipynb (Planned)
   └─ State sequence recovery → Viterbi decoding

04_anomaly_detection.ipynb (Planned)
   └─ Outlier detection in manifold space

05_rl_behavioral_evolution.ipynb (Planned)
   └─ Learning trajectories → Adaptation dynamics

06_manifold_visualization.ipynb (Planned)
   └─ 3D interactive visualizations

07_final_experiment_analysis.ipynb (Planned)
   └─ Integrated cross-pipeline analysis
```

## Module Dependency Graph

```
src/
├── simulation/          ← Foundation (generates data)
│   ├── agent.py ◄─────── BehaviorProfile, telemetry generation
│   ├── behavior_profiles.py
│   ├── environment.py
│   ├── reward_dynamics.py
│   └── telemetry_generator.py
│
├── telemetry/           ← Data pipeline
│   ├── preprocessing.py
│   ├── normalization.py
│   ├── feature_extraction.py
│   ├── statistics.py
│   └── windowing.py
│
├── manifold/            ← Primary analysis
│   ├── pca.py ◄───────── Core linear embedding
│   ├── umap_projection.py ◄───── Primary nonlinear embedding
│   ├── tsne.py
│   ├── manifold_metrics.py
│   ├── trajectory_geometry.py
│   └── covariance_analysis.py
│
├── evaluation/          ← Quantitative validation
│   ├── manifold_quality.py
│   ├── clustering_metrics.py
│   ├── trajectory_metrics.py
│   ├── stability_metrics.py
│   └── explained_variance.py
│
├── visualization/       ← Presentation
│   ├── manifold_plots.py
│   ├── trajectory_plots.py
│   ├── heatmaps.py
│   ├── temporal_dynamics.py
│   ├── state_transitions.py
│   └── dashboard.py
│
├── hmm/                 ← State inference (Planned)
│   ├── hidden_state_model.py
│   ├── sequence_inference.py
│   ├── transition_analysis.py
│   └── latent_state_metrics.py
│
├── drift/               ← Anomaly detection (Planned)
│   ├── drift_detection.py
│   ├── regime_shift_analysis.py
│   ├── kl_divergence.py
│   └── ewma.py
│
├── rl/                  ← Learning dynamics (Planned)
│   ├── q_learning.py
│   ├── policy.py
│   ├── exploration.py
│   ├── reward_tracking.py
│   └── adaptation_dynamics.py
│
└── utils/               ← Infrastructure
    ├── logging_utils.py
    ├── experiment_tracking.py
    ├── io.py
    └── random_seed.py
```

## Temporal Execution Flow

```
Simulation Epoch (each agent, each timestep):
  1. Current hidden state: s_t ∈ {stable, exploratory, adaptive, unstable}
  2. Sample telemetry: x_t ~ N(μ_{s_t}, Σ_{s_t}) with AR-1 correlation
  3. Record: (agent_id, t, s_t, x_t)
  4. Transition: Sample s_{t+1} from row s_t of transition matrix
  5. Repeat for next timestep

Analysis Pipeline (after simulation):
  1. Collect all telemetry records → Feature matrix X (N×6)
  2. Preprocess: normalize, handle missing data
  3. Fit embeddings: PCA, UMAP, t-SNE
  4. Evaluate: Silhouette, Davies-Bouldin, Trustworthiness, etc.
  5. Visualize: 2D/3D scatter, trajectories, heatmaps
  6. Interpret: Manifold structure encodes behavioral regimes

Optional Downstream (Planned):
  - HMM: Infer hidden state sequence from x_t
  - Drift: Detect regime shifts over time windows
  - RL: Modify policies, track manifold trajectory
  - Anomaly: Identify outliers in learned manifold
```
