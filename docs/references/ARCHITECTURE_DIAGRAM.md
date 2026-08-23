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
│   Orchestrator: TelemetryGenerator (.run/.save/.load)                   │
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
│  (clip_features,        (zscore/minmax)         (rolling stats,        │
│   drop_incomplete,      windowing.py            temporal_diff,         │
│   temporal split)       (sliding windows)       composite_health)      │
│                         statistics.py (regime_summary, Fisher sep.)    │
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
│         ├─ per_regime_density() ──→ Regime KDE estimates              │
│         └─ regime_connectivity() ──→ Boundary porosity metrics         │
│                                                                          │
│  ┌─ fit_tsne() ──────┐                                                  │
│  │ (Alternative,     │                                                  │
│  │  stratified       │                                                  │
│  │  subsample)       │  TSNEResult(embedding, perplexity, kl_div)     │
│  └──────────────────┘                                                  │
│                                                                          │
│  trajectory_geometry.py ──→ TrajectoryStats per agent (path_length,   │
│                              tortuosity, mean_speed, n_transitions)    │
│  covariance_analysis.py ──→ EMPTY FILE (no implementation)             │
│                                                                          │
└────────────┬────────────────────────────────────────────────────────────┘
             │
             ├─────────────────────────────────────────┐
             ↓ (2D/3D embeddings with labels)         ↓ (Original X + embedded X)
┌─────────────────────────────────────────────────┐  ┌──────────────────────┐
│      VISUALIZATION LAYER — mostly empty         │  │  EVALUATION LAYER    │
├─────────────────────────────────────────────────┤  ├──────────────────────┤
│                                                 │  │                      │
│ manifold_plots.py — one-off script:            │  │ embedding_scorecard()│
│   hardcoded paths, writes lbsm_umap3d.html     │  │ ├─ silhouette       │
│   on import (not a reusable function library)  │  │ ├─ davies_bouldin   │
│                                                 │  │ ├─ calinski_harabasz│
│ trajectory_plots.py, heatmaps.py,              │  │ ├─ trustworthiness  │
│ temporal_dynamics.py, state_transitions.py,    │  │ └─ continuity       │
│ dashboard.py  ──→ EMPTY FILES                  │  │                      │
│                                                 │  │ clustering_metrics, │
│ (actual plotting happens inline in notebooks,  │  │ manifold_quality,   │
│  matplotlib/seaborn/plotly)                    │  │ stability_metrics,  │
│                                                 │  │ trajectory_metrics  │
└─────────────────────────────────────────────────┘  └──────────────────────┘
             ↑
             └─ Notebooks 01-05: Research pipeline (06-07 empty, not started)


PARALLEL ANALYSIS BRANCHES — all implemented
─────────────────────────────────────────────

┌──────────────────────────────────────────────────────────────────────────┐
│                              HMM LAYER                                    │
│          (Infer hidden state sequence from observed telemetry,          │
│           purely unsupervised — GT used only for evaluation)            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  hidden_state_model.py ──→ fit_hmm(): Gaussian HMM, Baum-Welch/EM,     │
│                             Viterbi decode → HMMResult (ARI, accuracy,  │
│                             confusion matrix, Hungarian-aligned mapping)│
│  sequence_inference.py ──→ model_selection_sweep() (BIC/AIC over       │
│                             n_components), stationary_distribution()   │
│  transition_analysis.py ──→ transition_matrix_error(), spectral_gap(), │
│                              expected_dwell_times()                    │
│  latent_state_metrics.py ──→ per_regime_accuracy(), per_agent_metrics()│
│                               (Hungarian-aligned), posterior_entropy()  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                             DRIFT LAYER                                   │
│    (Three independent detectors + regime-shift characterization,        │
│     evaluated against HMM-decoded ground-truth changepoints)            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  drift_detection.py ──→ fit_healthy_envelope() + fit_mahalanobis()     │
│                          (Gaussian-envelope distance) +                │
│                          combined_anomaly_score() (fuses Mah. + EWMA)  │
│  ewma.py ──→ fit_ewma() — adaptive-threshold EWMA residual scoring     │
│  kl_divergence.py ──→ fit_kl_detector() — sliding-window KL vs.        │
│                        reference distribution                          │
│  regime_shift_analysis.py ──→ ground_truth_changepoints(),             │
│                                detection_latency(), shift_magnitude()  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                               RL LAYER                                    │
│    (Tabular Q-learning over a discretized behavioral MDP; maps         │
│     training dynamics back onto manifold/HMM/drift geometry)           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  environment.py ──→ BehavioralEnv: MDP over (latency, entropy) grid,   │
│                      actions nudge the agent's transition-matrix row   │
│  q_learning.py ──→ QLearningAgent.train() (ε-greedy tabular Q-learning)│
│  exploration.py ──→ EpsilonSchedule, CuriosityBonus (count-based)      │
│  reward_dynamics.py ──→ ManifoldPotential (potential-based shaping),   │
│                          RewardCurriculum (ramped unstable penalty)    │
│  reward_tracking.py ──→ learning_curve_df(), convergence_table(),     │
│                          dwell_evolution_df()                          │
│  adaptation_dynamics.py ──→ manifold_trajectory_stats(),               │
│                              cluster_migration_table(),                │
│                              transition_entropy_series()                │
│  policy.py ──→ greedy_policy(), policy_entropy(), action_state_heatmap()│
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
  - name, means (6), stds (6), autocorr, description, color

PCAResult / UMAPResult / TSNEResult:
  - embedding: np.ndarray
  - method-specific fitted model / hyperparameters

HMMResult:
  - model, pred_raw, pred_aligned, posteriors_all
  - mapping: Dict[int,int] (Hungarian-aligned label mapping)
  - ari, accuracy, log_likelihood, confusion, convergence_ll, n_iter_actual

HealthyEnvelope / MahalanobisResult:
  - mu, cov_inv, cov, regime_params  /  scores, flags, threshold

EWMAResult / KLDriftResult:
  - scores, threshold, alpha  /  kl_scores, t_starts, flags, window_size

QLearningConfig / EpisodeStats:
  - alpha, gamma, epsilon_start/end/decay, n_episodes, seed
  - episode, total_reward, regime_fractions, mean_mah_score, epsilon, n_steps

BehavioralEnv:
  - discretized (latency, entropy) state grid; .step()/.reset()/.trajectory()


EXPERIMENT PIPELINE — placeholder scripts, currently EMPTY
────────────────────────────────────────────────────────────

experiments/baseline/run_baseline.py           → 0 lines
experiments/manifold/run_projection_experiment.py → 0 lines
experiments/drift/run_drift_experiment.py      → 0 lines
experiments/rl_adaptive/run_rl_experiment.py   → 0 lines

Each has a populated sibling config.yaml, but nothing reads it yet.
The real pipelines run inside notebooks 01–05, not these scripts.


JUPYTER NOTEBOOKS (Active Research Documents)
──────────────────────────────────────────────

01_telemetry_generation.ipynb (populated)
   └─ Simulation → Telemetry generation → State dynamics visualization

02_manifold_learning.ipynb (populated)
   └─ PCA/UMAP/t-SNE comparison → Manifold quality metrics → Interpretation

03_hmm_inference.ipynb (populated)
   └─ Gaussian HMM (Baum-Welch/Viterbi) → BIC model selection → GT comparison

04_anomaly_detection.ipynb (populated)
   └─ EWMA / KL / Mahalanobis detectors → threshold sweeps → detection latency

05_rl_behavioral_evolution.ipynb (populated)
   └─ Q-learning over BehavioralEnv → learning curves → manifold/cluster migration

06_manifold_visualization.ipynb (EMPTY — not started)
   └─ Planned: 3D interactive visualizations

07_final_experiment_analysis.ipynb (EMPTY — not started)
   └─ Planned: integrated cross-pipeline analysis
```

## Module Dependency Graph

```
src/
├── simulation/          ← Foundation (generates data) — IMPLEMENTED
│   ├── agent.py ◄─────── BehaviorProfile, telemetry generation
│   ├── behavior_profiles.py
│   ├── telemetry_generator.py
│   ├── environment.py, reward_dynamics.py  (thin/unused here — see rl/)
│
├── telemetry/           ← Data pipeline — IMPLEMENTED
│   ├── preprocessing.py, normalization.py, feature_extraction.py
│   ├── statistics.py, windowing.py
│
├── manifold/            ← Primary analysis — IMPLEMENTED
│   ├── pca.py ◄───────── Core linear embedding
│   ├── umap_projection.py ◄───── Primary nonlinear embedding
│   ├── tsne.py, manifold_metrics.py, trajectory_geometry.py
│   ├── covariance_analysis.py  — EMPTY
│
├── hmm/                 ← State inference — IMPLEMENTED
│   ├── hidden_state_model.py, sequence_inference.py
│   ├── transition_analysis.py, latent_state_metrics.py
│
├── drift/               ← Anomaly / regime-shift detection — IMPLEMENTED
│   ├── drift_detection.py, ewma.py, kl_divergence.py
│   ├── regime_shift_analysis.py
│
├── rl/                  ← Learning dynamics — IMPLEMENTED
│   ├── environment.py, q_learning.py, policy.py, exploration.py
│   ├── reward_dynamics.py, reward_tracking.py, adaptation_dynamics.py
│
├── evaluation/          ← Quantitative validation — IMPLEMENTED
│   ├── manifold_quality.py, clustering_metrics.py
│   ├── trajectory_metrics.py, stability_metrics.py, explained_variance.py
│
├── visualization/       ← Presentation — MOSTLY EMPTY
│   ├── manifold_plots.py  (one-off script, not a library)
│   ├── trajectory_plots.py, heatmaps.py, temporal_dynamics.py,
│   │   state_transitions.py, dashboard.py  — all EMPTY
│
└── utils/               ← Infrastructure — ALL EMPTY
    ├── logging_utils.py, experiment_tracking.py, io.py, random_seed.py
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
  2. Preprocess: normalize, handle missing data, window
  3. Fit embeddings: PCA, UMAP, t-SNE
  4. Evaluate: Silhouette, Davies-Bouldin, Trustworthiness, etc.
  5. HMM: recover hidden state sequence, compare to ground truth
  6. Drift: score every timestep/window with EWMA + KL + Mahalanobis,
     compare flagged points to HMM-derived changepoints
  7. RL: train Q-learning agents in BehavioralEnv, track how training
     reshapes manifold trajectories, HMM transition entropy, and
     anomaly scores over episodes
  8. Visualize / interpret (ad hoc in notebooks — no shared viz module yet)
```

## Experiment Pipeline (as scripts — currently unimplemented)

```
experiments/baseline/run_baseline.py            (0 lines)
  └─ intended: generate agents → simulate telemetry → save CSV

experiments/manifold/run_projection_experiment.py (0 lines)
  └─ intended: load telemetry → PCA/UMAP/t-SNE → evaluate metrics → visualize

experiments/drift/run_drift_experiment.py       (0 lines)
  └─ intended: detect regime shifts → analyze drift characteristics

experiments/rl_adaptive/run_rl_experiment.py    (0 lines)
  └─ intended: agent learning → track manifold trajectory → measure adaptation
```
