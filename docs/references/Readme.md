# LBSM Research Ref - Quick Reference Index

## 📚 Documentation Files In This Directory :

### 1. **[CODEBASE_UNDERSTANDING](./CODEBASE_UNDERSTANDING.md)**
   - **Overview**: Project vision, architecture, research hypothesis
   - **Sections**:
     - Project Overview & Core Hypothesis
     - Complete Architecture (9 modules)
     - Data & Configuration structure
     - Jupyter Notebooks (research pipeline)
     - Key Scientific Concepts
     - Technology Stack
     - Project Status
     - Code Organization Principles
   - **Best for**: Getting oriented, understanding big picture

### 2. **[ARCHITECTURE_DIAGRAM](./ARCHITECTURE_DIAGRAM.md)**
   - **Visual representations** of the system:
     - Data Flow Diagram (simulation → processing → analysis)
     - Module Dependency Graph
     - Temporal Execution Flow
     - Experiment Pipeline Overview
   - **Key diagrams**:
     - End-to-end pipeline (main layers + the hmm/drift/rl branches)
     - Data structures illustration
   - **Best for**: Understanding system structure, data dependencies
   - for mermaid representations : [ARCHITECTURE_DIAGRAM-mermaid](./ARCHITECTURE_DIAGRAM-mermaid.md)

### 3. **[MODULE_REFERENCE](./MODULE_REFERENCE.md)**
   - **Detailed function catalog** for every module:
     - Class definitions with all methods
     - Function signatures with parameter details
     - Return type documentation
     - Which files are real vs. empty stubs
     - Configuration file schemas
   - **Coverage**: all 9 `src/` packages (55 files; see counts below)
   - **Best for**: API reference, looking up specific functions

### 4. **[CODE_PATTERNS_DETAILS](./CODE_PATTERNS_DETAILS.md)**
   - **Implementation deep-dives**:
     - Design patterns used throughout codebase
     - Statistical concepts & formulas (incl. HMM, drift, RL math)
     - Data flow in key functions
     - Error handling & validation
     - Type hints & documentation style
     - Testing patterns (what's real vs. empty)
   - **Best for**: Code style, patterns, implementation details

---

## 🎯 Quick Facts

*(counts as of the last doc refresh — re-run the commands in `CODEBASE_UNDERSTANDING.md`'s "How these docs were verified" note if code has moved since)*

| Metric | Value |
|--------|-------|
| **Repository** | PardhuSreeRushiVarma20060119/lbsm-research |
| **Main Branch** | main |
| **Python files (`src/`)** | 55 (34 non-empty, 21 empty stubs) |
| **Python LOC (`src/`)** | ~7,500 |
| **Top-level functions** | ~209 |
| **Classes / dataclasses** | ~24 |
| **Test files** | 6 (2 real: `test_simulation.py`, `test_manifold.py`; 4 empty: drift, hmm, metrics, projection, rl) |
| **Notebooks** | 7 (01–05 populated, 06–07 empty) |
| **Config Files** | 5 YAML (3 populated: simulation, telemetry, projection; 2 empty: rl, experiments) |
| **Experiment scripts** | 4 (`experiments/*/run_*.py`, **all currently empty** — 0 lines) |

---

## 🏗️ Architecture at a Glance

```
Simulation Layer (agent.py)
        ↓
Telemetry Processing (preprocessing, normalization, windowing, feature_extraction, statistics)
        ↓
Manifold Learning (PCA, UMAP, t-SNE, trajectory_geometry)
        ↓
┌─────────────────────────────────────────┐
│  Evaluation (clustering/trajectory/     │
│  stability metrics, embedding_scorecard)│
└─────────────────────────────────────────┘
        ↓
Parallel Branches — all implemented:
  hmm/   (state recovery)
  drift/ (EWMA, KL, Mahalanobis anomaly detection)
  rl/    (Q-learning behavioral adaptation)
        ↓
Visualization — mostly stubs (see status below)
```

---

## 📦 Key Classes & Data Structures

| Class | Module | Purpose |
|-------|--------|---------|
| `AdaptiveAgent` | simulation/agent.py | Core simulator with hidden Markov chain |
| `BehaviorProfile` | simulation/behavior_profiles.py | Statistical profile of each regime |
| `PCAResult` / `UMAPResult` / `TSNEResult` | manifold/*.py | Embedding result containers |
| `HMMResult` | hmm/hidden_state_model.py | Fitted HMM + decoded states + accuracy |
| `HealthyEnvelope` / `MahalanobisResult` | drift/drift_detection.py | Gaussian envelope + anomaly scores |
| `EWMAResult` / `KLDriftResult` | drift/ewma.py, kl_divergence.py | Online drift-detector results |
| `QLearningAgent` / `QLearningConfig` | rl/q_learning.py | Tabular Q-learning over `BehavioralEnv` |
| `BehavioralEnv` | rl/environment.py | Discrete-state MDP wrapping agent telemetry |
| `ManifoldPotential` / `RewardCurriculum` | rl/reward_dynamics.py | Potential-based reward shaping |

---

## 🔑 Key Functions by Category

### Simulation
- `make_agent()` / `make_agent_pool()` — create agent(s)
- `AdaptiveAgent.step()` / `.simulate()` — advance timesteps

### Manifold Learning
- `fit_pca()`, `fit_umap()`, `fit_tsne()` — embeddings
- `hyperparameter_sweep()` — UMAP grid search
- `regime_connectivity()` — boundary porosity metric

### HMM (`src/hmm`)
- `fit_hmm()` — Baum-Welch EM + Viterbi decode → `HMMResult`
- `model_selection_sweep()` — BIC/AIC over `n_components`
- `per_agent_metrics()` — Hungarian-aligned per-agent accuracy

### Drift (`src/drift`)
- `fit_ewma()`, `fit_kl_detector()`, `fit_mahalanobis()` — three independent detectors
- `combined_anomaly_score()` — fuses Mahalanobis + EWMA
- `detection_latency()`, `shift_magnitude()` — evaluate against ground-truth changepoints

### RL (`src/rl`)
- `QLearningAgent.train()` — tabular Q-learning over `BehavioralEnv`
- `manifold_trajectory_stats()`, `cluster_migration_table()` — map training onto manifold geometry

### Evaluation
- `embedding_scorecard()` — full quality metrics
- `compare_embeddings()` — comparison table
- `bootstrap_auc()` / `detector_stability_table()` — drift-detector stability

---

## 📊 Key Concepts

### Hidden States (4 regimes)
1. **Stable** — Low-volatility, high-confidence (0.75 self-loop)
2. **Exploratory** — High entropy, curiosity-driven (0.55 self-loop)
3. **Adaptive** — Learning and policy integration (0.50 self-loop)
4. **Unstable** — High-variance, anomalous (0.70 self-loop)

### Telemetry Features (6 dimensions)
1. latency (ms) — response time
2. entropy (bits) — action distribution entropy
3. reward (a.u.) — reward signal
4. memory_usage (MB) — resource usage
5. error_rate ([0,1]) — failure proportion
6. action_freq (Hz) — actions per second

### Quality Metrics
- **Silhouette** [−1, 1]: Cluster separation (↑ better)
- **Davies-Bouldin** [0, ∞): Cluster compactness (↓ better)
- **Calinski-Harabasz** [0, ∞): Variance ratio (↑ better)
- **Trustworthiness / Continuity** [0, 1]: Neighborhood preservation (↑ better)

---

## 🔄 Research Hypothesis

**Core Claim**: Adaptive behavioral systems generate telemetry that occupies **structured low-dimensional manifolds** rather than arbitrary high-dimensional space.

**Evidence Sought**:
- High manifold quality metrics (silhouette > 0.5, trustworthiness > 0.7)
- Clear cluster separation (regimes distinguish in embedding)
- Successful state recovery (HMM inference accuracy)
- RL training trajectories following (or opening new) manifold structure

---

## 🚀 Current Status

### ✅ Implemented (real code, not stubs)
- `simulation`, `telemetry`, `manifold`, `hmm`, `drift`, `rl`, `evaluation` — all have working functions
- Notebooks 01–05 (telemetry generation → manifold learning → HMM inference → anomaly detection → RL evolution)

### 🟡 Stubbed / empty
- `src/visualization/` — only `manifold_plots.py` has content, and it's a one-off script (hardcoded paths, writes `lbsm_umap3d.html`), not a reusable plotting library
- `src/utils/` — all four files empty (`io.py`, `logging_utils.py`, `experiment_tracking.py`, `random_seed.py`)
- `src/manifold/covariance_analysis.py` — empty
- `experiments/*/run_*.py` — all four scripts are empty (0 lines); the actual pipelines live in the notebooks
- `configs/rl.yaml`, `configs/experiments.yaml` — empty; RL hyperparameters are defined in code (`QLearningConfig` defaults) instead
- `tests/test_drift.py`, `test_hmm.py`, `test_metrics.py`, `test_projection.py`, `test_rl.py` — empty placeholders despite the modules they'd cover being implemented

### 📋 Not started
- `notebooks/06_manifold_visualization.ipynb`, `notebooks/07_final_experiment_analysis.ipynb` — 0 bytes

---

## 💡 Key Design Patterns

| Pattern | Usage | Location |
|---------|-------|----------|
| **Result Container** | Encapsulate complex outputs | `PCAResult`, `UMAPResult`, `HMMResult`, `EWMAResult`, `KLDriftResult`, `MahalanobisResult` |
| **Factory Functions** | Simplify object creation | `make_agent()`, `make_agent_pool()`, `make_env_pool()`, `train_agent_pool()` |
| **Index Mapping** | Bidirectional state↔index | `_STATE_INDEX`, `_INDEX_STATE`, `obs_to_grid()`/`grid_to_coords()` |
| **Validation** | Fail fast with clear errors | `_validate_transition_matrix()` |
| **AR-1 Correlation** | Realistic time series | `BehaviorProfile.sample()` |
| **Curriculum / potential-shaping** | RL reward design | `RewardCurriculum`, `ManifoldPotential` |
| **Sweep functions** | Hyperparameter search returning a scored DataFrame | `hyperparameter_sweep()`, `model_selection_sweep()`, `alpha_sweep()`, `window_size_sweep()`, `threshold_sweep()` |

---

## 📖 Technology Stack

| Layer | Technologies |
|-------|--------------|
| **Scientific Computing** | NumPy, SciPy, scikit-learn |
| **Data** | Pandas |
| **Manifold Learning** | UMAP-learn, scikit-learn (t-SNE, PCA) |
| **Sequence Modeling** | hmmlearn (Gaussian HMM, Baum-Welch/Viterbi) |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Notebooks** | JupyterLab, IPython |
| **Environment** | Nix (`flake.nix`, Python 3.13 + SageMath) / conda (`environment.yml`, Python 3.12) |
| **Statistics track** | R (`r/statistics`, `r/visualization`, `r/reports` — independent of the Python pipeline) |
| **Version Control** | Git |

---

## 📍 File System

```
lbsm-research/
├── src/
│   ├── simulation/        → Agent & behavior definitions (implemented)
│   ├── telemetry/         → Preprocessing, normalization, windowing (implemented)
│   ├── manifold/          → PCA, UMAP, t-SNE, trajectory geometry (implemented)
│   ├── hmm/               → Hidden state recovery (implemented)
│   ├── drift/             → EWMA / KL / Mahalanobis drift detection (implemented)
│   ├── rl/                → Q-learning + manifold-adaptation analysis (implemented)
│   ├── evaluation/        → Quality metrics (implemented)
│   ├── visualization/     → Mostly empty stubs
│   └── utils/             → All empty stubs
│
├── notebooks/
│   ├── 01_telemetry_generation.ipynb    (populated)
│   ├── 02_manifold_learning.ipynb       (populated)
│   ├── 03_hmm_inference.ipynb           (populated)
│   ├── 04_anomaly_detection.ipynb       (populated)
│   ├── 05_rl_behavioral_evolution.ipynb (populated)
│   ├── 06_manifold_visualization.ipynb  (empty)
│   └── 07_final_experiment_analysis.ipynb (empty)
│
├── experiments/            → run_*.py scripts, all currently empty
│   ├── baseline/ manifold/ drift/ rl_adaptive/
│
├── configs/
│   ├── simulation.yaml, telemetry.yaml, projection.yaml  (populated)
│   └── rl.yaml, experiments.yaml                          (empty)
│
├── data/
│   ├── raw/nb01–nb05      → .npy arrays per notebook
│   └── processed/nb01–nb05 → .csv summaries per notebook
│
├── outputs/
│   ├── figures/nb01–nb05  → generated plots
│   └── reports/nb01–nb04  → PDF writeups
│
├── paper/                  → TMLR LaTeX draft + SageMath math appendix
├── r/                       → Independent R statistics/visualization track
├── docs/
│   └── references/         → this directory
│
├── pyproject.toml, setup.cfg, requirements.txt, environment.yml, flake.nix, Makefile
```

---

## 🎓 References & Background

### Research Paper (in progress)
"Latent Behavioral State Machines: Manifold Geometry of Adaptive Agent Telemetry"
- Section outline: `paper/paper-skeleton.md` — maps directly onto the `src/` module structure (§3 simulation, §5 manifold, §6 HMM, §7 drift, §8 RL)
- Draft LaTeX source: `paper/latex-tmlr/`
- Math appendix (SageMath): `paper/math-supplementry/`

### Related Work
- Hidden Markov Models (HMM) for state inference
- Manifold learning (dimensionality reduction)
- Behavior analysis in adaptive systems
- Drift detection in time series
- Potential-based reward shaping in RL

---
