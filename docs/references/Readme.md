# LBSM Research Ref - Quick Reference Index

## 📚 Documentation Files In This Directory :

### 1. **[CODEBASE_UNDERSTANDING](./CODEBASE_UNDERSTANDING.md)** (13.3 KB)
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

### 2. **[ARCHITECTURE_DIAGRAM](./ARCHITECTURE_DIAGRAM.md)** (13 KB)
   - **Visual representations** of the system:
     - Data Flow Diagram (simulation → processing → analysis)
     - Module Dependency Graph
     - Temporal Execution Flow
     - Experiment Pipeline Overview
   - **Key diagrams**:
     - End-to-end pipeline (3 main layers)
     - Parallel analysis branches (HMM, Drift, RL)
     - Data structures illustration
   - **Best for**: Understanding system structure, data dependencies

### 3. **[MODULE_REFERENCE](./MODULE_REFERENCE.md)** (18.3 KB)
   - **Detailed function catalog** for every module:
     - Class definitions with all methods
     - Function signatures with parameter details
     - Return type documentation
     - Expected functions (not yet implemented)
     - Configuration file schemas
   - **Coverage**:
     - 9 implemented modules (src/)
     - 4 planned modules (HMM, Drift, RL, Evaluation)
     - Factory functions and result containers
   - **Best for**: API reference, looking up specific functions

### 4. **[CODE_PATTERNS_DETAILS](./CODE_PATTERNS_DETAILS.md)** (17 KB)
   - **Implementation deep-dives**:
     - Design patterns used throughout codebase
     - Statistical concepts & formulas
     - Data flow in key functions
     - Error handling & validation
     - Type hints & documentation style
     - Testing patterns (planned)
   - **Topics covered**:
     - Result container pattern (dataclasses)
     - Factory functions
     - AR-1 temporal correlation
     - Hidden state management
     - Index mapping strategies
   - **Best for**: Code style, patterns, implementation details

---

## 🎯 Quick Facts

| Metric | Value |
|--------|-------|
| **Repository** | PardhuSreeRushiVarma20060119/lbsm-research |
| **Main Branch** | main |
| **Total Size** | ~36 MB (mostly notebooks) |
| **Code Size** | ~2,191 lines Python |
| **Modules** | 51 Python files |
| **Functions** | ~58 definitions |
| **Test Files** | 6 (all empty) |
| **Notebooks** | 7 (2 populated, 5 planned) |
| **Config Files** | 5 YAML files |

---

## 🏗️ Architecture at a Glance

```
Simulation Layer (agent.py)
        ↓
Telemetry Processing (preprocessing, normalization)
        ↓
Manifold Learning (PCA, UMAP, t-SNE)
        ↓
┌─────────────────────────────────────┐
│  Visualization (plots, dashboard)   │
│  Evaluation (metrics, comparison)   │
└─────────────────────────────────────┘
        ↓
Parallel Branches (HMM, Drift, RL) — PLANNED
```

---

## 📦 Key Classes & Data Structures

| Class | Module | Purpose |
|-------|--------|---------|
| `AdaptiveAgent` | simulation/agent.py | Core simulator with hidden Markov chain |
| `BehaviorProfile` | simulation/behavior_profiles.py | Statistical profile of each regime |
| `PCAResult` | manifold/pca.py | Container for PCA analysis results |
| `UMAPResult` | manifold/umap_projection.py | Container for UMAP embedding |

---

## 🔑 Key Functions by Category

### Simulation
- `make_agent()` — Create single agent
- `make_agent_pool()` — Create multiple agents
- `AdaptiveAgent.step()` — Run one timestep
- `AdaptiveAgent.simulate()` — Run n timesteps

### Manifold Learning
- `fit_pca()` — PCA embedding
- `fit_umap()` — UMAP embedding
- `hyperparameter_sweep()` — Grid search for UMAP
- `regime_connectivity()` — Boundary porosity metric

### Evaluation
- `embedding_scorecard()` — Full quality metrics
- `compare_embeddings()` — Build comparison table

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
- **Trustworthiness** [0, 1]: Neighborhood preservation (↑ better)
- **Continuity** [0, 1]: Inverse trustworthiness (↑ better)

---

## 🔄 Research Hypothesis

**Core Claim**: Adaptive behavioral systems generate telemetry that occupies **structured low-dimensional manifolds** rather than arbitrary high-dimensional space.

**Evidence Sought**:
- High manifold quality metrics (silhouette > 0.5, trustworthiness > 0.7)
- Clear cluster separation (regimes distinguish in embedding)
- Successful state recovery (HMM inference accuracy)
- Stable manifold geometry (consistent across methods)

---

## 🚀 Current Status

### ✅ Complete
- Simulation engine (ground-truth hidden Markov chain)
- Telemetry generation (6-D feature vectors with AR-1 correlation)
- Manifold learning (PCA, UMAP, t-SNE implementations)
- Jupyter notebooks (Nb01: Simulation, Nb02: Manifold Learning)
- Evaluation metrics (silhouette, Davies-Bouldin, trustworthiness)

### 🔄 In Progress
- Test coverage (empty test files)
- Documentation completeness

### 📋 Planned
- HMM inference (recover hidden state sequence)
- Drift detection (regime shift analysis)
- RL integration (behavioral adaptation dynamics)
- Anomaly detection (outliers in manifold)
- Advanced visualization (3D interactive dashboards)

---

## 💡 Key Design Patterns

| Pattern | Usage | Location |
|---------|-------|----------|
| **Result Container** | Encapsulate complex outputs | PCAResult, UMAPResult |
| **Factory Functions** | Simplify object creation | make_agent(), make_agent_pool() |
| **Index Mapping** | Bidirectional state↔index | _STATE_INDEX, _INDEX_STATE |
| **Validation** | Fail fast with clear errors | _validate_transition_matrix() |
| **AR-1 Correlation** | Realistic time series | BehaviorProfile.sample() |

---

## 📖 Technology Stack

| Layer | Technologies |
|-------|--------------|
| **Scientific Computing** | NumPy, SciPy, scikit-learn |
| **Data** | Pandas |
| **Manifold Learning** | UMAP-learn, scikit-learn (t-SNE, PCA) |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Notebooks** | JupyterLab, IPython |
| **Environment** | Nix (flake.nix for reproducibility) |
| **Version Control** | Git |

---

## 📍 File System

```
lbsm-research/
├── src/
│   ├── simulation/        → Agent & behavior definitions
│   ├── manifold/         → PCA, UMAP, t-SNE
│   ├── telemetry/        → Data preprocessing pipeline
│   ├── evaluation/        → Quality metrics
│   ├── visualization/     → Plotting & dashboards
│   ├── hmm/              → HMM inference (planned)
│   ├── drift/            → Drift detection (planned)
│   ├── rl/               → RL dynamics (planned)
│   └── utils/            → Infrastructure
│
├── notebooks/
│   ├── 01_telemetry_generation.ipynb    (2.5 MB)
│   ├── 02_manifold_learning.ipynb       (4.1 MB)
│   ├── 03_hmm_inference.ipynb           (empty)
│   ├── 04_anomaly_detection.ipynb       (empty)
│   ├── 05_rl_behavioral_evolution.ipynb (empty)
│   ├── 06_manifold_visualization.ipynb  (empty)
│   └── 07_final_experiment_analysis.ipynb (empty)
│
├── experiments/
│   ├── baseline/         → Baseline simulation run
│   ├── manifold/         → Manifold learning pipeline
│   ├── drift/            → Drift detection experiments
│   └── rl_adaptive/      → RL-driven adaptation
│
├── configs/
│   ├── simulation.yaml
│   ├── projection.yaml
│   ├── telemetry.yaml
│   ├── rl.yaml
│   └── experiments.yaml
│
├── data/
│   └── processed/
│       ├── nb01/telemetry_n20_t2000.csv
│       └── nb02/{trajectory_stats.csv, transition_coords.csv}
│
├── docs/
│   ├── obsidian-notes/   → Research notes (Obsidian vault)
│   ├── lbsm-math/        → Mathematical derivations
│   └── NotebookAnalysis/ → Notebook analysis PDFs
│
├── pyproject.toml
├── setup.cfg
├── requirements.txt
├── environment.yml
├── flake.nix             → Nix environment config
└── Makefile
```

---

## 🎓 References & Background

### Research Paper (Still in progress)
"Latent Behavioral State Machines: Manifold Geometry of Adaptive Agent Telemetry"
- Extensively referenced in docstrings
- Sections cited: 3.1 (regimes), 3.2 (agent dynamics), 5.1 (PCA), 5.2 (UMAP), 5.4 (metrics)

### Related Work
- Hidden Markov Models (HMM) for state inference
- Manifold learning (dimensionality reduction)
- Behavior analysis in adaptive systems
- Drift detection in time series

---

##  📋 Ongoing Work

### Code Status
- ✅ Simulation: Production-ready
- ✅ Manifold Learning: Production-ready
- 📋 Testing: Empty test files (needs coverage - pending)
- 📋 HMM: Interfaces defined, implementation pending
- 📋 Drift: Interfaces defined, implementation pending
- 📋 RL: Interfaces defined, implementation pending

### Scalability
- Current: 20 agents × 2,000 timesteps (manageable)
- Challenge: N > 100K agents or T > 100K timesteps (memory/compute)

### Analysis Depth
- Visual manifold quality: ✅ Complete
- Statistical validation: ⚠️ Limited (needs more rigor - ongoing)
- Temporal dynamics: 📋 Planned (drift detection)
- Adaptive learning: 📋 Planned (RL integration)

---
