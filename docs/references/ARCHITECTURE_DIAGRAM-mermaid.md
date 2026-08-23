# LBSM Architecture

## 1 · Data flow & module interactions

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#1e1e2e", "primaryTextColor": "#ffffff", "primaryBorderColor": "#555", "lineColor": "#ffffff", "secondaryColor": "#2a2a2a", "tertiaryColor": "#2a2a2a", "background": "#0d0d0d", "mainBkg": "#2a2a2a", "nodeBorder": "#555", "clusterBkg": "#2a2a2a", "clusterBorder": "#555555", "titleColor": "#ffffff", "edgeLabelBackground": "#1a1a1a", "fontFamily": "monospace"}}}%%
flowchart TD
    classDef sim   fill:#6c3483,stroke:#c39bd3,color:#f9e4ff,stroke-width:1.5px
    classDef tel   fill:#1a5276,stroke:#5dade2,color:#d6eaf8,stroke-width:1.5px
    classDef man   fill:#0b5345,stroke:#52be80,color:#d5f5e3,stroke-width:1.5px
    classDef viz   fill:#3a3a3a,stroke:#888,color:#ccc,stroke-width:1px,stroke-dasharray:5 5
    classDef eval  fill:#78281f,stroke:#ec7063,color:#fdedec,stroke-width:1.5px

    subgraph SIM["🧬  SIMULATION LAYER"]
        BP["BehaviorProfile\nμ · Σ · ρ"]:::sim
        AA["AdaptiveAgent\n4 hidden states · Markov + AR-1"]:::sim
        OPS["step() · simulate() · reset()"]:::sim
        FAC["make_agent() / make_agent_pool()\nTelemetryGenerator"]:::sim
    end

    subgraph TEL["⚙️  TELEMETRY PROCESSING LAYER"]
        PRE["preprocessing.py\nclip_features · drop_incomplete"]:::tel
        NORM["normalization.py · zscore/minmax"]:::tel
        FEAT["feature_extraction.py\nrolling stats · temporal_diff"]:::tel
        AUX["windowing.py · statistics.py"]:::tel
    end

    subgraph MAN["🔭  MANIFOLD LEARNING LAYER"]
        PCA["fit_pca()\nPCAResult · linear baseline"]:::man
        UMAP["fit_umap()\nUMAPResult · primary nonlinear"]:::man
        TSNE["fit_tsne()\nTSNEResult · stratified subsample"]:::man
        TRAJ["trajectory_geometry.py\nTrajectoryStats"]:::man
    end

    subgraph VIZ["📊  VISUALIZATION LAYER — mostly empty"]
        VCORE["manifold_plots.py\n(one-off script, not a library)"]:::viz
        VSTUB["trajectory_plots · heatmaps\ntemporal_dynamics · state_transitions\ndashboard  — EMPTY FILES"]:::viz
    end

    subgraph EVAL["🧪  EVALUATION LAYER"]
        SCORE["embedding_scorecard()\nsilhouette · davies-bouldin\ncalinski-harabasz · trustworthiness · continuity"]:::eval
        CMP["compare_embeddings()\nPCA vs UMAP vs t-SNE"]:::eval
    end

    BP --> AA --> OPS
    FAC --> AA
    OPS -->|"raw telemetry N×T×6"| TEL
    PRE --> NORM --> FEAT
    AUX --> FEAT
    FEAT -->|"feature matrix N×6"| MAN
    PCA & UMAP & TSNE --> TRAJ
    PCA & UMAP & TSNE -->|"2-D / 3-D embeddings"| VCORE
    PCA & UMAP & TSNE -->|"X + embedded X"| SCORE
    SCORE --> CMP
```

---

## 2 · Parallel analysis branches — all implemented

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#1e1e2e", "primaryTextColor": "#ffffff", "primaryBorderColor": "#555", "lineColor": "#ffffff", "secondaryColor": "#2a2a2a", "tertiaryColor": "#2a2a2a", "background": "#0d0d0d", "mainBkg": "#2a2a2a", "nodeBorder": "#555", "clusterBkg": "#2a2a2a", "clusterBorder": "#555555", "titleColor": "#ffffff", "edgeLabelBackground": "#1a1a1a", "fontFamily": "monospace"}}}%%
flowchart LR
    classDef hmm   fill:#1a3a5c,stroke:#5dade2,color:#aed6f1,stroke-width:1.5px
    classDef drift fill:#0e3b2e,stroke:#48c9b0,color:#a2d9ce,stroke-width:1.5px
    classDef rl    fill:#4a235a,stroke:#c39bd3,color:#e8daef,stroke-width:1.5px

    subgraph HMM["🧠  HMM LAYER"]
        H1["hidden_state_model.py\nfit_hmm(): Baum-Welch/EM + Viterbi\n→ HMMResult"]:::hmm
        H2["sequence_inference.py\nmodel_selection_sweep() BIC/AIC"]:::hmm
        H3["transition_analysis.py\ntransition_matrix_error · spectral_gap"]:::hmm
        H4["latent_state_metrics.py\nper_agent_metrics (Hungarian-aligned)"]:::hmm
        H1 --> H2 --> H3 --> H4
    end

    subgraph DRIFT["📉  DRIFT LAYER"]
        D1["drift_detection.py\nfit_mahalanobis() · combined_anomaly_score()"]:::drift
        D2["regime_shift_analysis.py\nground_truth_changepoints · detection_latency"]:::drift
        D3["ewma.py · kl_divergence.py\nfit_ewma() · fit_kl_detector()"]:::drift
        D1 --> D2
        D3 --> D2
    end

    subgraph RL["🤖  RL LAYER"]
        R1["environment.py\nBehavioralEnv MDP"]:::rl
        R2["q_learning.py\nQLearningAgent.train()"]:::rl
        R3["reward_dynamics.py\nManifoldPotential · RewardCurriculum"]:::rl
        R4["adaptation_dynamics.py\nmanifold_trajectory_stats · cluster_migration"]:::rl
        R1 --> R2
        R3 --> R2
        R2 --> R4
    end
```

---

## 3 · Module dependency graph

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#1e1e2e", "primaryTextColor": "#ffffff", "primaryBorderColor": "#555", "lineColor": "#ffffff", "secondaryColor": "#2a2a2a", "tertiaryColor": "#2a2a2a", "background": "#0d0d0d", "mainBkg": "#2a2a2a", "nodeBorder": "#555", "clusterBkg": "#2a2a2a", "clusterBorder": "#555555", "titleColor": "#ffffff", "edgeLabelBackground": "#1a1a1a", "fontFamily": "monospace"}}}%%
graph TD
    classDef foundation fill:#5b2c6f,stroke:#a569bd,color:#e8daef,stroke-width:2px
    classDef pipeline   fill:#1a5276,stroke:#5dade2,color:#d6eaf8,stroke-width:1.5px
    classDef analysis   fill:#0b5345,stroke:#52be80,color:#d5f5e3,stroke-width:1.5px
    classDef validation fill:#7d6608,stroke:#f9e79f,color:#fef9e7,stroke-width:1.5px
    classDef present    fill:#78281f,stroke:#ec7063,color:#fdedec,stroke-width:1.5px
    classDef empty      fill:#1a1a1a,stroke:#444,color:#888,stroke-width:1px,stroke-dasharray:5 5
    classDef infra      fill:#1c2833,stroke:#566573,color:#aab7b8,stroke-width:1px

    SIM["simulation/\nagent · profiles · telemetry_generator"]:::foundation
    TEL["telemetry/\npreprocessing · normalization\nfeature_extraction · stats · windowing"]:::pipeline
    MAN["manifold/\npca · umap_projection · tsne\nmanifold_metrics · trajectory_geometry\n(covariance_analysis.py EMPTY)"]:::analysis
    EVA["evaluation/\nmanifold_quality · clustering_metrics\ntrajectory_metrics · stability · explained_var"]:::validation
    VIZ["visualization/  ⚠️ mostly empty\nmanifold_plots (one-off script)\n5 other files EMPTY"]:::empty
    HMM["hmm/\nhidden_state_model · sequence_inference\ntransition_analysis · latent_state_metrics"]:::analysis
    DRI["drift/\ndrift_detection · regime_shift_analysis\nkl_divergence · ewma"]:::analysis
    RLM["rl/\nenvironment · q_learning · policy\nexploration · reward_dynamics\nreward_tracking · adaptation_dynamics"]:::analysis
    UTL["utils/  ⚠️ all empty\nlogging · experiment_tracking · io · seed"]:::empty

    SIM -->|"raw telemetry"| TEL
    TEL -->|"feature matrix"| MAN
    MAN -->|"embeddings"| EVA
    MAN -.->|"one-off script"| VIZ
    TEL --> HMM
    TEL --> DRI
    MAN --> RLM
    HMM --> DRI
    RLM --> EVA
    UTL -.->|"not actually used"| SIM & TEL & MAN & EVA
```

---

## 4 · Temporal execution flow

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#1e1e2e", "primaryTextColor": "#ffffff", "primaryBorderColor": "#555", "lineColor": "#ffffff", "secondaryColor": "#2a2a2a", "tertiaryColor": "#2a2a2a", "background": "#0d0d0d", "mainBkg": "#2a2a2a", "nodeBorder": "#555", "clusterBkg": "#2a2a2a", "clusterBorder": "#555555", "titleColor": "#ffffff", "edgeLabelBackground": "#1a1a1a", "actorBkg": "#2a2a2a", "actorBorder": "#555", "actorTextColor": "#ffffff", "actorLineColor": "#ffffff", "signalColor": "#ffffff", "signalTextColor": "#ffffff", "labelBoxBkgColor": "#2a2a2a", "labelBoxBorderColor": "#555", "labelTextColor": "#ffffff", "loopTextColor": "#ffffff", "noteBkgColor": "#2a2a2a", "noteBorderColor": "#555", "noteTextColor": "#ffffff", "activationBkgColor": "#3a3a3a", "sequenceNumberColor": "#0d0d0d", "fontFamily": "monospace"}}}%%
sequenceDiagram
    autonumber
    participant A  as 🧬 Agent
    participant SM as 🔀 State Machine
    participant T  as 📡 Telemetry Store
    participant P  as ⚙️ Preprocessor
    participant E  as 🔭 Embedder
    participant H  as 🧠 HMM
    participant D  as 📉 Drift
    participant R  as 🤖 RL
    participant Q  as 🧪 Evaluator

    rect rgb(60,20,80)
        Note over A,T: Simulation epoch
        A  ->> SM: current hidden state s_t
        SM ->> T:  sample x_t ~ N(μ_st, Σ_st) with AR-1
        T  ->> T:  record (agent_id, t, s_t, x_t)
        SM -->> SM: transition s_{t+1}
    end

    rect rgb(10,40,70)
        Note over P,Q: Analysis pipeline (nb01-02)
        T  ->> P:  feature matrix X (N×6)
        P  ->> P:  normalise · window · handle missing data
        P  ->> E:  fit PCA, UMAP, t-SNE
        E  ->> Q:  silhouette · Davies-Bouldin · trustworthiness
    end

    rect rgb(10,50,35)
        Note over T,R: Downstream branches (nb03-05) — all implemented
        T  ->> H:  fit_hmm() — recover hidden state sequence
        H  ->> Q:  ARI · per-agent accuracy vs ground truth
        T  ->> D:  EWMA + KL + Mahalanobis anomaly scoring
        H  ->> D:  ground-truth changepoints for latency evaluation
        E  ->> R:  BehavioralEnv trains against manifold-derived reward shaping
        R  ->> Q:  learning curves · manifold trajectory stats
    end
```

---

## 5 · Experiment pipeline — scripts are placeholders (empty)

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#1e1e2e", "primaryTextColor": "#ffffff", "primaryBorderColor": "#555", "lineColor": "#ffffff", "secondaryColor": "#2a2a2a", "tertiaryColor": "#2a2a2a", "background": "#0d0d0d", "mainBkg": "#2a2a2a", "nodeBorder": "#555", "clusterBkg": "#2a2a2a", "clusterBorder": "#555555", "titleColor": "#ffffff", "edgeLabelBackground": "#1a1a1a", "fontFamily": "monospace"}}}%%
flowchart LR
    classDef entry fill:#1a1a1a,stroke:#444,color:#888,stroke-width:1px,stroke-dasharray:5 5
    classDef step  fill:#1a1a1a,stroke:#444,color:#888,stroke-width:1px,stroke-dasharray:5 5
    classDef out   fill:#1a1a1a,stroke:#444,color:#888,stroke-width:1px,stroke-dasharray:5 5
    classDef notebook fill:#0b5345,stroke:#52be80,color:#d5f5e3,stroke-width:1.5px

    subgraph BASE["🧪 baseline/  — run_baseline.py: 0 lines"]
        B1["intended:\ngenerate agents"]:::entry --> B2["simulate\ntelemetry"]:::step --> B3["save CSV"]:::out
    end
    subgraph PROJ["🔭 manifold/  — run_projection_experiment.py: 0 lines"]
        M1["intended:\nload telemetry"]:::entry --> M2["PCA / UMAP\n/ t-SNE"]:::step --> M3["evaluate\nmetrics"]:::out
    end
    subgraph DEXP["📉 drift/  — run_drift_experiment.py: 0 lines"]
        D1["intended:\ndetect regime shifts"]:::entry --> D2["analyse\ndrift"]:::out
    end
    subgraph RLEXP["🤖 rl_adaptive/  — run_rl_experiment.py: 0 lines"]
        R1["intended:\nagent learning"]:::entry --> R2["track manifold\ntrajectory"]:::out
    end

    REAL["✅ Real pipelines actually run here:\nnotebooks/01-05_*.ipynb"]:::notebook
    BASE -.->|"not wired up"| REAL
    PROJ -.->|"not wired up"| REAL
    DEXP -.->|"not wired up"| REAL
    RLEXP -.->|"not wired up"| REAL
```

---

## 6 · Jupyter notebook pipeline

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#1e1e2e", "primaryTextColor": "#ffffff", "primaryBorderColor": "#555", "lineColor": "#ffffff", "secondaryColor": "#2a2a2a", "tertiaryColor": "#2a2a2a", "background": "#0d0d0d", "mainBkg": "#2a2a2a", "nodeBorder": "#555", "clusterBkg": "#2a2a2a", "clusterBorder": "#555555", "titleColor": "#ffffff", "edgeLabelBackground": "#1a1a1a", "fontFamily": "monospace"}}}%%
flowchart TD
    classDef active  fill:#1a5276,stroke:#5dade2,color:#d6eaf8,stroke-width:2px
    classDef empty fill:#1a1a1a,stroke:#444,color:#888,stroke-width:1px,stroke-dasharray:5 5

    N1["📓 01 · telemetry generation\nSimulation → telemetry → state dynamics viz"]:::active
    N2["📓 02 · manifold learning\nPCA / UMAP / t-SNE · manifold quality · interpretation"]:::active
    N3["📓 03 · HMM inference\nGaussian HMM (Baum-Welch/Viterbi) · BIC selection · GT comparison"]:::active
    N4["📓 04 · anomaly detection\nEWMA / KL / Mahalanobis · threshold sweeps · detection latency"]:::active
    N5["📓 05 · RL behavioural evolution\nQ-learning · learning curves · manifold/cluster migration"]:::active
    N6["📓 06 · manifold visualisation  — EMPTY, not started"]:::empty
    N7["📓 07 · final experiment analysis  — EMPTY, not started"]:::empty

    N1 --> N2 --> N3 --> N4 --> N5 --> N6 --> N7
```
