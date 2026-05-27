# LBSM Architecture

## 1 · Data flow & module interactions

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#1e1e2e", "primaryTextColor": "#ffffff", "primaryBorderColor": "#555", "lineColor": "#ffffff", "secondaryColor": "#2a2a2a", "tertiaryColor": "#2a2a2a", "background": "#0d0d0d", "mainBkg": "#2a2a2a", "nodeBorder": "#555", "clusterBkg": "#2a2a2a", "clusterBorder": "#555555", "titleColor": "#ffffff", "edgeLabelBackground": "#1a1a1a", "fontFamily": "monospace"}}}%%
flowchart TD
    classDef sim   fill:#6c3483,stroke:#c39bd3,color:#f9e4ff,stroke-width:1.5px
    classDef tel   fill:#1a5276,stroke:#5dade2,color:#d6eaf8,stroke-width:1.5px
    classDef man   fill:#0b5345,stroke:#52be80,color:#d5f5e3,stroke-width:1.5px
    classDef viz   fill:#784212,stroke:#f0b27a,color:#fdebd0,stroke-width:1.5px
    classDef eval  fill:#78281f,stroke:#ec7063,color:#fdedec,stroke-width:1.5px

    subgraph SIM["🧬  SIMULATION LAYER"]
        BP["BehaviorProfile\nμ · Σ · ρ"]:::sim
        AA["AdaptiveAgent\n4 hidden states · Markov + AR-1"]:::sim
        OPS["step() · simulate() · reset()"]:::sim
        FAC["make_agent() / make_agent_pool()"]:::sim
    end

    subgraph TEL["⚙️  TELEMETRY PROCESSING LAYER"]
        PRE["preprocessing.py\nnulls · outliers"]:::tel
        NORM["normalization.py · Z-score"]:::tel
        FEAT["feature_extraction.py\nderive features"]:::tel
        AUX["windowing.py · statistics.py"]:::tel
    end

    subgraph MAN["🔭  MANIFOLD LEARNING LAYER"]
        PCA["fit_pca()\nPCAResult · linear baseline"]:::man
        UMAP["fit_umap()\nUMAPResult · primary nonlinear"]:::man
        TSNE["fit_tsne()\nalternative"]:::man
    end

    subgraph VIZ["📊  VISUALIZATION LAYER"]
        VCORE["manifold_plots · trajectory_plots\nheatmaps · temporal_dynamics\nstate_transitions · dashboard"]:::viz
    end

    subgraph EVAL["🧪  EVALUATION LAYER"]
        SCORE["embedding_scorecard()\nsilhouette · davies-bouldin\ncalinski-harabasz · trustworthiness · continuity"]:::eval
        CMP["PCA vs UMAP\ncomparison table"]:::eval
    end

    BP --> AA --> OPS
    FAC --> AA
    OPS -->|"raw telemetry N×T×6"| TEL
    PRE --> NORM --> FEAT
    AUX --> FEAT
    FEAT -->|"feature matrix N×6"| MAN
    PCA & UMAP & TSNE -->|"2-D / 3-D embeddings"| VCORE
    PCA & UMAP & TSNE -->|"X + embedded X"| SCORE
    SCORE --> CMP
```

---

## 2 · Parallel analysis branches (planned)

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#1e1e2e", "primaryTextColor": "#ffffff", "primaryBorderColor": "#555", "lineColor": "#ffffff", "secondaryColor": "#2a2a2a", "tertiaryColor": "#2a2a2a", "background": "#0d0d0d", "mainBkg": "#2a2a2a", "nodeBorder": "#555", "clusterBkg": "#2a2a2a", "clusterBorder": "#555555", "titleColor": "#ffffff", "edgeLabelBackground": "#1a1a1a", "fontFamily": "monospace"}}}%%
flowchart LR
    classDef hmm   fill:#1a3a5c,stroke:#5dade2,color:#aed6f1,stroke-width:1.5px
    classDef drift fill:#0e3b2e,stroke:#48c9b0,color:#a2d9ce,stroke-width:1.5px
    classDef rl    fill:#4a235a,stroke:#c39bd3,color:#e8daef,stroke-width:1.5px

    subgraph HMM["🧠  HMM LAYER — planned"]
        H1["hidden_state_model.py\nForward · Backward · Viterbi"]:::hmm
        H2["sequence_inference.py\nstate recovery"]:::hmm
        H3["transition_analysis.py\nempirical transition matrix"]:::hmm
        H4["latent_state_metrics.py\nstate-space diagnostics"]:::hmm
        H1 --> H2 --> H3 --> H4
    end

    subgraph DRIFT["📉  DRIFT LAYER — planned"]
        D1["drift_detection.py\ndetection algorithms"]:::drift
        D2["regime_shift_analysis.py\ncharacterise shift events"]:::drift
        D3["kl_divergence.py · ewma.py"]:::drift
        D1 --> D2
        D3 --> D2
    end

    subgraph RL["🤖  RL LAYER — planned"]
        R1["q_learning.py\nQ-learning policy"]:::rl
        R2["policy.py · exploration.py"]:::rl
        R3["reward_tracking.py\ncumulative rewards"]:::rl
        R4["adaptation_dynamics.py\nlearning → manifold geometry"]:::rl
        R1 --> R2 --> R3 --> R4
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
    classDef planned    fill:#1a1a1a,stroke:#444,color:#888,stroke-width:1px,stroke-dasharray:5 5
    classDef infra      fill:#1c2833,stroke:#566573,color:#aab7b8,stroke-width:1px

    SIM["simulation/\nagent · profiles · environment\nreward_dynamics · telemetry_generator"]:::foundation
    TEL["telemetry/\npreprocessing · normalization\nfeature_extraction · stats · windowing"]:::pipeline
    MAN["manifold/\npca · umap_projection · tsne\nmanifold_metrics · trajectory_geometry\ncovariance_analysis"]:::analysis
    EVA["evaluation/\nmanifold_quality · clustering_metrics\ntrajectory_metrics · stability · explained_var"]:::validation
    VIZ["visualization/\nmanifold_plots · trajectory_plots\nheatmaps · temporal_dynamics\nstate_transitions · dashboard"]:::present
    HMM["hmm/  ⏳\nhidden_state_model · sequence_inference\ntransition_analysis · latent_state_metrics"]:::planned
    DRI["drift/  ⏳\ndrift_detection · regime_shift_analysis\nkl_divergence · ewma"]:::planned
    RLM["rl/  ⏳\nq_learning · policy · exploration\nreward_tracking · adaptation_dynamics"]:::planned
    UTL["utils/\nlogging · experiment_tracking · io · seed"]:::infra

    SIM -->|"raw telemetry"| TEL
    TEL -->|"feature matrix"| MAN
    MAN -->|"embeddings"| EVA
    MAN -->|"embeddings"| VIZ
    TEL -.->|"planned"| HMM
    TEL -.->|"planned"| DRI
    MAN -.->|"planned"| RLM
    UTL --> SIM & TEL & MAN & EVA
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
    participant V  as 📊 Visualiser
    participant Q  as 🧪 Evaluator

    rect rgb(60,20,80)
        Note over A,T: Simulation epoch
        A  ->> SM: current hidden state s_t
        SM ->> T:  sample x_t ~ N(μ_st, Σ_st) with AR-1
        T  ->> T:  record (agent_id, t, s_t, x_t)
        SM -->> SM: transition s_{t+1}
    end

    rect rgb(10,40,70)
        Note over P,Q: Analysis pipeline
        T  ->> P:  feature matrix X (N×6)
        P  ->> P:  normalise · handle missing data
        P  ->> E:  fit PCA, UMAP, t-SNE
        E  ->> Q:  silhouette · Davies-Bouldin · trustworthiness
        E  ->> V:  2D/3D scatter · trajectories · heatmaps
    end

    rect rgb(10,50,35)
        Note over A,Q: Planned downstream
        T  -->> SM: HMM — infer hidden state
        T  -->> P:  Drift — detect regime shifts
        E  -->> A:  RL — modify policies
        E  -->> Q:  Anomaly — outliers in manifold
    end
```

---

## 5 · Experiment pipeline

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#1e1e2e", "primaryTextColor": "#ffffff", "primaryBorderColor": "#555", "lineColor": "#ffffff", "secondaryColor": "#2a2a2a", "tertiaryColor": "#2a2a2a", "background": "#0d0d0d", "mainBkg": "#2a2a2a", "nodeBorder": "#555", "clusterBkg": "#2a2a2a", "clusterBorder": "#555555", "titleColor": "#ffffff", "edgeLabelBackground": "#1a1a1a", "fontFamily": "monospace"}}}%%
flowchart LR
    classDef entry fill:#1a3a5c,stroke:#5dade2,color:#aed6f1,stroke-width:1.5px
    classDef step  fill:#1c2833,stroke:#566573,color:#aab7b8,stroke-width:1px
    classDef out   fill:#0b5345,stroke:#52be80,color:#d5f5e3,stroke-width:1.5px
    classDef plan  fill:#1a1a1a,stroke:#444,color:#888,stroke-width:1px,stroke-dasharray:5 5

    subgraph BASE["🧪 baseline/"]
        B1["run_baseline.py"]:::entry --> B2["generate\nagents"]:::step --> B3["simulate\ntelemetry"]:::step --> B4["save CSV"]:::out
    end
    subgraph PROJ["🔭 manifold/"]
        M1["run_projection_experiment.py"]:::entry --> M2["load\ntelemetry"]:::step --> M3["PCA / UMAP\n/ t-SNE"]:::step --> M4["evaluate\nmetrics"]:::step --> M5["visualise"]:::out
    end
    subgraph DEXP["📉 drift/  ⏳"]
        D1["run_drift_experiment.py"]:::plan --> D2["detect\nregime shifts"]:::plan --> D3["analyse\ndrift"]:::plan
    end
    subgraph RLEXP["🤖 rl_adaptive/  ⏳"]
        R1["run_rl_experiment.py"]:::plan --> R2["agent\nlearning"]:::plan --> R3["track manifold\ntrajectory"]:::plan --> R4["measure\nadaptation"]:::plan
    end

    BASE -->|"CSV telemetry"| PROJ
    BASE -.->|"planned"| DEXP
    BASE -.->|"planned"| RLEXP
```

---

## 6 · Jupyter notebook pipeline

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#1e1e2e", "primaryTextColor": "#ffffff", "primaryBorderColor": "#555", "lineColor": "#ffffff", "secondaryColor": "#2a2a2a", "tertiaryColor": "#2a2a2a", "background": "#0d0d0d", "mainBkg": "#2a2a2a", "nodeBorder": "#555", "clusterBkg": "#2a2a2a", "clusterBorder": "#555555", "titleColor": "#ffffff", "edgeLabelBackground": "#1a1a1a", "fontFamily": "monospace"}}}%%
flowchart TD
    classDef active  fill:#1a5276,stroke:#5dade2,color:#d6eaf8,stroke-width:2px
    classDef planned fill:#1a1a1a,stroke:#444,color:#888,stroke-width:1px,stroke-dasharray:5 5

    N1["📓 01 · telemetry generation\nSimulation → telemetry → state dynamics viz"]:::active
    N2["📓 02 · manifold learning\nPCA / UMAP / t-SNE · manifold quality · interpretation"]:::active
    N3["📓 03 · HMM inference  ⏳\nstate sequence recovery · Viterbi decoding"]:::planned
    N4["📓 04 · anomaly detection  ⏳\noutlier detection in manifold space"]:::planned
    N5["📓 05 · RL behavioural evolution  ⏳\nlearning trajectories · adaptation dynamics"]:::planned
    N6["📓 06 · manifold visualisation  ⏳\n3-D interactive visualisations"]:::planned
    N7["📓 07 · final experiment analysis  ⏳\nintegrated cross-pipeline analysis"]:::planned

    N1 --> N2 --> N3 --> N4 --> N5 --> N6 --> N7
```
