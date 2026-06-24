"""src.rl — LBSM Reinforcement Learning Layer.

This package implements tabular Q-learning over the discrete LBSM behavioral
environment characterised in NB01–NB04, and provides analysis tools for
tracking how RL training reshapes agent trajectories in the latent manifold.

Sub-modules
-----------
environment        — BehavioralEnv (step/reset), state discretisation, action definitions
q_learning         — QLearningAgent, QLearningConfig, train_agent_pool
policy             — Policy extraction, action heatmaps, policy entropy
exploration        — EpsilonSchedule, CuriosityBonus
reward_dynamics    — ManifoldPotential, RewardCurriculum, reward decomposition
reward_tracking    — Learning curves, convergence diagnostics, regime dwell analysis
adaptation_dynamics— Manifold trajectory stats, cluster migration, anomaly evolution

Public API (star imports)
-------------------------
"""

from .environment import (
    BehavioralEnv,
    BehavioralEnv as Env,
    StepResult,
    make_env_pool,
    obs_to_grid,
    grid_to_coords,
    N_STATES,
    N_ACTIONS,
    N_GRID_LATENCY,
    N_GRID_ENTROPY,
    ACTION_PUSH_STABLE,
    ACTION_PUSH_EXPLORATORY,
    ACTION_DO_NOTHING,
    DELTA_BASE,
    N_STEPS_PER_EPISODE,
)

from .q_learning import (
    QLearningAgent,
    QLearningConfig,
    EpisodeStats,
    train_agent_pool,
)

from .policy import (
    greedy_policy,
    policy_action_grid,
    value_grid,
    policy_entropy,
    policy_entropy_grid,
    action_frequency_from_trajectory,
    action_state_heatmap,
    policy_agreement,
    policy_summary_table,
    ACTION_NAMES,
)

from .exploration import (
    EpsilonSchedule,
    CuriosityBonus,
    geometric_epsilon,
    linear_epsilon,
    exploration_coverage,
    state_coverage,
)

from .reward_dynamics import (
    ManifoldPotential,
    RewardCurriculum,
    decompose_episode_rewards,
    reward_by_regime,
)

from .reward_tracking import (
    smooth,
    learning_curve_df,
    pool_learning_curves,
    convergence_episode,
    convergence_table,
    dwell_evolution_df,
    regime_delta_table,
    pool_regime_summary,
)

from .adaptation_dynamics import (
    manifold_trajectory_stats,
    cluster_migration_table,
    anomaly_score_evolution,
    transition_entropy_series,
    umap_episode_centroids,
    regime_novelty_score,
)

__all__ = [
    # environment
    "BehavioralEnv", "Env", "StepResult", "make_env_pool",
    "obs_to_grid", "grid_to_coords",
    "N_STATES", "N_ACTIONS", "N_GRID_LATENCY", "N_GRID_ENTROPY",
    "ACTION_PUSH_STABLE", "ACTION_PUSH_EXPLORATORY", "ACTION_DO_NOTHING",
    "DELTA_BASE", "N_STEPS_PER_EPISODE",
    # q_learning
    "QLearningAgent", "QLearningConfig", "EpisodeStats", "train_agent_pool",
    # policy
    "greedy_policy", "policy_action_grid", "value_grid",
    "policy_entropy", "policy_entropy_grid",
    "action_frequency_from_trajectory", "action_state_heatmap",
    "policy_agreement", "policy_summary_table", "ACTION_NAMES",
    # exploration
    "EpsilonSchedule", "CuriosityBonus",
    "geometric_epsilon", "linear_epsilon",
    "exploration_coverage", "state_coverage",
    # reward_dynamics
    "ManifoldPotential", "RewardCurriculum",
    "decompose_episode_rewards", "reward_by_regime",
    # reward_tracking
    "smooth", "learning_curve_df", "pool_learning_curves",
    "convergence_episode", "convergence_table",
    "dwell_evolution_df", "regime_delta_table", "pool_regime_summary",
    # adaptation_dynamics
    "manifold_trajectory_stats", "cluster_migration_table",
    "anomaly_score_evolution", "transition_entropy_series",
    "umap_episode_centroids", "regime_novelty_score",
]
