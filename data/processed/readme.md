# data/processed/

Per-notebook tabular exports (`.csv`), cleaned and summarised from `data/raw/`'s arrays — these are
the files cited in each notebook's report and read back by downstream notebooks/analysis.

| Directory | Contents |
|-----------|----------|
| `nb01/` | `telemetry_n20_t2000.csv` — the canonical base dataset (20 agents × 2,000 timesteps, 6 features), regenerated with a fixed seed at the start of most later notebooks too |
| `nb02/` | `trajectory_stats.csv`, `transition_coords.csv` |
| `nb03/` | `bic_sweep.csv`, `hmm_agent_metrics.csv`, `hmm_regime_accuracy.csv` |
| `nb04/` | `threshold_sweep.csv`, `detection_latency.csv`, `detector_stability.csv`, `shift_magnitude_summary.csv` |
| `nb05/` | `rl_training_log.csv`, `pool_learning_curves.csv`, `regime_dwell_summary.csv`, `convergence_table.csv`, `policy_summary.csv`, `anomaly_score_evolution.csv`, `transition_entropy.csv`, `cluster_migration.csv` |
| `nb06/` | `rl_trajectory_embedded.csv`, `phase_manifold_distance_summary.csv`, `kde_phase_statistics.csv`, `action_density_correspondence.csv` |
| `nb07/` | `robustness_grid_results.csv` (160-configuration full-scale grid), `full_scale_attempt_log.csv` (4,800-row per-attempt audit trail), `grid_vs_reliable_summary.csv`, `numerical_health_by_config.csv`, `stationary_distribution_mae.csv`, `umap_space_bic_recovery.csv`, `hmm_state_identity_diagnostic.csv` |

All files here are tracked in git. Each notebook's own report (`outputs/reports/nb0X/`) documents
the exact shape, provenance, and purpose of every file in its own Export Artefacts section; the
unified report (`outputs/reports/lbsm_unified_report.pdf`) collects all of these into one appendix.
