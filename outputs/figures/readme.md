# outputs/figures/

Generated plots, one directory per notebook.

| Directory | Files | Notes |
|-----------|------:|-------|
| `nb01/` | 7 | `fig01_state_frequencies.png` ... `fig07_pairplot.png` — no `nb01` infix, unlike later notebooks |
| `nb02/` | 12 | `fig_nb02_01_pca_scree.png` ... `fig_nb02_12_method_agreement.png` |
| `nb03/` | 11 | `fig_nb03_01_model_selection.png` ... `fig_nb03_10b_entropy_umap3d.png` |
| `nb04/` | 5 | `fig_nb04_01_mah_distributions.png` ... `fig_nb04_05_shift_magnitude.png` |
| `nb05/` | 8 | `fig_nb05_01_random_trajectory.png` ... `fig_nb05_08_cluster_migration.png` |
| `nb06/` | 8 | `fig_nb06_01_trajectory3d_by_phase.png` ... `fig_nb06_08_multi_agent_phase.png` |
| `nb07/` | 12 | `fig_nb07_01_full_vs_diag_ari.png`, `..._02_...`, `..._03_...`, plus `..._04/05/06_state_identity_*_K{4,5,6}.png` (3 K-values × 3 view types) |

Each file's exact purpose is documented in that notebook's own report (Export Artefacts / File
Manifest section) and consolidated in `outputs/reports/lbsm_unified_report.pdf`'s Appendix A.
Nothing here is git-ignored — regenerate by re-running the corresponding notebook if a figure
looks stale relative to `data/processed/`.
