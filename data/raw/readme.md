# data/raw/

Per-notebook intermediate arrays (`.npy`), the direct output of each notebook's own computation —
not yet cleaned into the tabular form used by reports (that's `data/processed/`).

| Directory | Contents |
|-----------|----------|
| `nb01/` | `X_telemetry.npy` (z-scored feature matrix), `y_labels.npy` (integer regime labels) |
| `nb02/` | `X_umap2.npy`, `X_umap3.npy`, `X_tsne.npy`, `idx_tsne.npy` (embeddings + subsample index), `regime_centroids_z.npy`, `y_labels.npy` |
| `nb03/` | `hmm_posteriors.npy`, `hmm_pred_aligned.npy`, `hmm_entropy.npy`, `y_gt_sorted.npy` |
| `nb04/` | `mah_scores.npy`, `ewma_scores.npy`, `kl_scores.npy`, `composite_scores.npy`, `composite_flags.npy`, `y_anom.npy` |
| `nb05/` | `q_tables.npy` — final Q-table per trained agent |
| `nb06/` | (no raw array exports; NB06 works directly from NB02/NB05's arrays and writes only to `data/processed/nb06/`) |
| `nb07/` | Parallel/checkpointed robustness-grid state: `checkpoints/`, `checkpoints_full/`, `bic_full_checkpoints/`, `bic_full_checkpoints_full/`, `attempt_log_full/` — one JSON file per `(N, T, seed)` configuration and per fitting attempt, written incrementally so an interrupted grid run resumes instead of restarting. Git-ignored (regenerable by re-running `notebooks/07_final_experiment_analysis.ipynb`; the full-scale grid produces thousands of files). |

`nb01`–`nb06`'s `.npy` arrays are tracked in git; only `*.npz` files and `data/raw/nb07/` (the
checkpoint directories above) are git-ignored project-wide — regenerate those by re-running the
corresponding notebook rather than expecting them to be checked in.
