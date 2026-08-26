# data/

Datasets for the LBSM pipeline, split by processing stage:

- **`raw/`** — per-notebook intermediate arrays (`.npy`), the direct output of each notebook's
  own computation (Q-tables, HMM posteriors, UMAP/t-SNE embeddings, anomaly scores, robustness-grid
  checkpoints). See `raw/readme.md`.
- **`processed/`** — per-notebook tabular exports (`.csv`), the cleaned/summarised results used
  in reports and downstream notebooks. See `processed/readme.md`.
- **`exports/`** — cross-format re-exports (`csv/`, `parquet/`, `r_inputs/`) of processed tables
  for the independent R statistics track (`r/`). Git-ignored — regenerate from `processed/` rather
  than expecting these to be checked in.

Each subdirectory is further split by notebook (`nb01`–`nb07`), matching the `notebooks/` and
`outputs/` directory layout. `data/raw/nb07/` and all `*.npz` files are also git-ignored (the
full-scale robustness grid's checkpoint files run into the thousands and are regenerable by
re-running `notebooks/07_final_experiment_analysis.ipynb`).
