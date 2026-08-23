"""
run_projection_experiment.py
=============================
Standalone manifold-learning experiment: fit PCA/UMAP/t-SNE on the baseline
telemetry and score embedding quality, without going through
notebooks/02_manifold_learning.ipynb.

Requires experiments/baseline/run_baseline.py to have been run first (or the
telemetry CSV to already exist at the configured path).

Usage
-----
    python experiments/manifold/run_projection_experiment.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.simulation.behavior_profiles import TELEMETRY_FEATURES
from src.telemetry import zscore_matrix
from src.manifold import fit_pca, fit_umap, fit_tsne, embedding_scorecard, compare_embeddings
from src.utils import get_logger, save_dataframe, track_run

log = get_logger(__name__)

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(cfg: dict) -> pd.DataFrame:
    telemetry_path = REPO_ROOT / cfg["input"]["telemetry_csv"]
    if not telemetry_path.exists():
        raise FileNotFoundError(
            f"{telemetry_path} not found — run experiments/baseline/run_baseline.py first."
        )
    df = pd.read_csv(telemetry_path)
    feature_cols = list(TELEMETRY_FEATURES)
    X_raw = df[feature_cols].values.astype(float)
    labels = df["hidden_state"].values
    X, _ = zscore_matrix(X_raw)
    log.info("Loaded %d observations, %d features", *X.shape)

    pca_cfg, umap_cfg, tsne_cfg = cfg["pca"], cfg["umap"], cfg["tsne"]

    pca_result = fit_pca(X, feature_cols, n_components=pca_cfg["n_components"],
                          random_state=pca_cfg["random_state"])
    umap_result = fit_umap(X, n_components=umap_cfg["n_components"],
                            n_neighbors=umap_cfg["n_neighbors"], min_dist=umap_cfg["min_dist"],
                            random_state=umap_cfg["random_state"])
    tsne_result = fit_tsne(X, labels, perplexity=tsne_cfg["perplexity"],
                            random_state=tsne_cfg["random_state"])

    scorecards = [
        embedding_scorecard(X, pca_result.embedding[:, :2], labels, "PCA"),
        embedding_scorecard(X, umap_result.embedding, labels, "UMAP"),
        embedding_scorecard(X[tsne_result.sample_idx], tsne_result.embedding,
                             labels[tsne_result.sample_idx], "t-SNE"),
    ]
    comparison = compare_embeddings(scorecards)

    out_path = REPO_ROOT / cfg["output"]["scorecard_csv"]
    save_dataframe(comparison.reset_index(), out_path)
    log.info("Saved embedding scorecard: %s", out_path)

    print("=== Manifold projection experiment complete ===")
    print(comparison.to_string())
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = load_config(args.config)
    with track_run("manifold", cfg) as record:
        run(cfg)
        record.add_output(cfg["output"]["scorecard_csv"])


if __name__ == "__main__":
    main()
