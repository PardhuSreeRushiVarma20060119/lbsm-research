"""
make_regime_centroids.py
=========================
Compute per-regime centroids in z-scored telemetry feature space and save
them to ``data/raw/nb02/regime_centroids_z.npy``.

This array is the HMM-space (raw 6-feature, not UMAP-embedding) analogue of
:func:`src.manifold.pca.regime_centroids_pca` — one centroid per ground-truth
regime, in the same z-scored feature space the HMM itself operates in. It
did not previously exist anywhere in the pipeline; NB02 only ever saved
embedding coordinates (UMAP/t-SNE), not raw-feature centroids.

Written to support the KMeans/centroid-warmed HMM initialisation strategy
described in outputs/reports/issues/lbsm_covariance_issue.pdf (Listing 9),
planned for Notebook 07's full-covariance robustness sweep.

Usage
-----
    python scripts/make_regime_centroids.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.simulation.behavior_profiles import TELEMETRY_FEATURES, PROFILE_NAMES
from src.telemetry import zscore_matrix

TELEMETRY_CSV = REPO_ROOT / "data" / "processed" / "nb01" / "telemetry_n20_t2000.csv"
OUT_PATH = REPO_ROOT / "data" / "raw" / "nb02" / "regime_centroids_z.npy"


def main() -> None:
    if not TELEMETRY_CSV.exists():
        raise FileNotFoundError(
            f"{TELEMETRY_CSV} not found — run experiments/baseline/run_baseline.py first."
        )

    df = pd.read_csv(TELEMETRY_CSV)
    feature_cols = list(TELEMETRY_FEATURES)
    X_raw = df[feature_cols].values.astype(float)
    X_z, _ = zscore_matrix(X_raw)

    centroids = np.zeros((len(PROFILE_NAMES), len(feature_cols)))
    for i, regime in enumerate(PROFILE_NAMES):
        mask = (df["hidden_state"] == regime).values
        centroids[i] = X_z[mask].mean(axis=0)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_PATH, centroids)

    print(f"Saved {OUT_PATH}  shape={centroids.shape}")
    print(f"Regime order (rows): {PROFILE_NAMES}")
    print(f"Feature order (cols): {feature_cols}")
    print()
    print(pd.DataFrame(centroids, index=PROFILE_NAMES, columns=feature_cols).round(3).to_string())


if __name__ == "__main__":
    main()
