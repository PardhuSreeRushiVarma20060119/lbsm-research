"""
run_drift_experiment.py
========================
Standalone drift-detection experiment: fit the healthy envelope and score
EWMA / KL-divergence / Mahalanobis detectors against the ``is_anomaly``
ground truth, without going through notebooks/04_anomaly_detection.ipynb.

Requires experiments/baseline/run_baseline.py to have been run first (or the
telemetry CSV to already exist at the configured path).

Usage
-----
    python experiments/drift/run_drift_experiment.py
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
from src.telemetry import temporal_train_test_split
from src.drift import (
    fit_healthy_envelope, mahalanobis_scores, fit_mahalanobis,
    ewma_all_agents, fit_reference, kl_all_agents, threshold_sweep,
)
from src.utils import get_logger, save_array, save_dataframe, track_run

log = get_logger(__name__)

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(cfg: dict) -> None:
    telemetry_path = REPO_ROOT / cfg["input"]["telemetry_csv"]
    if not telemetry_path.exists():
        raise FileNotFoundError(
            f"{telemetry_path} not found — run experiments/baseline/run_baseline.py first."
        )
    df = pd.read_csv(telemetry_path).sort_values(["agent_id", "timestep"]).reset_index(drop=True)
    feature_cols = list(TELEMETRY_FEATURES)

    env_cfg = cfg["healthy_envelope"]
    df_train, df_test = temporal_train_test_split(df, test_frac=env_cfg["test_frac"])
    envelope = fit_healthy_envelope(
        df=df_train, feature_cols=feature_cols, regime_col="hidden_state",
        healthy_regimes=tuple(env_cfg["healthy_regimes"]), regularize=env_cfg["regularize"],
    )
    log.info("Healthy envelope fitted on %d training observations", len(df_train))

    X_all = df[feature_cols].values.astype(float)
    healthy_mask = df["hidden_state"].isin(env_cfg["healthy_regimes"]).values

    mah_scores = mahalanobis_scores(X_all, envelope)
    mah_cfg = cfg["mahalanobis"]
    mah_result = fit_mahalanobis(X_all, envelope, threshold_pct=mah_cfg["threshold_pct"],
                                  y_healthy=healthy_mask)

    ewma_cfg = cfg["ewma"]
    ewma_df = ewma_all_agents(
        df, feature_cols, alpha=ewma_cfg["alpha"],
        threshold_k=ewma_cfg["threshold_k"], warmup=ewma_cfg["warmup"],
    )

    mu_ref, var_ref = fit_reference(X_all[healthy_mask])
    kl_cfg = cfg["kl_divergence"]
    kl_df = kl_all_agents(
        df, feature_cols, mu_ref, var_ref,
        window_size=kl_cfg["window_size"], step=kl_cfg["step"],
    )

    y_anom = df["is_anomaly"].values.astype(int) if "is_anomaly" in df.columns else None
    if y_anom is not None:
        sweep_df = threshold_sweep(mah_scores, y_anom)
        out_path = REPO_ROOT / cfg["output"]["threshold_sweep_csv"]
        save_dataframe(sweep_df, out_path)
        log.info("Saved threshold sweep: %s", out_path)

    save_array(mah_scores, REPO_ROOT / cfg["output"]["composite_scores_npy"])

    print("=== Drift detection experiment complete ===")
    print(f"  Mahalanobis: mean={mah_scores.mean():.3f}  flag_rate={mah_result.flags.mean():.1%}")
    print(f"  EWMA rows  : {len(ewma_df)}")
    print(f"  KL rows    : {len(kl_df)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = load_config(args.config)
    with track_run("drift", cfg) as record:
        run(cfg)
        record.add_output(cfg["output"]["composite_scores_npy"])


if __name__ == "__main__":
    main()
