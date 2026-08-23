"""
run_baseline.py
================
Standalone baseline experiment: generate ground-truth agent telemetry and
save it, without going through notebooks/01_telemetry_generation.ipynb.

Usage
-----
    python experiments/baseline/run_baseline.py
    python experiments/baseline/run_baseline.py --config path/to/other.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.simulation import TelemetryGenerator
from src.utils import get_logger, ensure_dir, save_array, track_run

log = get_logger(__name__)

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(cfg: dict) -> None:
    sim_cfg = cfg["simulation"]
    out_cfg = cfg["output"]

    log.info(
        "Generating telemetry: n_agents=%d n_timesteps=%d seed=%s",
        sim_cfg["n_agents"], sim_cfg["n_timesteps"], sim_cfg["random_seed"],
    )
    gen = TelemetryGenerator(
        n_agents=sim_cfg["n_agents"],
        n_timesteps=sim_cfg["n_timesteps"],
        initial_states=sim_cfg.get("initial_states"),
        seed=sim_cfg["random_seed"],
        verbose=False,
    )
    df = gen.run()

    telemetry_path = REPO_ROOT / out_cfg["telemetry_csv"]
    ensure_dir(telemetry_path.parent)
    gen.save(str(telemetry_path))
    log.info("Saved telemetry: %s (%d rows)", telemetry_path, len(df))

    X = gen.feature_matrix(z_scored=False)
    y = gen.labels()
    save_array(X, REPO_ROOT / out_cfg["feature_matrix_npy"])
    save_array(y, REPO_ROOT / out_cfg["labels_npy"])
    log.info("Saved feature matrix %s and labels %s", X.shape, y.shape)

    print("=== Baseline experiment complete ===")
    print(f"  Telemetry     : {telemetry_path}  ({len(df)} rows)")
    print(f"  Feature matrix: {out_cfg['feature_matrix_npy']}  shape={X.shape}")
    print(f"  Labels        : {out_cfg['labels_npy']}  shape={y.shape}")
    print()
    print("State frequencies:")
    print(gen.state_frequencies().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = load_config(args.config)
    with track_run("baseline", cfg) as record:
        run(cfg)
        record.add_output(cfg["output"]["telemetry_csv"])
        record.add_output(cfg["output"]["feature_matrix_npy"])
        record.add_output(cfg["output"]["labels_npy"])


if __name__ == "__main__":
    main()
