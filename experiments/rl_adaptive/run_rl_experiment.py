"""
run_rl_experiment.py
=====================
Standalone RL experiment: train a Q-learning pool inside BehavioralEnv and
evaluate it against a random-policy baseline (the same comparison used for
NB05's C1/C2 evidence-checklist criteria — see outputs/reports/nb05), without
going through notebooks/05_rl_behavioral_evolution.ipynb.

Requires experiments/baseline/run_baseline.py to have been run first (or the
telemetry CSV to already exist, needed to fit the healthy envelope).

Usage
-----
    python experiments/rl_adaptive/run_rl_experiment.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.simulation.behavior_profiles import TELEMETRY_FEATURES
from src.telemetry import temporal_train_test_split
from src.drift import fit_healthy_envelope
from src.rl import (
    BehavioralEnv, make_env_pool, QLearningConfig, train_agent_pool,
    N_ACTIONS, pool_learning_curves,
)
from src.utils import get_logger, save_array, save_dataframe, track_run

log = get_logger(__name__)

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _random_policy_baseline(n_episodes: int, seed: int, n_steps: int, delta: float) -> pd.DataFrame:
    """Independent random-policy episodes — the null-model reference used by C1/C2."""
    rows = []
    for i in range(n_episodes):
        env = BehavioralEnv(agent_id=f"rand_baseline_{i}", rng_seed=seed + i,
                             n_steps=n_steps, delta=delta)
        env.reset()
        rng = np.random.default_rng(seed + i)
        total_r, steps, n_unstable = 0.0, 0, 0
        while True:
            result = env.step(int(rng.integers(0, N_ACTIONS)))
            total_r += result.reward
            n_unstable += result.info["hidden_state"] == "unstable"
            steps += 1
            if result.done:
                break
        rows.append({"episode": i, "total_reward": total_r, "unstable_frac": n_unstable / steps})
    return pd.DataFrame(rows)


def run(cfg: dict) -> None:
    telemetry_path = REPO_ROOT / "data" / "processed" / "nb01" / "telemetry_n20_t2000.csv"
    if not telemetry_path.exists():
        raise FileNotFoundError(
            f"{telemetry_path} not found — run experiments/baseline/run_baseline.py first."
        )
    df = pd.read_csv(telemetry_path)
    feature_cols = list(TELEMETRY_FEATURES)
    env_cfg = cfg["healthy_envelope"]
    df_train, _ = temporal_train_test_split(df, test_frac=0.20)
    envelope = fit_healthy_envelope(
        df=df_train, feature_cols=feature_cols, regime_col="hidden_state",
        healthy_regimes=tuple(env_cfg["healthy_regimes"]), regularize=env_cfg["regularize"],
    )

    e_cfg, q_cfg, p_cfg, ev_cfg = cfg["environment"], cfg["q_learning"], cfg["pool"], cfg["evaluation"]

    log.info("Training %d agents x %d episodes x %d steps",
              p_cfg["n_envs"], q_cfg["n_episodes"], e_cfg["n_steps_per_episode"])
    envs = make_env_pool(n_envs=p_cfg["n_envs"], base_seed=p_cfg["base_seed"],
                          delta=e_cfg["delta_base"], n_steps=e_cfg["n_steps_per_episode"])
    q_config = QLearningConfig(
        alpha=q_cfg["alpha"], gamma=q_cfg["gamma"],
        epsilon_start=q_cfg["epsilon_start"], epsilon_end=q_cfg["epsilon_end"],
        epsilon_decay=q_cfg["epsilon_decay"], n_episodes=q_cfg["n_episodes"], seed=q_cfg["seed"],
    )
    agents, train_dfs = train_agent_pool(envs=envs, config=q_config, healthy_envelope=envelope)

    pool_df = pool_learning_curves(train_dfs, smooth_window=10)
    train_log_path = REPO_ROOT / cfg["output"]["training_log_csv"]
    save_dataframe(
        pd.concat([d.assign(agent_id=f"agent_{i:04d}") for i, d in enumerate(train_dfs)],
                   ignore_index=True),
        train_log_path,
    )
    Q_all = np.stack([a.Q for a in agents])
    save_array(Q_all, REPO_ROOT / cfg["output"]["q_tables_npy"])

    eval_dfs = [a.evaluate(n_episodes=ev_cfg["n_eval_episodes"], healthy_envelope=envelope)
                for a in agents]
    eval_reward = float(np.mean([d["total_reward"].mean() for d in eval_dfs]))
    eval_unstable = float(np.mean([d["unstable_frac"].mean() for d in eval_dfs]))

    baseline_df = _random_policy_baseline(
        n_episodes=ev_cfg["n_baseline_episodes"], seed=q_cfg["seed"],
        n_steps=e_cfg["n_steps_per_episode"], delta=e_cfg["delta_base"],
    )
    baseline_reward = baseline_df["total_reward"].mean()
    baseline_unstable = baseline_df["unstable_frac"].mean()

    c1 = eval_unstable < baseline_unstable
    c2 = eval_reward > baseline_reward

    print("=== RL adaptive experiment complete ===")
    print(f"  Pool final unstable frac (last 10 ep): {pool_df['mean_unstable'].values[-10:].mean():.3f}")
    print(f"  Greedy eval  — reward={eval_reward:.1f}  unstable_frac={eval_unstable:.3f}")
    print(f"  Random base  — reward={baseline_reward:.1f}  unstable_frac={baseline_unstable:.3f}")
    print(f"  C1 (unstable frac, trained < random): {'PASS' if c1 else 'FAIL'}")
    print(f"  C2 (reward, trained > random)       : {'PASS' if c2 else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = load_config(args.config)
    with track_run("rl_adaptive", cfg) as record:
        run(cfg)
        record.add_output(cfg["output"]["training_log_csv"])
        record.add_output(cfg["output"]["q_tables_npy"])


if __name__ == "__main__":
    main()
