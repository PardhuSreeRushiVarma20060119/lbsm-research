"""
dashboard.py
============
LBSM — Composite multi-panel analysis dashboard, built from the other
src.visualization modules rather than duplicating their plotting logic.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..simulation.behavior_profiles import PROFILE_NAMES
from .trajectory_plots import plot_trajectory_2d
from .heatmaps import plot_transition_matrix_heatmap
from .temporal_dynamics import plot_feature_timeseries, plot_state_sequence


def create_analysis_dashboard(
    df: pd.DataFrame,
    embedding: np.ndarray,
    transition_matrix: np.ndarray,
    feature: str = "latency",
    agent_id: Optional[str] = None,
    labels: Optional[np.ndarray] = None,
    regime_col: str = "hidden_state",
    profile_names: Sequence[str] = PROFILE_NAMES,
    title: str = "LBSM Analysis Dashboard",
) -> Figure:
    """Four-panel summary figure: manifold embedding, transition matrix,
    a feature's time series, and one agent's regime-occupancy timeline.

    Parameters
    ----------
    df                : long-format telemetry (one row per agent x timestep)
    embedding         : (N, 2) manifold embedding aligned with ``df`` rows
    transition_matrix : (k, k) row-stochastic transition matrix to display
    feature           : which telemetry feature to plot as a time series
    agent_id          : agent to use for the regime-sequence panel; defaults
                        to the first agent in ``df`` if not given
    labels            : regime labels for the embedding scatter, aligned
                        with ``embedding`` rows; defaults to ``df[regime_col]``

    Returns
    -------
    fig : the composed :class:`matplotlib.figure.Figure`
    """
    if labels is None:
        labels = df[regime_col].values
    if agent_id is None:
        agent_id = sorted(df["agent_id"].unique())[0]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    plot_trajectory_2d(embedding, labels, ax=axes[0, 0], profile_names=profile_names,
                        title="Manifold Embedding")
    plot_transition_matrix_heatmap(transition_matrix, profile_names=profile_names,
                                    ax=axes[0, 1], title="Transition Matrix")
    plot_feature_timeseries(df, feature, ax=axes[1, 0], title=f"{feature} (pooled mean ± std)")

    agent_df = df[df["agent_id"] == agent_id].sort_values("timestep")
    plot_state_sequence(agent_df[regime_col].values, time=agent_df["timestep"].values,
                         ax=axes[1, 1], profile_names=profile_names,
                         title=f"Regime Sequence (agent {agent_id})")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
