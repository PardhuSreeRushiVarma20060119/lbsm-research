"""
trajectory_plots.py
====================
LBSM — Trajectory visualisation in feature or embedding space (2D/3D
scatter and per-agent line trajectories, coloured by behavioural regime).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ..simulation.behavior_profiles import PROFILE_NAMES, BEHAVIOR_PROFILES

PALETTE = {name: BEHAVIOR_PROFILES[name].color for name in PROFILE_NAMES}


def plot_trajectory_2d(
    embedding: np.ndarray,
    labels: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    profile_names: Sequence[str] = PROFILE_NAMES,
    title: str = "2D Embedding",
    xlabel: str = "Dim 1",
    ylabel: str = "Dim 2",
    s: float = 6,
    alpha: float = 0.5,
) -> Axes:
    """2D scatter of an embedding, coloured by regime label if provided.

    Parameters
    ----------
    embedding : (N, 2) array
    labels    : (N,) regime names or indices into ``profile_names``; if None,
                all points are plotted in a single colour
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    if labels is None:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=s, alpha=alpha, color="#3498db")
    else:
        labels = np.asarray(labels)
        for name in profile_names:
            mask = labels == name if labels.dtype.kind in "OU" else labels == profile_names.index(name)
            if mask.any():
                ax.scatter(embedding[mask, 0], embedding[mask, 1], s=s, alpha=alpha,
                           color=PALETTE.get(name, None), label=name)
        ax.legend(fontsize=8, markerscale=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax


def plot_trajectory_3d(
    embedding: np.ndarray,
    labels: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    profile_names: Sequence[str] = PROFILE_NAMES,
    title: str = "3D Embedding",
    s: float = 4,
    alpha: float = 0.4,
):
    """3D scatter of an embedding, coloured by regime label if provided.

    Parameters
    ----------
    embedding : (N, 3) array

    Returns
    -------
    ax : a 3D :class:`matplotlib.axes.Axes` (created via ``projection='3d'``
         if ``ax`` is None).
    """
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

    if labels is None:
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=s, alpha=alpha, color="#3498db")
    else:
        labels = np.asarray(labels)
        for name in profile_names:
            mask = labels == name if labels.dtype.kind in "OU" else labels == profile_names.index(name)
            if mask.any():
                ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                           s=s, alpha=alpha, color=PALETTE.get(name, None), label=name)
        ax.legend(fontsize=8, markerscale=2)

    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.set_zlabel("Dim 3")
    ax.set_title(title)
    return ax


def plot_regime_trajectories(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    agent_col: str = "agent_id",
    regime_col: str = "hidden_state",
    ax: Optional[Axes] = None,
    profile_names: Sequence[str] = PROFILE_NAMES,
    max_agents: Optional[int] = None,
    linewidth: float = 0.7,
    alpha: float = 0.5,
    title: str = "Per-Agent Trajectories",
) -> Axes:
    """Line-plot per-agent trajectories in ``(x_col, y_col)`` feature space.

    Each agent's path is drawn as a single line (coloured by its most
    frequent regime) rather than per-point regime colouring, so multi-agent
    plots stay legible.

    Parameters
    ----------
    df         : long-format telemetry, one row per (agent, timestep)
    max_agents : if given, only the first N agent ids (by sort order) are drawn
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))

    agent_ids = sorted(df[agent_col].unique())
    if max_agents is not None:
        agent_ids = agent_ids[:max_agents]

    seen_regimes = set()
    for aid in agent_ids:
        sub = df[df[agent_col] == aid]
        dominant = sub[regime_col].mode().iloc[0]
        label = dominant if dominant not in seen_regimes else None
        seen_regimes.add(dominant)
        ax.plot(sub[x_col].values, sub[y_col].values, color=PALETTE.get(dominant, "gray"),
                linewidth=linewidth, alpha=alpha, label=label)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.legend(fontsize=8, title="Dominant regime")
    return ax
