"""
temporal_dynamics.py
=====================
LBSM — Time-series visualisation: a single feature's evolution, and a
regime-occupancy timeline strip.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from ..simulation.behavior_profiles import PROFILE_NAMES, BEHAVIOR_PROFILES

PALETTE = {name: BEHAVIOR_PROFILES[name].color for name in PROFILE_NAMES}


def plot_feature_timeseries(
    df: pd.DataFrame,
    feature: str,
    agent_id: Optional[str] = None,
    time_col: str = "timestep",
    agent_col: str = "agent_id",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    color: str = "#1a5276",
    alpha: float = 0.8,
) -> Axes:
    """Line plot of one feature over time, for a single agent or averaged
    across all agents at each timestep.

    Parameters
    ----------
    agent_id : if given, plot that agent's raw series; otherwise plot the
               mean (and shaded std) across all agents at each timestep
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.5))

    if agent_id is not None:
        sub = df[df[agent_col] == agent_id].sort_values(time_col)
        ax.plot(sub[time_col].values, sub[feature].values, color=color, alpha=alpha)
    else:
        grouped = df.groupby(time_col)[feature]
        mean = grouped.mean()
        std = grouped.std()
        ax.plot(mean.index.values, mean.values, color=color, alpha=alpha, label="mean")
        ax.fill_between(mean.index.values, (mean - std).values, (mean + std).values,
                         color=color, alpha=0.15, label="±1 std")
        ax.legend(fontsize=8)

    ax.set_xlabel(time_col)
    ax.set_ylabel(feature)
    ax.set_title(title or f"{feature} over time" + (f" (agent {agent_id})" if agent_id else " (pooled)"))
    return ax


def plot_state_sequence(
    states: Sequence[str],
    time: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    profile_names: Sequence[str] = PROFILE_NAMES,
    title: str = "Regime Sequence",
) -> Axes:
    """Timeline strip showing regime occupancy as coloured horizontal spans.

    Parameters
    ----------
    states : sequence of regime names, one per timestep
    time   : optional x-axis values (defaults to ``range(len(states))``)
    """
    states = np.asarray(states)
    n = len(states)
    if time is None:
        time = np.arange(n)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 1.2))

    # Compress consecutive identical states into (start, end, state) spans
    change_points = np.where(states[1:] != states[:-1])[0] + 1
    boundaries = np.concatenate(([0], change_points, [n]))
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        regime = states[start]
        ax.axvspan(time[start], time[min(end, n - 1)], color=PALETTE.get(regime, "gray"), alpha=0.9)

    ax.set_yticks([])
    ax.set_xlim(time[0], time[-1])
    ax.set_xlabel("timestep")
    ax.set_title(title)
    handles = [Patch(color=PALETTE[name], label=name) for name in profile_names]
    ax.legend(handles=handles, fontsize=7, ncol=len(profile_names), loc="upper center",
              bbox_to_anchor=(0.5, -0.35), frameon=False)
    return ax
