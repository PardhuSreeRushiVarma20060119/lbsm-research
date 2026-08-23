"""
state_transitions.py
=====================
LBSM — Regime-dynamics visualisations: dwell-duration histograms and a
transition-flow diagram (a 100%-stacked bar per source regime, since a true
multi-node Sankey needs a plotting dependency this project doesn't otherwise
use; the stacked-bar form shows the same from→to probability mass without one).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ..simulation.behavior_profiles import PROFILE_NAMES, BEHAVIOR_PROFILES

PALETTE = {name: BEHAVIOR_PROFILES[name].color for name in PROFILE_NAMES}


def plot_regime_duration_histogram(
    durations: Sequence[int],
    ax: Optional[Axes] = None,
    bins: int = 30,
    color: str = "#5b2c6f",
    title: str = "Regime Dwell-Duration Distribution",
) -> Axes:
    """Histogram of regime dwell durations (consecutive timesteps in one regime).

    Parameters
    ----------
    durations : one value per contiguous regime run (e.g. from run-length
                encoding a hidden-state sequence)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.hist(durations, bins=bins, color=color, alpha=0.85, edgecolor="white")
    ax.set_xlabel("Dwell duration (timesteps)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    return ax


def plot_transition_flow(
    transition_matrix: np.ndarray,
    profile_names: Sequence[str] = PROFILE_NAMES,
    ax: Optional[Axes] = None,
    title: str = "Transition Flow",
) -> Axes:
    """100%-stacked horizontal bar chart of a row-stochastic transition matrix.

    One bar per source ("from") regime; each bar's segments show what
    fraction of its outgoing probability mass goes to each destination
    ("to") regime, coloured by destination.

    Parameters
    ----------
    transition_matrix : (k, k) row-stochastic matrix, ``T[i, j] = P(j | i)``
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.6 * len(profile_names) + 1.5))

    y_pos = np.arange(len(profile_names))
    left = np.zeros(len(profile_names))
    for j, to_name in enumerate(profile_names):
        widths = transition_matrix[:, j]
        ax.barh(y_pos, widths, left=left, color=PALETTE.get(to_name, "gray"),
                label=to_name, edgecolor="white", height=0.7)
        left += widths

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"from {n}" for n in profile_names])
    ax.set_xlabel("P(to state)")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.legend(title="to state", fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    return ax
