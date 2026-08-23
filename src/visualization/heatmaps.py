"""
heatmaps.py
===========
LBSM — Matrix visualisations: transition matrices and covariance/correlation
matrices. Every function accepts an optional ``ax`` so panels compose into
:mod:`src.visualization.dashboard`, and returns that ``ax`` for chaining.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ..simulation.behavior_profiles import PROFILE_NAMES


def plot_transition_matrix_heatmap(
    T: np.ndarray,
    profile_names: Sequence[str] = PROFILE_NAMES,
    ax: Optional[Axes] = None,
    title: str = "Transition Matrix",
    cmap: str = "viridis",
    annotate: bool = True,
    fmt: str = ".2f",
) -> Axes:
    """Heatmap of a row-stochastic transition matrix ``T[i, j] = P(j | i)``.

    Parameters
    ----------
    T             : (k, k) row-stochastic matrix
    profile_names : row/column labels, length k
    ax            : existing axes to draw into; a new figure is created if None
    annotate      : if True, print each cell's value
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))

    im = ax.imshow(T, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(profile_names)))
    ax.set_yticks(range(len(profile_names)))
    ax.set_xticklabels(profile_names, rotation=45, ha="right")
    ax.set_yticklabels(profile_names)
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title(title)

    if annotate:
        threshold = T.max() / 2.0
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                color = "white" if T[i, j] < threshold else "black"
                ax.text(j, i, format(T[i, j], fmt), ha="center", va="center", color=color, fontsize=9)

    fig = ax.get_figure()
    fig.colorbar(im, ax=ax, shrink=0.85, label="P(to | from)")
    return ax


def plot_covariance_heatmap(
    cov: np.ndarray,
    feature_names: Sequence[str],
    ax: Optional[Axes] = None,
    title: str = "Covariance Matrix",
    cmap: str = "coolwarm",
    annotate: bool = True,
    fmt: str = ".2f",
) -> Axes:
    """Heatmap of a covariance (or correlation) matrix, symmetric colour scale.

    Parameters
    ----------
    cov           : (d, d) covariance or correlation matrix
    feature_names : row/column labels, length d
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))

    vmax = np.abs(cov).max()
    im = ax.imshow(cov, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")
    ax.set_yticklabels(feature_names)
    ax.set_title(title)

    if annotate:
        for i in range(cov.shape[0]):
            for j in range(cov.shape[1]):
                color = "white" if abs(cov[i, j]) > vmax * 0.6 else "black"
                ax.text(j, i, format(cov[i, j], fmt), ha="center", va="center", color=color, fontsize=8)

    fig = ax.get_figure()
    fig.colorbar(im, ax=ax, shrink=0.85)
    return ax
