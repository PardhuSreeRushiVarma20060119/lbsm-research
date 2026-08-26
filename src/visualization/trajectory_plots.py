"""
trajectory_plots.py
====================
LBSM — Trajectory visualisation in feature or embedding space (2D/3D
scatter and per-agent line trajectories, coloured by behavioural regime).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

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


def plot_trajectory_overlay_2d(
    background_embedding: np.ndarray,
    background_labels: np.ndarray,
    overlay_embedding: np.ndarray,
    overlay_values: np.ndarray,
    overlay_kind: str = "categorical",
    overlay_palette: Optional[Dict] = None,
    overlay_cmap: str = "viridis",
    ax: Optional[Axes] = None,
    profile_names: Sequence[str] = PROFILE_NAMES,
    title: str = "Trajectory Overlay on Manifold (2D)",
    background_s: float = 3.0,
    background_alpha: float = 0.12,
    overlay_s: float = 8.0,
    overlay_alpha: float = 0.6,
    overlay_legend_title: str = "",
    colorbar_label: str = "",
) -> Axes:
    """2-D twin of :func:`plot_trajectory_overlay_3d` — same semantics, one
    fewer dimension. See that function's docstring for parameter meaning.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 6.5))

    bg_labels = np.asarray(background_labels)
    for name in profile_names:
        mask = (
            bg_labels == name
            if bg_labels.dtype.kind in "OU"
            else bg_labels == profile_names.index(name)
        )
        if mask.any():
            ax.scatter(
                background_embedding[mask, 0], background_embedding[mask, 1],
                s=background_s, alpha=background_alpha,
                color=PALETTE.get(name, "#999999"), linewidths=0,
            )

    if overlay_kind == "categorical":
        if overlay_palette is None:
            raise ValueError("overlay_palette is required when overlay_kind='categorical'")
        values = np.asarray(overlay_values)
        for val in sorted(set(values.tolist())):
            mask = values == val
            if not mask.any():
                continue
            ax.scatter(
                overlay_embedding[mask, 0], overlay_embedding[mask, 1],
                s=overlay_s, alpha=overlay_alpha,
                color=overlay_palette.get(val, "#333333"), linewidths=0, label=str(val),
            )
        ax.legend(fontsize=8, title=overlay_legend_title, markerscale=2, loc="best")
    elif overlay_kind == "continuous":
        sc = ax.scatter(
            overlay_embedding[:, 0], overlay_embedding[:, 1],
            s=overlay_s, alpha=overlay_alpha, c=overlay_values, cmap=overlay_cmap, linewidths=0,
        )
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label(colorbar_label or "value")
    else:
        raise ValueError("overlay_kind must be 'categorical' or 'continuous'")

    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    return ax


def plot_trajectory_overlay_3d(
    background_embedding: np.ndarray,
    background_labels: np.ndarray,
    overlay_embedding: np.ndarray,
    overlay_values: np.ndarray,
    overlay_kind: str = "categorical",
    overlay_palette: Optional[Dict] = None,
    overlay_cmap: str = "viridis",
    ax: Optional[Axes] = None,
    profile_names: Sequence[str] = PROFILE_NAMES,
    title: str = "Trajectory Overlay on Manifold",
    background_s: float = 2.0,
    background_alpha: float = 0.10,
    overlay_s: float = 10.0,
    overlay_alpha: float = 0.85,
    overlay_legend_title: str = "",
    colorbar_label: str = "",
    elev: float = 22.0,
    azim: float = -60.0,
) -> Axes:
    """3D scatter of a foreground trajectory over a faint background manifold.

    Designed for overlaying an RL trajectory (or any new sequence of points)
    on top of the full NB02 UMAP embedding, so the reader can see where the
    trajectory sits relative to the regimes characterised there. The
    background is deliberately small/low-alpha so the overlay — the actual
    subject of the figure — reads clearly on top of it.

    Parameters
    ----------
    background_embedding : (N, 3) array — e.g. the full NB02 UMAP embedding
    background_labels    : (N,) ground-truth regime names or indices
    overlay_embedding    : (M, 3) array — e.g. an RL trajectory projected
        into the same embedding space via the fitted reducer's ``.transform``
    overlay_values : (M,) array coloring the overlay points. Interpretation
        depends on ``overlay_kind``:
          - "categorical": values are looked up in ``overlay_palette``
            (dict value -> color); legend entry per unique value.
          - "continuous": values are mapped through ``overlay_cmap`` with a
            colorbar (e.g. episode index / training phase progression).
    overlay_kind : "categorical" or "continuous"
    overlay_palette : required when overlay_kind == "categorical"
    elev, azim : matplotlib 3D view angle (``Axes3D.view_init``)

    Returns
    -------
    ax : the 3D Axes
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")

    # ── Background: faint, regime-colored context
    bg_labels = np.asarray(background_labels)
    for name in profile_names:
        mask = (
            bg_labels == name
            if bg_labels.dtype.kind in "OU"
            else bg_labels == profile_names.index(name)
        )
        if mask.any():
            ax.scatter(
                background_embedding[mask, 0],
                background_embedding[mask, 1],
                background_embedding[mask, 2],
                s=background_s, alpha=background_alpha,
                color=PALETTE.get(name, "#999999"),
                linewidths=0, label=None,
            )

    # ── Overlay: the actual subject of the figure
    if overlay_kind == "categorical":
        if overlay_palette is None:
            raise ValueError("overlay_palette is required when overlay_kind='categorical'")
        values = np.asarray(overlay_values)
        for val in sorted(set(values.tolist())):
            mask = values == val
            if not mask.any():
                continue
            ax.scatter(
                overlay_embedding[mask, 0],
                overlay_embedding[mask, 1],
                overlay_embedding[mask, 2],
                s=overlay_s, alpha=overlay_alpha,
                color=overlay_palette.get(val, "#333333"),
                linewidths=0, label=str(val),
            )
        ax.legend(fontsize=8, title=overlay_legend_title, markerscale=2, loc="upper left")
    elif overlay_kind == "continuous":
        sc = ax.scatter(
            overlay_embedding[:, 0], overlay_embedding[:, 1], overlay_embedding[:, 2],
            s=overlay_s, alpha=overlay_alpha,
            c=overlay_values, cmap=overlay_cmap, linewidths=0,
        )
        cb = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.08)
        cb.set_label(colorbar_label or "value")
    else:
        raise ValueError("overlay_kind must be 'categorical' or 'continuous'")

    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.set_zlabel("UMAP 3")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    return ax


def plot_trajectory_density_3d(
    background_embedding: np.ndarray,
    background_labels: np.ndarray,
    phase_points: Dict[str, np.ndarray],
    phase_palette: Dict[str, str],
    ax: Optional[Axes] = None,
    profile_names: Sequence[str] = PROFILE_NAMES,
    title: str = "Trajectory Density over Manifold (floor-projected KDE)",
    grid_resolution: int = 80,
    background_s: float = 1.5,
    background_alpha: float = 0.08,
    overlay_s: float = 4.0,
    overlay_alpha: float = 0.35,
    contour_alpha: float = 0.55,
    n_contour_levels: int = 6,
    elev: float = 22.0,
    azim: float = -60.0,
) -> Axes:
    """3-D scatter with each phase's 2-D (UMAP1, UMAP2) KDE density contour
    projected onto the plot's floor (``zdir='z'`` at the minimum Z).

    True 3-D isosurface rendering (marching cubes) is possible but heavy and
    hard to read in a static figure; projecting the XY-marginal density onto
    the floor is the standard, robust way to add density information to a 3-D
    scatter without that complexity — the scatter still carries the full 3-D
    structure, the floor contours summarise where each phase concentrates.

    Parameters
    ----------
    phase_points  : dict phase_name -> (N_phase, 3) array (UMAP1, UMAP2, UMAP3)
    phase_palette : dict phase_name -> color (same keys as ``phase_points``)
    """
    from scipy.stats import gaussian_kde

    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

    # Background context (faint, regime-colored)
    bg_labels = np.asarray(background_labels)
    for name in profile_names:
        mask = (
            bg_labels == name
            if bg_labels.dtype.kind in "OU"
            else bg_labels == profile_names.index(name)
        )
        if mask.any():
            ax.scatter(
                background_embedding[mask, 0], background_embedding[mask, 1],
                background_embedding[mask, 2],
                s=background_s, alpha=background_alpha,
                color=PALETTE.get(name, "#999999"), linewidths=0,
            )

    all_pts = np.concatenate(list(phase_points.values()), axis=0)
    z_floor = all_pts[:, 2].min()
    x_min, x_max = all_pts[:, 0].min() - 0.5, all_pts[:, 0].max() + 0.5
    y_min, y_max = all_pts[:, 1].min() - 0.5, all_pts[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid_pts = np.vstack([xx.ravel(), yy.ravel()])

    for phase, pts in phase_points.items():
        color = phase_palette.get(phase, "#333333")
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            s=overlay_s, alpha=overlay_alpha, color=color, linewidths=0, label=phase,
        )
        kde = gaussian_kde(pts[:, :2].T, bw_method=0.15)
        zz = kde(grid_pts).reshape(xx.shape)
        ax.contour(
            xx, yy, zz, zdir="z", offset=z_floor,
            levels=n_contour_levels, colors=[color], alpha=contour_alpha, linewidths=1.2,
        )

    ax.legend(fontsize=8, title="phase", markerscale=3, loc="upper left")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.set_zlabel("UMAP 3")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    return ax
