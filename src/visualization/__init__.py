"""src.visualization — LBSM plotting library.

Note: ``manifold_plots.py`` is deliberately NOT re-exported here. It is a
one-off script (hardcoded relative data paths, writes an HTML file as an
import-time side effect) rather than a function library — importing it from
here would trigger that side effect on every ``import src.visualization``.
Use the modules below instead for manifold/embedding scatter plots.
"""

from .trajectory_plots import (
    plot_trajectory_2d, plot_trajectory_3d, plot_regime_trajectories,
    plot_trajectory_overlay_3d, plot_trajectory_overlay_2d, plot_trajectory_density_3d,
)
from .heatmaps import plot_transition_matrix_heatmap, plot_covariance_heatmap
from .temporal_dynamics import plot_feature_timeseries, plot_state_sequence
from .state_transitions import plot_regime_duration_histogram, plot_transition_flow
from .dashboard import create_analysis_dashboard

__all__ = [
    "plot_trajectory_2d", "plot_trajectory_3d", "plot_regime_trajectories",
    "plot_trajectory_overlay_3d", "plot_trajectory_overlay_2d", "plot_trajectory_density_3d",
    "plot_transition_matrix_heatmap", "plot_covariance_heatmap",
    "plot_feature_timeseries", "plot_state_sequence",
    "plot_regime_duration_histogram", "plot_transition_flow",
    "create_analysis_dashboard",
]
