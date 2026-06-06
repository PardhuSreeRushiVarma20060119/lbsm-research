"""src.evaluation — LBSM cross-notebook evaluation metrics."""

from .clustering_metrics import clustering_scorecard, per_class_silhouette, ari_score
from .manifold_quality import (
    embedding_trustworthiness, embedding_continuity,
    neighbourhood_purity, procrustes_agreement,
)
from .explained_variance import (
    pca_explained_variance, n_components_for_threshold,
    intrinsic_dimensionality_estimate,
)
from .stability_metrics import bootstrap_auc, detector_stability_table
from .trajectory_metrics import (
    path_length, displacement, tortuosity,
    mean_speed, trajectory_summary,
)

__all__ = [
    "clustering_scorecard", "per_class_silhouette", "ari_score",
    "embedding_trustworthiness", "embedding_continuity",
    "neighbourhood_purity", "procrustes_agreement",
    "pca_explained_variance", "n_components_for_threshold",
    "intrinsic_dimensionality_estimate",
    "bootstrap_auc", "detector_stability_table",
    "path_length", "displacement", "tortuosity",
    "mean_speed", "trajectory_summary",
]