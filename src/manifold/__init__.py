"""src.manifold — LBSM manifold learning & geometry analysis."""

from .pca import fit_pca, PCAResult, regime_centroids_pca, loading_dominance
from .umap_projection import fit_umap, UMAPResult, hyperparameter_sweep, per_regime_density, regime_connectivity
from .tsne import fit_tsne, TSNEResult, perplexity_sweep, intra_regime_spread
from .manifold_metrics import (
    embedding_scorecard, compare_embeddings, per_regime_silhouette,
    neighbourhood_purity, embedding_agreement, continuity,
)
from .trajectory_geometry import (
    extract_agent_trajectories, compute_trajectory_stats,
    transition_embedding_coords, manifold_velocity, regime_arc_statistics,
)

__all__ = [
    # pca
    "fit_pca", "PCAResult", "regime_centroids_pca", "loading_dominance",
    # umap
    "fit_umap", "UMAPResult", "hyperparameter_sweep",
    "per_regime_density", "regime_connectivity",
    # tsne
    "fit_tsne", "TSNEResult", "perplexity_sweep", "intra_regime_spread",
    # metrics
    "embedding_scorecard", "compare_embeddings", "per_regime_silhouette",
    "neighbourhood_purity", "embedding_agreement", "continuity",
    # trajectory
    "extract_agent_trajectories", "compute_trajectory_stats",
    "transition_embedding_coords", "manifold_velocity", "regime_arc_statistics",
]