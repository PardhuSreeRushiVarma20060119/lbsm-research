"""src.drift — LBSM online drift detection and regime shift analysis."""

from .ewma import (
    EWMAResult, ewma_scores, fit_ewma,
    ewma_all_agents, alpha_sweep,
)
from .kl_divergence import (
    KLDriftResult, gaussian_kl, fit_reference,
    kl_drift_scores, fit_kl_detector,
    kl_all_agents, window_size_sweep,
)
from .drift_detection import (
    HealthyEnvelope, MahalanobisResult,
    fit_healthy_envelope, mahalanobis_scores, fit_mahalanobis,
    combined_anomaly_score, threshold_sweep,
)
from .regime_shift_analysis import (
    ground_truth_changepoints, detection_latency,
    detection_latency_summary, shift_magnitude,
    transition_shift_summary,
)

__all__ = [
    # ewma
    "EWMAResult", "ewma_scores", "fit_ewma",
    "ewma_all_agents", "alpha_sweep",
    # kl_divergence
    "KLDriftResult", "gaussian_kl", "fit_reference",
    "kl_drift_scores", "fit_kl_detector",
    "kl_all_agents", "window_size_sweep",
    # drift_detection
    "HealthyEnvelope", "MahalanobisResult",
    "fit_healthy_envelope", "mahalanobis_scores", "fit_mahalanobis",
    "combined_anomaly_score", "threshold_sweep",
    # regime_shift_analysis
    "ground_truth_changepoints", "detection_latency",
    "detection_latency_summary", "shift_magnitude",
    "transition_shift_summary",
]