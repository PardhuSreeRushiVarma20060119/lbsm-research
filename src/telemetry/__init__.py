"""src.telemetry — LBSM telemetry processing utilities."""

from .windowing import sliding_windows, window_statistics, reference_test_split, per_agent_windows
from .normalization import (
    fit_zscore, apply_zscore, zscore_matrix,
    fit_minmax, apply_minmax, normalize_scores,
    zscore_dataframe, ZScoreParams, MinMaxParams,
)
from .statistics import (
    regime_summary, fisher_separability,
    anomaly_rate_by_regime, per_agent_summary,
    bhattacharyya_distance,
)
from .feature_extraction import (
    rolling_mean, rolling_std, temporal_diff,
    composite_health_score, augment_phase_space,
)
from .preprocessing import (
    clip_features, enforce_dtypes, drop_incomplete,
    temporal_train_test_split, to_feature_matrix,
    FEATURE_BOUNDS,
)

__all__ = [
    # windowing
    "sliding_windows", "window_statistics",
    "reference_test_split", "per_agent_windows",
    # normalization
    "fit_zscore", "apply_zscore", "zscore_matrix",
    "fit_minmax", "apply_minmax", "normalize_scores",
    "zscore_dataframe", "ZScoreParams", "MinMaxParams",
    # statistics
    "regime_summary", "fisher_separability",
    "anomaly_rate_by_regime", "per_agent_summary",
    "bhattacharyya_distance",
    # feature extraction
    "rolling_mean", "rolling_std", "temporal_diff",
    "composite_health_score", "augment_phase_space",
    # preprocessing
    "clip_features", "enforce_dtypes", "drop_incomplete",
    "temporal_train_test_split", "to_feature_matrix",
    "FEATURE_BOUNDS",
]