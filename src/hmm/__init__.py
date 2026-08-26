"""src.hmm — LBSM Hidden Markov Model inference.

Public API
----------
The five sub-modules are:

hidden_state_model   — HMMResult dataclass, prepare_sequences, fit_hmm
latent_state_metrics — per_regime_accuracy, per_agent_metrics, posterior_entropy
sequence_inference   — model_selection_sweep, stationary_distribution
transition_analysis  — transition_matrix_error, spectral_gap,
                       expected_dwell_times, empirical_transition_counts
robust_fitting       — fit_hmm_robust, check_covariance_health,
                       check_data_sufficiency (LBSM-ISSUE-NB07-001 mitigation
                       stack for covariance_type="full" on small samples)
"""

from .hidden_state_model import (
    HMMResult,
    prepare_sequences,
    fit_hmm,
    align_and_score,
)

from .latent_state_metrics import (
    per_regime_accuracy,
    per_agent_metrics,
    posterior_entropy,
)

from .sequence_inference import (
    model_selection_sweep,
    stationary_distribution,
)

from .transition_analysis import (
    transition_matrix_error,
    spectral_gap,
    expected_dwell_times,
    empirical_transition_counts,
)

from .robust_fitting import (
    RobustHMMResult,
    fit_hmm_robust,
    check_covariance_health,
    check_data_sufficiency,
)

__all__ = [
    # Core model
    "HMMResult",
    "prepare_sequences",
    "fit_hmm",
    "align_and_score",
    # Metrics
    "per_regime_accuracy",
    "per_agent_metrics",
    "posterior_entropy",
    # Sequence / model selection
    "model_selection_sweep",
    "stationary_distribution",
    # Transition analysis
    "transition_matrix_error",
    "spectral_gap",
    "expected_dwell_times",
    "empirical_transition_counts",
    # Robust fitting (LBSM-ISSUE-NB07-001)
    "RobustHMMResult",
    "fit_hmm_robust",
    "check_covariance_health",
    "check_data_sufficiency",
]