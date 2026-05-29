"""src.hmm — LBSM Hidden Markov Model inference.

Public API
----------
The four sub-modules are:

hidden_state_model   — HMMResult dataclass, prepare_sequences, fit_hmm
latent_state_metrics — per_regime_accuracy, per_agent_metrics, posterior_entropy
sequence_inference   — model_selection_sweep, stationary_distribution
transition_analysis  — transition_matrix_error, spectral_gap,
                       expected_dwell_times, empirical_transition_counts
"""

from .hidden_state_model import (
    HMMResult,
    prepare_sequences,
    fit_hmm,
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

__all__ = [
    # Core model
    "HMMResult",
    "prepare_sequences",
    "fit_hmm",
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
]