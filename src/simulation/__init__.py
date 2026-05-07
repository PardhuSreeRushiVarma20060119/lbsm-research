"""simulation — LBSM core simulation package."""

from .behavior_profiles import (
    BEHAVIOR_PROFILES,
    PROFILE_NAMES,
    TELEMETRY_FEATURES,
    BehaviorProfile,
    get_profile,
    profile_distance_matrix,
    regime_separability_ratio,
)
from .agent import AdaptiveAgent, make_agent, make_agent_pool, DEFAULT_TRANSITION_MATRIX
from .telemetry_generator import TelemetryGenerator, generate_and_save

__all__ = [
    # profiles
    "BEHAVIOR_PROFILES", "PROFILE_NAMES", "TELEMETRY_FEATURES",
    "BehaviorProfile", "get_profile",
    "profile_distance_matrix", "regime_separability_ratio",
    # agent
    "AdaptiveAgent", "make_agent", "make_agent_pool",
    "DEFAULT_TRANSITION_MATRIX",
    # generator
    "TelemetryGenerator", "generate_and_save",
]