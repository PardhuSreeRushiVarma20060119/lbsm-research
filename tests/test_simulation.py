# tests/test_simulation.py

import numpy as np
import pytest

from src.simulation.agent import AdaptiveAgent
from src.simulation.behavior_profiles import (
    PROFILE_NAMES,
    TELEMETRY_FEATURES,
)


# ---------------------------------------------------------------------------
# Agent initialization
# ---------------------------------------------------------------------------

def test_agent_initialization():

    agent = AdaptiveAgent(
        agent_id="test_agent",
        initial_state="stable",
    )

    print("\n=== AGENT INITIALIZATION ===")

    print("Agent ID:", agent.agent_id)
    print("Initial state:", agent.current_state)
    print("History length:", len(agent.history))

    assert agent.current_state == "stable"
    assert len(agent.history) == 0


# ---------------------------------------------------------------------------
# Single simulation step
# ---------------------------------------------------------------------------

def test_agent_step():

    agent = AdaptiveAgent(rng_seed=42)

    record = agent.step(timestep=0)

    print("\n=== AGENT STEP OUTPUT ===")

    for k, v in record.items():
        print(f"{k:20s}: {v}")

    print("\nTelemetry features present:")

    for feat in TELEMETRY_FEATURES:
        print(f"  {feat}: {'YES' if feat in record else 'NO'}")

    assert record["timestep"] == 0
    assert record["hidden_state"] in PROFILE_NAMES
    assert all(feat in record for feat in TELEMETRY_FEATURES)
    assert len(agent.history) == 1


# ---------------------------------------------------------------------------
# Transition matrix validation
# ---------------------------------------------------------------------------

def test_transition_matrix_validation():

    invalid_T = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ])

    print("\n=== INVALID TRANSITION MATRIX ===")
    print(invalid_T)

    with pytest.raises(ValueError):
        AdaptiveAgent._validate_transition_matrix(invalid_T)

    print("Validation correctly raised ValueError")


# ---------------------------------------------------------------------------
# Stationary distribution
# ---------------------------------------------------------------------------

def test_stationary_distribution():

    agent = AdaptiveAgent()

    pi = agent.stationary_distribution()

    print("\n=== STATIONARY DISTRIBUTION ===")

    for state, prob in zip(PROFILE_NAMES, pi):
        print(f"{state:15s}: {prob:.6f}")

    print("\nDistribution sum:", pi.sum())

    assert np.allclose(pi.sum(), 1.0)
    assert (pi >= 0).all()