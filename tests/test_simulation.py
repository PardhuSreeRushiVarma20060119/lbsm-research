# tests/test_simulation.py
def test_agent_initialization():
    agent = AdaptiveAgent(agent_id="test_agent", initial_state="stable")
    assert agent.current_state == "stable"
    assert len(agent.history) == 0


def test_agent_step():
    agent = AdaptiveAgent(rng_seed=42)
    record = agent.step(timestep=0)

    assert record["timestep"] == 0
    assert record["hidden_state"] in PROFILE_NAMES
    assert all(feat in record for feat in TELEMETRY_FEATURES)
    assert len(agent.history) == 1


def test_transition_matrix_validation():
    invalid_T = np.array([[0.5, 0.5], [0.5, 0.5]])  # Wrong shape
    with pytest.raises(ValueError):
        AdaptiveAgent._validate_transition_matrix(invalid_T)


def test_stationary_distribution():
    agent = AdaptiveAgent()
    pi = agent.stationary_distribution()
    assert np.allclose(pi.sum(), 1.0)  # Probability distribution
    assert (pi >= 0).all()  # Non-negative
