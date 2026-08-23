"""
random_seed.py
===============
LBSM — Reproducibility helpers.

Centralises the seeding idiom already used ad hoc throughout the codebase
(``rng_seed``, ``base_seed + i`` in :func:`src.simulation.agent.make_agent_pool`,
:func:`src.rl.environment.make_env_pool`, etc.) so scripts and notebooks share
one implementation instead of re-deriving it.
"""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python's ``random`` and NumPy's legacy global RNG.

    Only affects code that uses the global ``numpy.random`` functions or the
    stdlib ``random`` module. Code using an explicit
    :class:`numpy.random.Generator` (the convention elsewhere in this
    codebase) is unaffected and should be seeded via :func:`make_rng`.
    """
    random.seed(seed)
    np.random.seed(seed)


def make_rng(seed: Optional[int]) -> np.random.Generator:
    """Return a fresh :class:`numpy.random.Generator` for the given seed.

    Thin wrapper around ``np.random.default_rng`` kept here so call sites
    read ``make_rng(seed)`` instead of importing numpy just for this.
    """
    return np.random.default_rng(seed)


def spawn_seeds(base_seed: int, n: int) -> List[int]:
    """Derive ``n`` deterministic, distinct seeds from a base seed.

    Matches the ``base_seed + i`` convention already used by
    :func:`make_agent_pool` and :func:`make_env_pool` — kept as a named
    helper so new pool-style factories don't re-derive the same one-liner.
    """
    return [base_seed + i for i in range(n)]
