"""Shared fixtures.

Statistical validation needs replications, and meaningful replication counts are
too slow for an edit-test loop. Every such test scales its work off
``n_replications``, which is small locally and full-size in CI where
``SORTITION_FULL_SIMS`` is set.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from sortition.sim import BanditProblem, make_problem

FULL = bool(os.environ.get("SORTITION_FULL_SIMS"))

# CI builds the wheel, installs it with base dependencies only, and runs this
# suite against it -- which is how the minimal install gets proven to work. Most
# tests need an optional extra, so they are skipped wholesale rather than failing
# collection. Declared here rather than as per-module guards so there is one
# place to look when a test unexpectedly does not run.
_NEEDS: dict[str, tuple[str, ...]] = {
    "test_ci.py": ("scipy",),
    "test_decide.py": ("scipy",),
    "test_estimators.py": ("scipy", "sklearn"),
    "test_health.py": ("polars", "scipy"),
    "test_report.py": ("polars", "scipy"),
    "test_store.py": ("polars", "duckdb"),
    "test_cli.py": ("typer", "polars"),
}
collect_ignore = [
    name
    for name, modules in _NEEDS.items()
    if any(importlib.util.find_spec(m) is None for m in modules)
]


@pytest.fixture(scope="session")
def n_replications() -> int:
    return 1000 if FULL else 120


@pytest.fixture(scope="session")
def full_sims() -> bool:
    return FULL


@pytest.fixture
def problem() -> BanditProblem:
    return make_problem(n_contexts=400, n_arms=4, n_features=6, seed=0)


@pytest.fixture
def weights() -> tuple[np.ndarray, np.ndarray]:
    """A behavior and a target scoring matrix that genuinely disagree."""
    rng = np.random.default_rng(1)
    return rng.standard_normal((6, 4)), rng.standard_normal((6, 4))
