"""Shared fixtures.

Statistical validation needs replications, and meaningful replication counts are
too slow for an edit-test loop. Every such test scales its work off
``n_replications``, which is :func:`simcheck.reps_for`: 100 normally, 400 when
``SIMCHECK_DEEP`` is set, and whatever ``SIMCHECK_REPS`` says when the scheduled
job wants a deeper study than that.

The count comes from simcheck rather than from a repo-local environment variable
because the assertions it feeds derive their tolerance from it. Raising the
replicate count tightens every band without a test being edited, and lowering it
cannot quietly weaken one -- the band widens visibly in the failure message. A
local ``SORTITION_FULL_SIMS`` returning a hand-picked 1000 or 120 kept the two
numbers independent, which is the arrangement that lets a threshold drift away
from the study it is meant to describe.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from simcheck import reps_for

from sortition.sim import BanditProblem, make_problem

# CI builds the wheel, installs it with base dependencies only, and runs this
# suite against it -- which is how the minimal install gets proven to work. Most
# tests need an optional extra, so they are skipped wholesale rather than failing
# collection. Declared here rather than as per-module guards so there is one
# place to look when a test unexpectedly does not run.
_NEEDS: dict[str, tuple[str, ...]] = {
    "test_ci.py": ("scipy",),
    "test_decide.py": ("scipy",),
    "test_estimators.py": ("scipy", "sklearn"),
    "test_dashboard.py": ("polars", "scipy"),
    "test_health.py": ("polars", "scipy"),
    "test_report.py": ("polars", "scipy"),
    "test_reporting.py": ("polars", "scipy", "rich"),
    "test_store.py": ("polars", "duckdb"),
    "test_targets.py": ("scipy",),
    "test_clean_cases.py": ("polars", "scipy", "lightgbm"),
    "test_sweep.py": ("polars", "scipy", "lightgbm"),
    "test_train.py": ("polars", "scipy", "lightgbm"),
    "test_cli.py": ("typer", "polars"),
}
collect_ignore = [
    name
    for name, modules in _NEEDS.items()
    if any(importlib.util.find_spec(m) is None for m in modules)
]


@pytest.fixture(scope="session")
def n_replications() -> int:
    return reps_for()


@pytest.fixture
def problem() -> BanditProblem:
    return make_problem(n_contexts=400, n_arms=4, n_features=6, seed=0)


@pytest.fixture
def weights() -> tuple[np.ndarray, np.ndarray]:
    """A behavior and a target scoring matrix that genuinely disagree."""
    rng = np.random.default_rng(1)
    return rng.standard_normal((6, 4)), rng.standard_normal((6, 4))
