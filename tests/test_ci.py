"""Confidence intervals must cover at the rate they advertise.

Coverage is the only property that matters here, and it is only observable over
replications: a single interval either contains the truth or does not, which
says nothing. Each test below re-runs the whole experiment many times and counts.
"""

from __future__ import annotations

import numpy as np
import pytest

from sortition.eval.ci import betting_ci, bootstrap_ci, normal_ci


def _bernoulli_coverage(
    p_true: float,
    n: int,
    reps: int,
    method: str,
    *,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Empirical coverage and mean width of an interval for a Bernoulli mean."""
    rng = np.random.default_rng(seed)
    covered = 0
    width = 0.0
    for _ in range(reps):
        x = (rng.random(n) < p_true).astype(np.float64)
        if method == "betting":
            interval = betting_ci(x, alpha=alpha)
        elif method == "betting_anytime":
            interval = betting_ci(x, alpha=alpha, anytime=True)
        elif method == "bootstrap":
            interval = bootstrap_ci(
                x, alpha=alpha, n_resamples=399, seed=int(rng.integers(1_000_000))
            )
        else:
            interval = normal_ci(x, alpha=alpha)
        covered += interval.covers(p_true)
        width += interval.width
    return covered / reps, width / reps


class TestBettingInterval:
    """The betting interval's guarantee is one-sided: coverage is at least the
    nominal level. It is conservative by construction, so the assertion is a
    floor, not a band."""

    @pytest.mark.parametrize(("p_true", "n"), [(0.5, 200), (0.1, 300), (0.9, 300)])
    def test_covers_at_least_nominal(
        self, p_true: float, n: int, n_replications: int
    ) -> None:
        coverage, _ = _bernoulli_coverage(
            p_true, n, min(n_replications, 300), "betting"
        )
        assert coverage >= 0.95

    def test_anytime_variant_also_covers(self, n_replications: int) -> None:
        coverage, _ = _bernoulli_coverage(
            0.3, 300, min(n_replications, 300), "betting_anytime"
        )
        assert coverage >= 0.95

    def test_anytime_costs_width(self) -> None:
        # Time-uniform validity is not free. If this ever inverts, the two
        # variants have been wired up backwards.
        rng = np.random.default_rng(3)
        x = (rng.random(2_000) < 0.4).astype(np.float64)
        assert betting_ci(x, anytime=True).width > betting_ci(x).width

    def test_beats_normal_on_rare_outcomes(self, n_replications: int) -> None:
        # The case that motivates the whole thing: with a rare outcome the
        # normal approximation quietly under-covers, and thumbs-up rates on
        # routing traffic are often rare.
        reps = min(n_replications, 400)
        betting, _ = _bernoulli_coverage(0.02, 400, reps, "betting", seed=5)
        normal, _ = _bernoulli_coverage(0.02, 400, reps, "normal", seed=5)
        assert betting >= 0.95
        assert normal < betting

    def test_narrows_as_data_accumulates(self) -> None:
        rng = np.random.default_rng(4)
        big = (rng.random(20_000) < 0.5).astype(np.float64)
        assert betting_ci(big[:500]).width > betting_ci(big).width

    def test_rejects_observations_outside_declared_bounds(self) -> None:
        # Validity rests on the bound being real, so an out-of-range observation
        # is an error rather than something to clip away.
        with pytest.raises(ValueError, match="outside the declared bounds"):
            betting_ci(np.array([0.5, 1.5]))

    def test_handles_degenerate_all_zero_and_all_one(self) -> None:
        for x, truth in ((np.zeros(200), 0.0), (np.ones(200), 1.0)):
            interval = betting_ci(x)
            assert interval.covers(truth)
            assert 0.0 <= interval.low <= interval.high <= 1.0

    def test_rescales_to_arbitrary_bounds(self) -> None:
        # Importance-weighted scores live on [0, w_max], not [0, 1].
        rng = np.random.default_rng(6)
        x = rng.random(1_000) * 8.0
        interval = betting_ci(x, lower=0.0, upper=8.0)
        assert interval.covers(4.0)
        assert interval.method == "betting"


class TestBootstrapInterval:
    def test_covers_a_skewed_mean(self, n_replications: int) -> None:
        # Cost per request is lognormal-ish; this is the path it takes.
        rng = np.random.default_rng(7)
        truth = float(np.exp(0.5))
        covered = 0
        reps = min(n_replications, 200)
        for _ in range(reps):
            x = rng.lognormal(0.0, 1.0, 2_000)
            covered += bootstrap_ci(
                x, n_resamples=399, seed=int(rng.integers(1_000_000))
            ).covers(truth)
        assert covered / reps >= 0.88

    def test_needs_more_than_one_observation(self) -> None:
        with pytest.raises(ValueError, match="at least two observations"):
            bootstrap_ci(np.array([1.0]))
