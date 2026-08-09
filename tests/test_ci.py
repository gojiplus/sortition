"""Confidence intervals must cover at the rate they advertise.

Coverage is the only property that matters here, and it is only observable over
replications: a single interval either contains the truth or does not, which
says nothing. Each test below re-runs the whole experiment many times and counts.

**Every tolerance here comes from the replicate count.** These assertions used to
read ``coverage >= 0.95`` and ``covered / reps >= 0.88``, neither of which
measures anything: a correct nominal-95% interval lands below 0.95 about half the
time at any replicate count, and 0.88 is loose enough that a badly broken
bootstrap clears it. :func:`simcheck.binomial_band` gives the interval a
well-calibrated rate should fall in given how many replicates were run, so the
same line of test code tightens automatically when the deep tier raises the
count, and cannot be quietly weakened by lowering it.

**And the intervals' width is gated, not just their coverage.** A one-sided
floor -- which is the right shape for the betting interval, see below -- passes
an interval so wide it always covers.
:func:`simcheck.assert_intervals_informative` is what stops that, and it needs
the endpoints rather than a hit count, which is why the study below records them.
"""

from __future__ import annotations

import numpy as np
import pytest
from simcheck import (
    Estimate,
    MonteCarloResult,
    assert_count_rate,
    assert_intervals_informative,
    binomial_band,
    monte_carlo,
)

from sortition.eval.ci import Interval, betting_ci, bootstrap_ci, normal_ci


def _interval(
    x: np.ndarray, method: str, alpha: float, rng: np.random.Generator
) -> Interval:
    """The interval a named method produces for the mean of ``x``."""
    if method == "betting":
        return betting_ci(x, alpha=alpha)
    if method == "betting_anytime":
        return betting_ci(x, alpha=alpha, anytime=True)
    if method == "bootstrap":
        return bootstrap_ci(
            x, alpha=alpha, n_resamples=399, seed=int(rng.integers(1_000_000))
        )
    return normal_ci(x, alpha=alpha)


def _bernoulli_study(
    p_true: float,
    n: int,
    reps: int,
    method: str,
    *,
    alpha: float = 0.05,
    seed: int = 0,
) -> MonteCarloResult:
    """Refit an interval for a Bernoulli mean over many draws and record it.

    The endpoints are recorded, not just whether they covered, because a
    one-sided coverage floor is gameable without them: an interval wide enough
    to cover every time clears any floor, and coverage alone cannot tell
    conservatism from vacuity. ``assert_intervals_informative`` reads the widths.

    No standard error is reported. These intervals are not symmetric around the
    estimate -- the betting interval least of all -- so halving a width and
    calling it a standard error would be inventing a number the method never
    produced.

    Replicate ``i`` depends on ``(seed, i)`` alone here, where the loop this
    replaces drew every replicate from one shared stream, so raising the
    replicate count now extends the study instead of redrawing it.
    """

    def replicate(rng: np.random.Generator) -> Estimate:
        x = (rng.random(n) < p_true).astype(np.float64)
        interval = _interval(x, method, alpha, rng)
        return Estimate(value=float(x.mean()), lower=interval.low, upper=interval.high)

    return monte_carlo(replicate, truth=p_true, reps=reps, seed=seed)


class TestBettingInterval:
    """The betting interval's guarantee is one-sided, and the gate matches it.

    ``betting_ci`` is a confidence *sequence*: a nonnegative-martingale
    construction whose guarantee is coverage at least the nominal level
    simultaneously at every sample size, not equal to it at any one. Conservatism
    is the design, so :func:`simcheck.assert_coverage`, which bands the rate on
    both sides of nominal, is the wrong gate here -- it would fail the interval
    for honouring its own guarantee. Measured coverage at these settings runs
    0.98 to 1.00 against a nominal 0.95.

    What the floor is, though, still comes from the replicate count:
    :func:`simcheck.binomial_band`'s lower edge, which is where a rate consistent
    with 0.95 stops being consistent with it at this many replicates.

    The one thing a one-sided floor cannot see is an interval so wide it always
    covers, which is exactly how a conservative method could be broken without
    any coverage assertion noticing. That gap is closed by
    :func:`simcheck.assert_intervals_informative`, which fails only when *both*
    the interval is far wider than this estimator's own measured spread requires
    *and* the study never once saw it miss -- the conjunction being what keeps it
    off a correct conservative procedure. Measured width ratios here are 1.3 to
    1.6 times a calibrated normal interval, against a vacuity threshold of 1.78
    at 100 replicates and 1.92 at 300.
    """

    @pytest.mark.parametrize(("p_true", "n"), [(0.5, 200), (0.1, 300), (0.9, 300)])
    def test_covers_at_least_nominal(
        self, p_true: float, n: int, n_replications: int
    ) -> None:
        reps = min(n_replications, 300)
        study = _bernoulli_study(p_true, n, reps, "betting")
        floor, _ = binomial_band(0.95, reps)
        assert study.coverage >= floor
        assert_intervals_informative(study, 0.95, f"betting at p={p_true}")

    def test_anytime_variant_also_covers(self, n_replications: int) -> None:
        reps = min(n_replications, 300)
        study = _bernoulli_study(0.3, 300, reps, "betting_anytime")
        floor, _ = binomial_band(0.95, reps)
        assert study.coverage >= floor
        assert_intervals_informative(study, 0.95, "anytime betting at p=0.3")

    def test_anytime_costs_width(self) -> None:
        # Time-uniform validity is not free. If this ever inverts, the two
        # variants have been wired up backwards. A single pair of fits rather
        # than a study: there is nothing Monte Carlo about it, and no tolerance
        # to derive -- the two widths are computed from the same sample and one
        # must simply exceed the other.
        rng = np.random.default_rng(3)
        x = (rng.random(2_000) < 0.4).astype(np.float64)
        assert betting_ci(x, anytime=True).width > betting_ci(x).width

    def test_beats_normal_on_rare_outcomes(self, n_replications: int) -> None:
        # The case that motivates the whole thing: with a rare outcome the
        # normal approximation quietly under-covers, and thumbs-up rates on
        # routing traffic are often rare.
        reps = min(n_replications, 400)
        betting = _bernoulli_study(0.02, 400, reps, "betting", seed=5)
        normal = _bernoulli_study(0.02, 400, reps, "normal", seed=5)
        floor, _ = binomial_band(0.95, reps)
        assert betting.coverage >= floor
        # Left as a relative comparison rather than `normal.coverage < floor`.
        # The normal interval's true coverage here is around 0.90, and separating
        # 0.90 from 0.95 takes a few hundred replicates: at the fast tier's 100
        # the study measures 0.96 and a band-based negative control would fail on
        # noise. Asserting it only where it resolves would make the test
        # conditional on the tier, which is worse than a weaker claim that always
        # holds.
        assert normal.coverage < betting.coverage
        # No width gate on this one. At p=0.02 the betting interval runs 1.77
        # times calibrated at 100 replicates against a threshold of 1.78, which
        # is inside the noise of the ratio itself; asserting there would be
        # asserting the seed.

    def test_narrows_as_data_accumulates(self) -> None:
        # Also a single pair of fits, for the same reason as
        # test_anytime_costs_width.
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
        #
        # Unlike the betting interval, the percentile bootstrap claims coverage
        # *equal* to nominal rather than at least it, so the gate is two-sided.
        # The old `>= 0.88` allowed the interval to be 7 points overconfident
        # without complaint, which for a metric people set budgets on is the
        # error that matters.
        rng = np.random.default_rng(7)
        truth = float(np.exp(0.5))
        covered = 0
        reps = min(n_replications, 200)
        for _ in range(reps):
            x = rng.lognormal(0.0, 1.0, 2_000)
            covered += bootstrap_ci(
                x, n_resamples=399, seed=int(rng.integers(1_000_000))
            ).covers(truth)
        assert_count_rate(covered, reps, 0.95, label="bootstrap, lognormal mean")

    def test_needs_more_than_one_observation(self) -> None:
        with pytest.raises(ValueError, match="at least two observations"):
            bootstrap_ci(np.array([1.0]))
