"""Confidence intervals for off-policy estimates.

Two regimes, because routing metrics come in two shapes.

*Bounded* outcomes -- thumbs, resolution flags, graded eval scores -- get the
hedged betting interval of Waudby-Smith and Ramdas (2023). It is tighter than a
normal approximation at realistic sample sizes and, being derived from a
nonnegative martingale, stays valid at arbitrary stopping times. That property is
not academic here: teams watch routing dashboards daily and act when a number
looks good, which is precisely the optional stopping that invalidates a fixed-n
interval.

*Unbounded* metrics -- cost, latency -- get a bootstrap, because no
[0, 1] rescaling exists for them. Their importance-weighted scores are heavy
tailed, so the caller is expected to winsorize first and read the reported bias
bound alongside the interval.

``confseq``, the reference implementation from the same authors, is not a usable
dependency: it last shipped in January 2023 with wheels up to cp310 and requires
Boost headers to build from source. The betting interval below is validated by
coverage simulation instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

# Keeps every capital-process factor strictly positive: with lambda truncated to
# c/m, the smallest factor is 1 - c.
_BET_TRUNCATION = 0.75


@dataclass(frozen=True)
class Interval:
    """A confidence interval and the method that produced it."""

    low: float
    high: float
    level: float
    method: Literal["betting", "bootstrap", "normal"]

    @property
    def width(self) -> float:
        """Distance between the interval endpoints."""
        return self.high - self.low

    def covers(self, value: float) -> bool:
        """Whether ``value`` lies inside the interval.

        Args:
            value: The quantity to test, typically a known ground truth.

        Returns:
            True if the value is within the interval, inclusive.
        """
        return self.low <= value <= self.high


def _predictable_lambdas(x: FloatArray, alpha: float, *, anytime: bool) -> FloatArray:
    """Betting fractions from the predictable plug-in of WSR (2023).

    Each lambda_t may depend only on observations before t, which is what keeps
    the capital process a martingale under the null. The 1/2 and 1/4 priors are
    the standard choice and keep the first few terms finite.

    ``anytime`` selects the guarantee. The time-uniform variant divides by
    ``t log(1+t)`` and is valid at every sample size simultaneously; the
    fixed-n variant divides by ``n`` and is tighter but valid only at the ``n``
    it was computed for.

    The global truncation to ``_BET_TRUNCATION`` is not optional. Without it the
    first few lambdas are O(1) to O(10) -- lambda_1 exceeds 6 at alpha = 0.05 --
    and those opening bets lose enough capital that the interval degenerates
    towards the whole support.
    """
    n = x.shape[0]
    t = np.arange(1, n + 1, dtype=np.float64)

    running_mean = (0.5 + np.cumsum(x)) / (1.0 + t)
    # sigma^2_t uses mu_hat_t, so shift the mean sequence to align with x_t.
    centered = (x - running_mean) ** 2
    running_var = (0.25 + np.cumsum(centered)) / (1.0 + t)

    # lambda_t depends on sigma^2_{t-1}: prepend the prior, drop the last.
    var_prev = np.concatenate(([0.25], running_var[:-1]))
    scale = t * np.log1p(t) if anytime else np.full_like(t, float(n))
    lam = np.sqrt(2.0 * np.log(2.0 / alpha) / (var_prev * scale))
    return np.asarray(np.minimum(lam, _BET_TRUNCATION), dtype=np.float64)


def _log_capital(x: FloatArray, m: float, lam: FloatArray) -> float:
    """Log of the hedged capital process at candidate mean ``m``.

    Two one-sided processes are mixed with equal weight: one that grows when the
    truth exceeds ``m``, one that grows when it falls below. Evidence against
    ``m`` in either direction accumulates capital.
    """
    # Truncating each side keeps every factor at least 1 - _BET_TRUNCATION > 0.
    # At the endpoints one side needs no truncation (the corresponding factor is
    # already >= 1) while the other still does, so both bounds are computed
    # rather than special-cased away -- getting this wrong produces NaNs from
    # log1p of a negative argument exactly at m = 0 and m = 1.
    up_cap = np.inf if m <= 0.0 else _BET_TRUNCATION / m
    down_cap = np.inf if m >= 1.0 else _BET_TRUNCATION / (1.0 - m)
    lam_up = np.minimum(lam, up_cap)
    lam_down = np.minimum(lam, down_cap)

    diff = x - m
    log_up = float(np.log1p(lam_up * diff).sum())
    log_down = float(np.log1p(-lam_down * diff).sum())

    # log(0.5 * exp(a) + 0.5 * exp(b)), stably.
    hi = max(log_up, log_down)
    lo = min(log_up, log_down)
    return hi + float(np.log1p(np.exp(lo - hi))) - float(np.log(2.0))


def betting_ci(
    x: FloatArray,
    *,
    alpha: float = 0.05,
    lower: float = 0.0,
    upper: float = 1.0,
    anytime: bool = False,
    tol: float = 1e-6,
) -> Interval:
    """Hedged betting interval for the mean of observations in ``[lower, upper]``.

    Values outside the declared bounds are a caller error, not something to clip
    silently: the interval's validity rests on the bound being real.

    Set ``anytime=True`` for the time-uniform version, which stays valid however
    long you keep watching and whenever you decide to stop. It is meaningfully
    wider; pay for it only when someone will actually peek.

    The capital process is quasi-convex in the candidate mean with its minimum at
    the sample mean, so each endpoint is found by bisecting outward from there
    rather than by scanning a grid, which would cost an n-by-grid outer product.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        raise ValueError("betting_ci needs at least one observation")
    if upper <= lower:
        raise ValueError(f"upper ({upper}) must exceed lower ({lower})")
    if x.min() < lower - 1e-9 or x.max() > upper + 1e-9:
        raise ValueError(
            f"observations span [{x.min()}, {x.max()}], outside the declared "
            f"bounds [{lower}, {upper}]; the interval would not be valid"
        )

    scale = upper - lower
    z = np.clip((x - lower) / scale, 0.0, 1.0)

    lam = _predictable_lambdas(z, alpha, anytime=anytime)
    threshold = float(np.log(1.0 / alpha))
    center = float(z.mean())

    def rejected(m: float) -> bool:
        return _log_capital(z, m, lam) >= threshold

    def bisect(lo: float, hi: float, *, find_low: bool) -> float:
        # Invariant: the endpoint being sought lies in [lo, hi], with the
        # rejected side at lo (for the lower endpoint) or hi (for the upper).
        for _ in range(200):
            if hi - lo < tol:
                break
            mid = 0.5 * (lo + hi)
            if rejected(mid):
                if find_low:
                    lo = mid
                else:
                    hi = mid
            elif find_low:
                hi = mid
            else:
                lo = mid
        return lo if find_low else hi

    low = 0.0 if not rejected(0.0) else bisect(0.0, center, find_low=True)
    high = 1.0 if not rejected(1.0) else bisect(center, 1.0, find_low=False)

    return Interval(
        low=lower + low * scale,
        high=lower + high * scale,
        level=1.0 - alpha,
        method="betting",
    )


def bootstrap_ci(
    scores: FloatArray,
    *,
    alpha: float = 0.05,
    n_resamples: int = 4999,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap for the mean of per-row estimator scores.

    Uses ``scipy.stats.bootstrap`` rather than a hand-rolled resampler. BCa is
    the usual default but it is unstable when the score distribution is as
    heavy-tailed as importance-weighted cost, where the acceleration term is
    driven by a handful of jackknife points; the percentile method degrades more
    gracefully there.
    """
    from scipy.stats import bootstrap

    scores = np.asarray(scores, dtype=np.float64)
    if scores.size < 2:
        raise ValueError("bootstrap_ci needs at least two observations")

    result = bootstrap(
        (scores,),
        np.mean,
        confidence_level=1.0 - alpha,
        n_resamples=n_resamples,
        method="percentile",
        rng=np.random.default_rng(seed),
    )
    return Interval(
        low=float(result.confidence_interval.low),
        high=float(result.confidence_interval.high),
        level=1.0 - alpha,
        method="bootstrap",
    )


def normal_ci(scores: FloatArray, *, alpha: float = 0.05) -> Interval:
    """Normal-approximation interval. Present as a baseline to compare against.

    Included so the coverage suite can demonstrate what it costs: with heavy
    importance weights this under-covers, which is the failure the betting
    interval exists to avoid.
    """
    from scipy.stats import norm

    scores = np.asarray(scores, dtype=np.float64)
    n = scores.size
    if n < 2:
        raise ValueError("normal_ci needs at least two observations")
    mean = float(scores.mean())
    se = float(scores.std(ddof=1) / np.sqrt(n))
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return Interval(
        low=mean - z * se, high=mean + z * se, level=1.0 - alpha, method="normal"
    )
