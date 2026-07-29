"""Off-policy estimators for logged bandit feedback.

Given a log written under a behavior policy pi_b, each estimator answers: what
would metric M have been under a different policy pi_t?

The family trades bias against variance:

    IPS       unbiased, variance explodes when the policies disagree
    SNIPS     self-normalized; slight bias, far steadier
    DM        pure outcome model; low variance, biased if the model is wrong
    DR        DM plus an IPS correction on its residuals -- the default
    switch-DR DR where the weight is small, DM where it is not
    DR-os     DR with weights shrunk toward zero to bound the variance

DR is the default because it is consistent if *either* the propensities or the
outcome model are right, and routing logs usually have exact propensities (the
sampler recorded them) with a mediocre outcome model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from sortition.eval.ci import Interval, betting_ci, bootstrap_ci, normal_ci
from sortition.eval.diagnostics import Diagnostics, compute_diagnostics

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

EstimatorName = Literal["ips", "snips", "dm", "dr", "switch_dr", "dr_os"]
CIMethod = Literal["betting", "bootstrap", "normal", "none"]


@dataclass(frozen=True)
class Estimate:
    """A policy value estimate, with everything needed to judge it."""

    value: float
    estimator: EstimatorName
    metric: str
    n: int
    se: float
    interval: Interval | None
    diagnostics: Diagnostics

    @property
    def trustworthy(self) -> bool:
        return self.diagnostics.overlap_ok

    def __str__(self) -> str:
        head = f"{self.metric} under target policy: {self.value:.6g}"
        if self.interval is not None:
            head += (
                f"  [{self.interval.low:.6g}, {self.interval.high:.6g}] "
                f"{self.interval.level:.0%} {self.interval.method}"
            )
        head += f"\n  estimator={self.estimator}  n={self.n}  se={self.se:.4g}"
        if not self.trustworthy:
            head += "\n  NOT TRUSTWORTHY -- see diagnostics"
        if self.diagnostics.warnings:
            head += "\n" + "\n".join(f"  ! {w}" for w in self.diagnostics.warnings)
        return head


def importance_weights(
    action: IntArray,
    propensity: FloatArray,
    target_probs: FloatArray,
) -> FloatArray:
    """``pi_t(a|x) / pi_b(a|x)`` for the action actually taken on each row."""
    if np.any(propensity <= 0.0):
        raise ValueError(
            "propensity must be strictly positive; a zero-probability action "
            "cannot have been logged, and would give an infinite weight"
        )
    rows = np.arange(action.shape[0])
    return np.asarray(target_probs[rows, action] / propensity, dtype=np.float64)


def _taken(values: FloatArray, action: IntArray) -> FloatArray:
    return np.asarray(values[np.arange(action.shape[0]), action], dtype=np.float64)


def ips_scores(
    action: IntArray,
    reward: FloatArray,
    propensity: FloatArray,
    target_probs: FloatArray,
) -> FloatArray:
    """Per-row IPS contributions. Unbiased for the target policy value."""
    return importance_weights(action, propensity, target_probs) * reward


def dm_scores(target_probs: FloatArray, q_hat: FloatArray) -> FloatArray:
    """Per-row direct-method contributions: the model's own expectation."""
    return np.asarray((target_probs * q_hat).sum(axis=1), dtype=np.float64)


def dr_scores(
    action: IntArray,
    reward: FloatArray,
    propensity: FloatArray,
    target_probs: FloatArray,
    q_hat: FloatArray,
) -> FloatArray:
    """Doubly robust contributions.

    ``q_hat`` must be fit out-of-fold. An outcome model that has seen the row it
    is predicting makes the residual too small, the correction term too quiet,
    and the estimate biased toward the model -- silently, since the estimator
    reports nothing unusual.
    """
    weights = importance_weights(action, propensity, target_probs)
    baseline = dm_scores(target_probs, q_hat)
    residual = reward - _taken(q_hat, action)
    return np.asarray(baseline + weights * residual, dtype=np.float64)


def switch_dr_scores(
    action: IntArray,
    reward: FloatArray,
    propensity: FloatArray,
    target_probs: FloatArray,
    q_hat: FloatArray,
    *,
    tau: float = 10.0,
) -> FloatArray:
    """DR where the importance weight is below ``tau``, direct method above it.

    Trades a little bias for a bounded contribution from any single row.
    """
    weights = importance_weights(action, propensity, target_probs)
    baseline = dm_scores(target_probs, q_hat)
    residual = reward - _taken(q_hat, action)
    correction = np.where(weights <= tau, weights * residual, 0.0)
    return np.asarray(baseline + correction, dtype=np.float64)


def dr_os_scores(
    action: IntArray,
    reward: FloatArray,
    propensity: FloatArray,
    target_probs: FloatArray,
    q_hat: FloatArray,
    *,
    lam: float = 100.0,
) -> FloatArray:
    """DR with optimistic shrinkage (Su et al., 2020).

    Replaces ``w`` with ``lam * w / (w^2 + lam)``, which is near-identity for
    small weights and decays for large ones -- a smooth version of the switch,
    with no threshold to choose.
    """
    weights = importance_weights(action, propensity, target_probs)
    shrunk = lam * weights / (weights**2 + lam)
    baseline = dm_scores(target_probs, q_hat)
    residual = reward - _taken(q_hat, action)
    return np.asarray(baseline + shrunk * residual, dtype=np.float64)


def snips_value(
    action: IntArray,
    reward: FloatArray,
    propensity: FloatArray,
    target_probs: FloatArray,
) -> float:
    """Self-normalized IPS: ``sum(w r) / sum(w)``.

    A ratio, not an average of per-row scores, so it has no score vector and its
    interval comes from the bootstrap. Slightly biased, but immune to the
    failure where the weights happen to sum to much more or less than n.
    """
    weights = importance_weights(action, propensity, target_probs)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("importance weights sum to zero; no overlap at all")
    return float((weights * reward).sum() / total)


def scores_for(
    estimator: EstimatorName,
    *,
    action: IntArray,
    reward: FloatArray,
    propensity: FloatArray,
    target_probs: FloatArray,
    q_hat: FloatArray | None = None,
) -> FloatArray | None:
    """Per-row contributions for ``estimator``, or ``None`` for SNIPS.

    Exposed because any quantity derived from an estimate -- a difference
    between two policies, a subgroup breakdown -- has to be built from the same
    per-row scores as the estimate itself. Deriving a point estimate one way and
    its interval another produces the failure where the reported difference sits
    outside its own confidence interval.
    """
    if estimator == "snips":
        return None
    if estimator == "ips":
        return ips_scores(action, reward, propensity, target_probs)
    if q_hat is None:
        raise ValueError(f"estimator {estimator!r} needs an out-of-fold q_hat")
    if estimator == "dm":
        return dm_scores(target_probs, q_hat)
    if estimator == "dr":
        return dr_scores(action, reward, propensity, target_probs, q_hat)
    if estimator == "switch_dr":
        return switch_dr_scores(action, reward, propensity, target_probs, q_hat)
    if estimator == "dr_os":
        return dr_os_scores(action, reward, propensity, target_probs, q_hat)
    raise ValueError(f"unknown estimator {estimator!r}")


def _score_bounds(
    estimator: EstimatorName,
    weight_bound: float,
    reward_lo: float,
    reward_hi: float,
) -> tuple[float, float]:
    """Worst-case range of a per-row score, for the betting interval.

    ``weight_bound`` must come from a design parameter of the logging policy --
    an exploration floor gives ``w <= |A| / epsilon`` -- not from the observed
    maximum, which would make the bound depend on the data it is used to
    analyze.
    """
    if estimator == "dm":
        return reward_lo, reward_hi
    if estimator == "ips":
        return weight_bound * reward_lo, weight_bound * reward_hi
    # DR-family: a direct-method baseline in [lo, hi] plus a weighted residual
    # spanning +/- weight_bound * (hi - lo).
    span = weight_bound * (reward_hi - reward_lo)
    return reward_lo - span, reward_hi + span


def estimate(
    *,
    action: IntArray,
    reward: FloatArray,
    propensity: FloatArray,
    target_probs: FloatArray,
    q_hat: FloatArray | None = None,
    estimator: EstimatorName = "dr",
    metric: str = "outcome",
    bounded: bool = True,
    reward_bounds: tuple[float, float] = (0.0, 1.0),
    eligible: BoolArray | None = None,
    n_excluded_leakage: int = 0,
    alpha: float = 0.05,
    ci_method: CIMethod | None = None,
    anytime: bool = False,
    weight_bound: float | None = None,
    seed: int = 0,
) -> Estimate:
    """Estimate the value of a target policy from logged bandit feedback.

    ``bounded`` selects the interval: bounded outcomes get the betting interval,
    unbounded metrics such as cost and latency get the bootstrap.

    Diagnostics are computed regardless and attached to the result. When overlap
    fails, the estimate is still returned but marked untrustworthy -- callers
    that want a hard refusal should check ``.trustworthy``.
    """
    action = np.asarray(action, dtype=np.int64)
    reward = np.asarray(reward, dtype=np.float64)
    propensity = np.asarray(propensity, dtype=np.float64)
    target_probs = np.asarray(target_probs, dtype=np.float64)

    n = int(action.shape[0])
    if n == 0:
        raise ValueError("no rows to estimate from")
    if target_probs.shape[0] != n:
        raise ValueError(f"target_probs has {target_probs.shape[0]} rows, expected {n}")

    needs_model = estimator in ("dm", "dr", "switch_dr", "dr_os")
    if needs_model and q_hat is None:
        raise ValueError(f"estimator {estimator!r} needs an out-of-fold q_hat")
    if q_hat is not None:
        q_hat = np.asarray(q_hat, dtype=np.float64)
        if bounded:
            # A regression will happily predict a thumbs-up rate of 1.08. When
            # the outcome is known to be bounded, clipping is a free improvement
            # -- the truth is never outside the range, so a clipped prediction is
            # never further from it -- and it keeps the DM score inside the
            # bounds the interval is entitled to assume.
            q_hat = np.clip(q_hat, reward_bounds[0], reward_bounds[1])

    weights = importance_weights(action, propensity, target_probs)
    diagnostics = compute_diagnostics(
        weights,
        target_probs=target_probs,
        eligible=eligible,
        n_excluded_leakage=n_excluded_leakage,
    )

    scores = scores_for(
        estimator,
        action=action,
        reward=reward,
        propensity=propensity,
        target_probs=target_probs,
        q_hat=q_hat,
    )

    if scores is not None:
        value = float(scores.mean())
        se = float(scores.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    else:
        value = snips_value(action, reward, propensity, target_probs)
        # A ratio estimator has no per-row score; the bootstrap resamples rows
        # and recomputes the ratio, which the generic path cannot express.
        se = float("nan")

    if ci_method is None:
        ci_method = "betting" if (bounded and scores is not None) else "bootstrap"

    interval = _build_interval(
        scores=scores,
        estimator=estimator,
        ci_method=ci_method,
        bounded=bounded,
        reward_bounds=reward_bounds,
        weights=weights,
        weight_bound=weight_bound,
        propensity=propensity,
        reward=reward,
        alpha=alpha,
        anytime=anytime,
        seed=seed,
    )
    if scores is None and interval is not None:
        se = interval.width / 4.0  # rough, from the interval rather than scores

    return Estimate(
        value=value,
        estimator=estimator,
        metric=metric,
        n=n,
        se=se,
        interval=interval,
        diagnostics=diagnostics,
    )


def _build_interval(
    *,
    scores: FloatArray | None,
    estimator: EstimatorName,
    ci_method: CIMethod,
    bounded: bool,
    reward_bounds: tuple[float, float],
    weights: FloatArray,
    weight_bound: float | None,
    propensity: FloatArray,
    reward: FloatArray,
    alpha: float,
    anytime: bool,
    seed: int,
) -> Interval | None:
    if ci_method == "none":
        return None

    if scores is None:
        # SNIPS: resample rows and recompute the ratio.
        from scipy.stats import bootstrap

        numer = weights * reward

        def ratio(idx: NDArray[np.int64]) -> NDArray[np.float64]:
            return np.asarray(numer[idx].sum(axis=-1) / weights[idx].sum(axis=-1))

        result = bootstrap(
            (np.arange(weights.shape[0]),),
            ratio,
            confidence_level=1.0 - alpha,
            n_resamples=1999,
            method="percentile",
            rng=np.random.default_rng(seed),
        )
        return Interval(
            low=float(result.confidence_interval.low),
            high=float(result.confidence_interval.high),
            level=1.0 - alpha,
            method="bootstrap",
        )

    if ci_method == "normal":
        return normal_ci(scores, alpha=alpha)
    if ci_method == "bootstrap":
        return bootstrap_ci(scores, alpha=alpha, seed=seed)

    if not bounded:
        raise ValueError("the betting interval requires a bounded metric")
    bound = weight_bound if weight_bound is not None else float(1.0 / propensity.min())
    lo, hi = _score_bounds(estimator, bound, *reward_bounds)
    return betting_ci(scores, alpha=alpha, lower=lo, upper=hi, anytime=anytime)
