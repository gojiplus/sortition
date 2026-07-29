"""The questions teams actually ask, answered from a log table.

Three of them:

    How is the router doing?              evaluate(logs, target=<the logged policy>)
    What would X have cost?               evaluate(logs, target="always:premium")
    Is B better than A?                   compare(logs, a=..., b=...)

The third is the one that justifies the project, and it is the one no other
open-source router can answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from sortition.eval.estimators import (
    Estimate,
    EstimatorName,
    estimate,
    importance_weights,
    scores_for,
)
from sortition.eval.outcome_model import fit_outcome_model
from sortition.frame import EvalArrays, to_arrays
from sortition.targets import TargetPolicy, parse_target

if TYPE_CHECKING:
    import polars as pl

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

# Metrics known to lie in [0, 1]; everything else gets the bootstrap.
BOUNDED_METRICS = frozenset({"outcome", "reward", "thumbs", "resolved", "success"})


def _as_arrays(logs: pl.DataFrame | EvalArrays) -> EvalArrays:
    return logs if isinstance(logs, EvalArrays) else to_arrays(logs)


def _resolve(target: str | TargetPolicy) -> TargetPolicy:
    return parse_target(target) if isinstance(target, str) else target


@dataclass(frozen=True)
class _Prepared:
    """One metric's rows, filtered and fitted, ready for any target policy.

    Shared between ``evaluate`` and ``compare`` so that both see the same rows
    and the same outcome model. If they refit independently, two estimates that
    are supposed to be comparable are not.
    """

    data: EvalArrays
    metric: str
    estimator: EstimatorName
    values: FloatArray
    action: IntArray
    propensity: FloatArray
    eligible: BoolArray
    observed: BoolArray
    q_hat: FloatArray | None

    @property
    def bounded(self) -> bool:
        return self.metric in BOUNDED_METRICS

    def probabilities(self, policy: TargetPolicy) -> FloatArray:
        probs = policy.probabilities(self.data.features, self.data.eligible, self.data.arms)
        return probs if self.observed.all() else probs[self.observed]


def _prepare(data: EvalArrays, metric: str, estimator: EstimatorName, seed: int) -> _Prepared:
    if metric not in data.metrics:
        raise ValueError(f"metric {metric!r} is not in the log; available: {sorted(data.metrics)}")
    values = data.metrics[metric]
    observed = ~np.isnan(values)

    action, propensity, eligible = data.action, data.propensity, data.eligible
    contexts = data.contexts
    if not observed.all():
        # Rows whose outcome never arrived carry no information about this
        # metric. Dropping them is right for late-arriving labels; it would be
        # wrong if missingness depended on the arm, which the diagnostics cannot
        # see and the caller should think about.
        action, propensity = action[observed], propensity[observed]
        eligible = eligible[observed]
        values, contexts = values[observed], contexts[observed]

    q_hat = None
    if estimator in ("dm", "dr", "switch_dr", "dr_os"):
        if contexts.shape[1] == 0:
            if estimator == "dm":
                raise ValueError("the direct method needs numeric features, and this log has none")
            # Without features there is nothing to fit an outcome model on, and
            # a DR estimate with a constant outcome model is IPS with extra steps.
            estimator = "ips"
        else:
            q_hat = fit_outcome_model(
                contexts, action, values, len(data.arms), seed=seed, cross_fit=True
            )
            if metric in BOUNDED_METRICS:
                # Clip here rather than leaving it to the estimator, so that
                # anything reading these scores directly -- the paired
                # difference in `compare`, for one -- sees exactly the model the
                # marginal estimates were built from.
                q_hat = np.clip(q_hat, 0.0, 1.0)

    return _Prepared(
        data=data,
        metric=metric,
        estimator=estimator,
        values=values,
        action=action,
        propensity=propensity,
        eligible=eligible,
        observed=observed,
        q_hat=q_hat,
    )


def _estimate_from(
    prepared: _Prepared,
    policy: TargetPolicy,
    *,
    alpha: float,
    anytime: bool,
    seed: int,
) -> Estimate:
    return estimate(
        action=prepared.action,
        reward=prepared.values,
        propensity=prepared.propensity,
        target_probs=prepared.probabilities(policy),
        q_hat=prepared.q_hat,
        estimator=prepared.estimator,
        metric=prepared.metric,
        bounded=prepared.bounded,
        eligible=prepared.eligible,
        n_excluded_leakage=prepared.data.n_excluded_leakage,
        alpha=alpha,
        anytime=anytime,
        seed=seed,
    )


def evaluate(
    logs: pl.DataFrame | EvalArrays,
    target: str | TargetPolicy,
    *,
    metric: str = "outcome",
    estimator: EstimatorName = "dr",
    alpha: float = 0.05,
    anytime: bool = False,
    seed: int = 0,
) -> Estimate:
    """Estimate what ``metric`` would have been under ``target``."""
    data = _as_arrays(logs)
    prepared = _prepare(data, metric, estimator, seed)
    return _estimate_from(prepared, _resolve(target), alpha=alpha, anytime=anytime, seed=seed)


@dataclass(frozen=True)
class Comparison:
    """Two policies, side by side, on the same logged traffic."""

    metric: str
    a_name: str
    b_name: str
    a: Estimate
    b: Estimate
    difference: float
    difference_interval: tuple[float, float]

    @property
    def relative_change(self) -> float:
        return self.difference / self.a.value if self.a.value else float("nan")

    @property
    def significant(self) -> bool:
        low, high = self.difference_interval
        return low > 0.0 or high < 0.0

    @property
    def trustworthy(self) -> bool:
        return self.a.trustworthy and self.b.trustworthy

    def __str__(self) -> str:
        low, high = self.difference_interval
        verdict = "" if self.significant else "  (not distinguishable from zero)"
        text = (
            f"{self.metric}: {self.b_name} vs {self.a_name}\n"
            f"  {self.a_name:>24} = {self.a.value:.6g}\n"
            f"  {self.b_name:>24} = {self.b.value:.6g}\n"
            f"  {'difference':>24} = {self.difference:+.6g} "
            f"[{low:+.6g}, {high:+.6g}]{verdict}"
        )
        if not self.trustworthy:
            text += "\n  NOT TRUSTWORTHY -- overlap is insufficient for at least one policy"
        return text


def compare(
    logs: pl.DataFrame | EvalArrays,
    a: str | TargetPolicy,
    b: str | TargetPolicy,
    *,
    metrics: tuple[str, ...] = ("outcome", "cost_usd"),
    estimator: EstimatorName = "dr",
    alpha: float = 0.05,
    n_resamples: int = 1999,
    seed: int = 0,
) -> list[Comparison]:
    """Compare two target policies on the same logged traffic.

    The interval is on the *difference*, from a paired bootstrap over rows rather
    than from the two marginal intervals. Both policies are evaluated on the same
    logged rows, so their errors are strongly correlated; treating them as
    independent would badly overstate the uncertainty on the gap, which is the
    only quantity anyone acts on.

    The difference and its interval are both built from the same per-row scores
    as the marginal estimates. Mixing sources -- a DR point estimate with an IPS
    interval, say -- produces the specific embarrassment of a reported difference
    that falls outside its own confidence interval.
    """
    from scipy.stats import bootstrap

    data = _as_arrays(logs)
    policy_a, policy_b = _resolve(a), _resolve(b)

    results: list[Comparison] = []
    for metric in metrics:
        if metric not in data.metrics:
            continue
        prepared = _prepare(data, metric, estimator, seed)
        est_a = _estimate_from(prepared, policy_a, alpha=alpha, anytime=False, seed=seed)
        est_b = _estimate_from(prepared, policy_b, alpha=alpha, anytime=False, seed=seed)

        def row_scores(policy: TargetPolicy, prepared: _Prepared = prepared) -> FloatArray | None:
            return scores_for(
                prepared.estimator,
                action=prepared.action,
                reward=prepared.values,
                propensity=prepared.propensity,
                target_probs=prepared.probabilities(policy),
                q_hat=prepared.q_hat,
            )

        scores_a = row_scores(policy_a)
        scores_b = row_scores(policy_b)

        if scores_a is None or scores_b is None:
            # SNIPS is a ratio with no per-row score, so the difference is
            # bootstrapped by resampling row indices and recomputing both.
            difference, low, high = _snips_difference(
                prepared, policy_a, policy_b, alpha=alpha, n_resamples=n_resamples, seed=seed
            )
        else:
            paired = scores_b - scores_a
            # By construction mean(paired) == est_b.value - est_a.value.
            difference = float(paired.mean())
            interval = bootstrap(
                (paired,),
                np.mean,
                confidence_level=1.0 - alpha,
                n_resamples=n_resamples,
                method="percentile",
                rng=np.random.default_rng(seed),
            )
            low = float(interval.confidence_interval.low)
            high = float(interval.confidence_interval.high)

        results.append(
            Comparison(
                metric=metric,
                a_name=policy_a.name,
                b_name=policy_b.name,
                a=est_a,
                b=est_b,
                difference=difference,
                difference_interval=(low, high),
            )
        )
    if not results:
        raise ValueError(f"none of {list(metrics)} are present in the log")
    return results


def _snips_difference(
    prepared: _Prepared,
    policy_a: TargetPolicy,
    policy_b: TargetPolicy,
    *,
    alpha: float,
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap the gap between two self-normalized estimates.

    SNIPS is a ratio, so there is no per-row score to difference. Row indices are
    resampled instead and both ratios recomputed on the same resample, which
    keeps the pairing that makes the interval on the gap tight.
    """
    from scipy.stats import bootstrap

    weights_a = importance_weights(
        prepared.action, prepared.propensity, prepared.probabilities(policy_a)
    )
    weights_b = importance_weights(
        prepared.action, prepared.propensity, prepared.probabilities(policy_b)
    )
    values = prepared.values

    def gap(idx: NDArray[np.int64]) -> NDArray[np.float64]:
        wa, wb = weights_a[idx], weights_b[idx]
        v = values[idx]
        return np.asarray(
            (wb * v).sum(axis=-1) / wb.sum(axis=-1) - (wa * v).sum(axis=-1) / wa.sum(axis=-1)
        )

    rows = np.arange(prepared.action.shape[0])
    result = bootstrap(
        (rows,),
        gap,
        confidence_level=1.0 - alpha,
        n_resamples=n_resamples,
        method="percentile",
        rng=np.random.default_rng(seed),
    )
    return (
        float(gap(rows)),
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    )


def doctor(logs: pl.DataFrame | EvalArrays, target: str | TargetPolicy = "uniform") -> str:
    """Report whether a log can support counterfactual claims at all.

    Run this before believing any estimate. It answers the prior question -- is
    there enough overlap and exploration here to learn anything? -- which a point
    estimate will never volunteer.
    """
    data = _as_arrays(logs)
    policy = _resolve(target)
    probs = policy.probabilities(data.features, data.eligible, data.arms)
    weights = probs[np.arange(data.n), data.action] / data.propensity

    from sortition.eval.diagnostics import compute_diagnostics

    diagnostics = compute_diagnostics(
        weights,
        target_probs=probs,
        eligible=data.eligible,
        n_excluded_leakage=data.n_excluded_leakage,
    )

    deterministic = float((data.propensity >= 1.0 - 1e-9).mean())
    lines = [
        f"log: {data.n} evaluable rows, {len(data.arms)} arms {list(data.arms)}",
        f"against target: {policy.name}",
        diagnostics.explain(),
        f"deterministic decisions: {deterministic:.1%} of rows had propensity 1.0",
    ]
    if deterministic > 0.99:
        lines.append(
            "WARNING: this log is effectively unexplored. Every request went to "
            "the arm the policy already preferred, so it can confirm what that "
            "policy does and nothing else. Set an exploration floor."
        )
    if data.n_excluded_missing:
        lines.append(f"note: {data.n_excluded_missing} rows lacked a propensity and were dropped")
    return "\n".join(lines)


def to_dicts(comparisons: list[Comparison]) -> list[dict[str, Any]]:
    """Flatten comparisons for a DataFrame or a JSON report."""
    return [
        {
            "metric": c.metric,
            "policy_a": c.a_name,
            "policy_b": c.b_name,
            "value_a": c.a.value,
            "value_b": c.b.value,
            "difference": c.difference,
            "ci_low": c.difference_interval[0],
            "ci_high": c.difference_interval[1],
            "relative_change": c.relative_change,
            "significant": c.significant,
            "trustworthy": c.trustworthy,
            "n": c.a.n,
        }
        for c in comparisons
    ]
