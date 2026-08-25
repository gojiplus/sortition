"""Choosing the exchange rate between quality and price, instead of typing it.

``cost_weight`` says how much predicted quality a policy will give up to move a
request to a cheaper arm. Nothing about a routing problem makes that number
guessable in advance: it depends on the price gap between the arms, on how often
the dearer one is actually better, and on the traffic mix. Left as an operator
constant it is either zero -- cost ignored -- or a number nobody can defend.

The sweep replaces the guess with a measurement. It fits the two boosters once,
rescores the tuning log at every weight on the grid, and reports what each point
would have cost and scored, with intervals, using the same estimators as the rest
of the project.

**Why a third split.** The point is chosen on ``tune`` and reported on a holdout
the sweep never saw. Selecting a weight on a split and then quoting that split's
number is the same error :func:`~sortition.train.trainer.train_test_split` exists
to prevent, one level up: the winner of a nine-way search is partly whichever
grid point got luckiest on the rows it was scored on, and with a flat frontier
that luck is most of the reported gain.

**Why the default rule is not "maximize quality minus cost".** Those are in
different units -- a probability and a dollar -- and any single objective smuggles
in an exchange rate, which is the number being chosen.

What the rule does instead is a non-inferiority test. The operator states the
most quality they are willing to spend (``tolerance``), and the sweep returns the
cheapest point the log can *prove* stays inside it: the lower end of the interval
on the paired difference must sit above ``-tolerance``.

**A margin only fires if it exceeds the interval's half-width.** This is the
property to understand before setting one. The rule accepts a point when the
lower end of the interval on the paired difference clears ``-tolerance``, and for
a trade that truly costs nothing the interval is centred on zero -- so its lower
end is minus the half-width, and the margin has to be wider than that. On the
equal-quality problem in ``tests/test_clean_cases.py`` the half-width is 0.034 at
12,000 logged rows, 0.018 at 40,000 and 0.0098 at 120,000, so the default margin
of 0.01 needs a log of several hundred thousand rows before it accepts anything
at all.

That is correct behaviour and it is also easy to mistake for the opposite
conclusion, because a refusal on a thin log looks exactly like a refusal on
traffic where the dear arm is genuinely worth its price.
:attr:`SweepResult.next_best` exists for that: it names the margin that would
have cleared and the saving it would have bought, so the operator is choosing
rather than guessing.

A margin of exactly zero is degenerate and the default is not zero for that
reason. Proving a difference is no worse than zero means an interval whose lower
end reaches zero, and an interval around a true difference of zero straddles it
whatever the sample size. So a margin of zero refuses every trade -- including the
one case it most obviously should take, two arms of identical quality where one
costs ten times more. That is not conservatism, it is a rule that never fires,
and it is what :class:`tests.test_clean_cases.TestEqualQualityDifferentPrice`
exists to catch. Pass ``tolerance=0.0`` deliberately to disable trading.

The obvious alternative -- take the cheapest point not *measurably* worse -- is
wrong in a way that is easy to miss and hard to notice afterwards. It reads
absence of evidence as evidence of absence: on a small log nothing is measurable,
every point passes, and the rule hands back the most aggressive weight on the
grid precisely when there was least reason to trust it. A non-inferiority margin
inverts that. A wide interval fails the test, so too little data produces
timidity rather than confidence.

How much quality a dollar is worth is not in the log and cannot be learned from
it. ``tolerance`` and ``budget`` are the two places that judgment enters, and
they are arguments rather than defaults for that reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from sortition.decide.tree import TreePolicy, discount_matrix
from sortition.exploration import apply_epsilon_floor
from sortition.frame import EvalArrays, to_arrays
from sortition.train.trainer import train

if TYPE_CHECKING:
    import polars as pl

    from sortition.schema import PolicyArtifact

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

DEFAULT_TOLERANCE = 0.01
"""How much of the outcome a trade may cost before the sweep refuses it.

One point of a bounded outcome. Small enough that nobody notices the quality and
large enough that the test can actually pass: at a margin of exactly zero the
rule accepts nothing at all, because proving a difference is not below zero needs
an interval that excludes everything negative, and the interval around a true
difference of zero never does. Zero is available and means "do not trade"."""

DEFAULT_GRID = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
"""Weights to try, in units of the outcome.

Cost is normalized within each request's eligible set, so a weight of 1.0 says
"the dearest eligible arm must be a full point of outcome better than the
cheapest to be worth taking". For a bounded outcome in [0, 1] that is already
decisive, so the grid brackets the interesting range and doubles through it
rather than searching finely somewhere arbitrary.
"""


@dataclass(frozen=True)
class FrontierPoint:
    """One cost weight, and what it would have done to the tuning log."""

    cost_weight: float
    quality: float
    quality_interval: tuple[float, float] | None
    cost: float
    cost_interval: tuple[float, float] | None
    quality_difference: float
    """Estimated quality against the cost-blind policy, paired over rows."""

    quality_difference_interval: tuple[float, float]
    cost_difference: float
    trustworthy: bool

    def non_inferior_at(self, tolerance: float) -> bool:
        """Whether the log proves this point spends less than ``tolerance`` quality.

        The interval is on the paired difference, not on the two marginals: both
        policies are scored on the same rows, so their errors move together and
        treating them as independent would overstate the uncertainty on the only
        quantity anyone acts on.

        Args:
            tolerance: The most quality the operator will spend, as a
                non-negative amount in the units of the outcome.

        Returns:
            True if the whole interval sits above ``-tolerance``.

        Raises:
            ValueError: If ``tolerance`` is negative.
        """
        if tolerance < 0.0:
            raise ValueError(f"tolerance must be non-negative, got {tolerance}")
        return self.trustworthy and self.quality_difference_interval[0] >= -tolerance

    @property
    def saving(self) -> float:
        """Dollars per request saved against the cost-blind policy."""
        return -self.cost_difference

    @property
    def tolerance_required(self) -> float:
        """The smallest margin at which this point would be accepted.

        The interval's lower end, sign-flipped. Reporting it turns a refusal into
        a number the operator can act on: a sweep that silently returns "no
        trade" leaves them unable to tell a log too thin to prove anything from
        traffic where the dear arm is genuinely worth it.
        """
        return max(0.0, -self.quality_difference_interval[0])


@dataclass(frozen=True)
class SweepResult:
    """A frontier over cost weights, and the point selected from it."""

    frontier: tuple[FrontierPoint, ...]
    chosen: FrontierPoint
    epsilon: float
    metric: str
    cost_metric: str
    n_fit: int
    n_tune: int
    feature_spec: tuple[str, ...]
    arms: tuple[str, ...]
    budget: float | None
    tolerance: float
    _policy: TreePolicy

    @property
    def next_best(self) -> FrontierPoint | None:
        """The cheapest point a slightly larger tolerance would have accepted.

        ``None`` when the selected point is already the cheapest on the grid.
        Its :attr:`FrontierPoint.tolerance_required` is what the operator would
        have to accept to take it, and its :attr:`FrontierPoint.saving` is what
        they would get for it.
        """
        cheaper = [p for p in self.frontier if p.cost < self.chosen.cost]
        return min(cheaper, key=lambda p: p.tolerance_required) if cheaper else None

    def policy_at(self, cost_weight: float) -> TreePolicy:
        """The trained policy at one weight.

        Every point on the frontier shares two boosters and differs only in this
        number, which is why the sweep costs one fit rather than one per point.

        Args:
            cost_weight: The weight to set.

        Returns:
            The policy.
        """
        return replace(self._policy, cost_weight=cost_weight)

    @property
    def policy(self) -> TreePolicy:
        """The policy at the selected weight."""
        return self.policy_at(self.chosen.cost_weight)

    def artifact(self, *, name: str | None = None) -> PolicyArtifact:
        """Package the selected policy for deployment.

        Args:
            name: Optional label prefixed to the content hash.

        Returns:
            The versioned artifact.
        """
        from sortition.decide.artifact import build
        from sortition.decide.engine import ExplorationConfig

        return build(self.policy, ExplorationConfig(epsilon=self.epsilon), name=name)


def split_three_ways(
    logs: pl.DataFrame, *, tune: float = 0.2, holdout: float = 0.3, seed: int = 0
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split logs into rows to fit on, rows to tune on, and rows to report on.

    Args:
        logs: The log table.
        tune: Share of rows the sweep may look at when choosing a weight.
        holdout: Share reserved for the final estimate, seen by neither.
        seed: Seed for the split.

    Returns:
        The fitting rows, the tuning rows and the held-out rows.

    Raises:
        ValueError: If the shares are not each in (0, 1) and leave rows to fit on.
    """
    if not 0.0 < tune < 1.0 or not 0.0 < holdout < 1.0:
        raise ValueError(
            f"tune and holdout must be in (0, 1), got {tune} and {holdout}"
        )
    if tune + holdout >= 1.0:
        raise ValueError(
            f"tune ({tune}) and holdout ({holdout}) leave nothing to fit on"
        )
    rng = np.random.default_rng(seed)
    draw = rng.random(logs.height)
    return (
        logs.filter(draw >= tune + holdout),
        logs.filter(draw < tune),
        logs.filter((draw >= tune) & (draw < tune + holdout)),
    )


@dataclass(frozen=True)
class _Precomputed:
    """A target policy whose probabilities were already computed.

    The estimators want a policy they can ask; the sweep has already scored every
    row at every weight in two batched booster calls. This adapter hands the
    array over unchanged, so the frontier goes through exactly the estimator path
    that ``evaluate`` uses rather than a parallel one.
    """

    probs: FloatArray
    name: str

    def probabilities(
        self,
        features: list[dict[str, Any]],
        eligible: BoolArray,
        arms: tuple[str, ...],
    ) -> FloatArray:
        """Return the stored probabilities.

        Args:
            features: Ignored; scoring already happened.
            eligible: Ignored, but its shape must match what was scored.
            arms: Ignored.

        Returns:
            An ``(n, len(arms))`` array whose rows sum to 1.

        Raises:
            ValueError: If the log does not have the shape that was scored.
        """
        if eligible.shape != self.probs.shape:
            raise ValueError(
                f"precomputed probabilities are {self.probs.shape} but the log "
                f"is {eligible.shape}; they describe different rows"
            )
        return self.probs


def probabilities_at(
    quality: FloatArray,
    cost: FloatArray,
    eligible: BoolArray,
    *,
    cost_weight: float,
    epsilon: float,
) -> FloatArray:
    """Action probabilities a tree policy would produce, for a whole log at once.

    Mirrors :class:`~sortition.targets.PolicyTarget` on a ``TreePolicy``: greedy
    over the discounted score, then the shared epsilon floor. Ties break toward
    the arm that sorts last, which is what the per-row path does; a test asserts
    the two agree, because a fast path that quietly disagreed would tune a weight
    for a policy that never ships.

    Args:
        quality: ``(n, K)`` predicted outcomes.
        cost: ``(n, K)`` predicted costs.
        eligible: ``(n, K)`` mask of arms surviving the hard filter.
        cost_weight: The weight to score at.
        epsilon: The exploration floor.

    Returns:
        An ``(n, K)`` array whose rows sum to 1.
    """
    scores = discount_matrix(quality, cost, eligible, cost_weight)
    k = scores.shape[1]
    greedy = np.zeros_like(scores)
    best = (k - 1) - scores[:, ::-1].argmax(axis=1)
    greedy[np.arange(len(scores)), best] = 1.0
    return apply_epsilon_floor(greedy, eligible, epsilon)


def sweep(
    fit_rows: pl.DataFrame,
    tune_rows: pl.DataFrame | EvalArrays,
    *,
    grid: tuple[float, ...] = DEFAULT_GRID,
    metric: str = "outcome",
    cost_metric: str = "cost_usd",
    epsilon: float = 0.05,
    budget: float | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    alpha: float = 0.05,
    name: str | None = None,
    seed: int = 0,
    **booster_kwargs: Any,
) -> SweepResult:
    """Trace the quality/price frontier over cost weights and pick a point on it.

    Args:
        fit_rows: Rows to fit the two boosters on.
        tune_rows: Rows to score the grid on. Must not overlap ``fit_rows``, and
            must not be the rows the result is finally reported on.
        grid: Cost weights to try. Zero -- ignore cost -- is added if absent,
            since every other point is reported as a difference against it.
        metric: The outcome column to trade against price.
        cost_metric: The dollar column.
        epsilon: Exploration floor for the resulting policy.
        budget: Optional ceiling on ``cost_metric`` per request. With one, the
            rule is the best quality that fits, and ``tolerance`` is unused.
        tolerance: The most quality the operator will spend to save money, in the
            units of ``metric``. Zero refuses every trade by construction -- see
            :data:`DEFAULT_TOLERANCE`.
        alpha: One minus the confidence level on every interval.
        name: Label prefixed to the artifact's content hash.
        seed: Seed for the boosters and the bootstrap.
        **booster_kwargs: Passed to ``LGBMRegressor``.

    Returns:
        The frontier and the selected point.

    Raises:
        ValueError: If the log carries no cost column to price arms with, if the
            tuning rows describe a different set of arms than the fitting rows,
            or if no point on the grid comes in under ``budget``.
    """
    from sortition.eval import compare

    fitted = train(
        fit_rows,
        metric=metric,
        cost_metric=cost_metric,
        cost_weight=0.0,
        epsilon=epsilon,
        name=name,
        seed=seed,
        **booster_kwargs,
    )
    policy = fitted.policy
    if policy.cost_booster_text is None:
        raise ValueError(
            f"the fitting rows carry no usable {cost_metric!r} column, so there "
            "is no price to trade quality against and nothing to sweep"
        )

    data = tune_rows if isinstance(tune_rows, EvalArrays) else to_arrays(tune_rows)
    if data.arms != policy.arms:
        raise ValueError(
            f"the tuning rows describe arms {data.arms} but the policy was fitted "
            f"on {policy.arms}; an arm index would not mean the same thing on "
            "both sides"
        )
    if cost_metric not in data.metrics:
        raise ValueError(
            f"the tuning rows carry no {cost_metric!r} column, so the frontier "
            "would have no price axis"
        )

    weights = tuple(sorted({0.0, *grid}))
    # Two booster calls for the whole sweep: the weight changes what is chosen,
    # never what is predicted.
    quality, cost = policy.score_matrix(data.features, data.eligible)
    targets = {
        w: _Precomputed(
            probabilities_at(
                quality, cost, data.eligible, cost_weight=w, epsilon=epsilon
            ),
            name=f"cost_weight={w:g}",
        )
        for w in weights
    }

    frontier = []
    for w in weights:
        by_metric = {
            c.metric: c
            for c in compare(
                data,
                a=targets[0.0],
                b=targets[w],
                metrics=(metric, cost_metric),
                alpha=alpha,
                seed=seed,
            )
        }
        on_quality, on_cost = by_metric[metric], by_metric[cost_metric]
        frontier.append(
            FrontierPoint(
                cost_weight=w,
                quality=on_quality.b.value,
                quality_interval=_bounds(on_quality.b),
                cost=on_cost.b.value,
                cost_interval=_bounds(on_cost.b),
                quality_difference=on_quality.difference,
                quality_difference_interval=on_quality.difference_interval,
                cost_difference=on_cost.difference,
                trustworthy=on_quality.b.trustworthy and on_cost.b.trustworthy,
            )
        )

    chosen = _select(tuple(frontier), budget, tolerance)
    logger.info(
        "swept %d cost weights on %d tuning rows; chose %g (%.6g per request, "
        "%+.6g quality against ignoring cost)",
        len(weights),
        data.n,
        chosen.cost_weight,
        chosen.cost,
        chosen.quality_difference,
    )
    return SweepResult(
        frontier=tuple(frontier),
        chosen=chosen,
        epsilon=epsilon,
        metric=metric,
        cost_metric=cost_metric,
        n_fit=fitted.n_rows,
        n_tune=data.n,
        feature_spec=fitted.feature_spec,
        arms=fitted.arms,
        budget=budget,
        tolerance=tolerance,
        _policy=policy,
    )


def _bounds(estimate: Any) -> tuple[float, float] | None:
    """The interval on an estimate as a plain pair.

    Args:
        estimate: An :class:`~sortition.eval.Estimate`.

    Returns:
        ``(low, high)``, or ``None`` when the estimator produced no interval.
    """
    interval = estimate.interval
    return None if interval is None else (interval.low, interval.high)


def _select(
    frontier: tuple[FrontierPoint, ...], budget: float | None, tolerance: float
) -> FrontierPoint:
    """Pick a point on the frontier.

    Args:
        frontier: The measured points, in increasing weight.
        budget: Optional ceiling on cost per request.
        tolerance: The most quality the operator will spend.

    Returns:
        The selected point.

    Raises:
        ValueError: If nothing on the grid comes in under ``budget``.
    """
    if budget is not None:
        affordable = [p for p in frontier if p.cost <= budget]
        if not affordable:
            cheapest = min(frontier, key=lambda p: p.cost)
            raise ValueError(
                f"no cost weight on the grid comes in under a budget of "
                f"{budget:.6g} per request; the cheapest is {cheapest.cost:.6g} "
                f"at weight {cheapest.cost_weight:g}. Widen the grid, or the "
                "budget is not reachable by routing alone."
            )
        # Ties on quality go to the cheaper point, which is the whole objective.
        return max(affordable, key=lambda p: (p.quality, -p.cost))

    # The cost-blind point is always here: its difference against itself is
    # exactly zero, so it is non-inferior at any tolerance including zero.
    safe = [p for p in frontier if p.non_inferior_at(tolerance)]
    if not safe:
        # Only reachable when even the cost-blind point is untrustworthy, which
        # means the log cannot support any of this. Returning it keeps the caller
        # on the policy it would have trained anyway; `doctor` is what should be
        # shouting about the log.
        blind = min(frontier, key=lambda p: p.cost_weight)
        logger.warning(
            "no cost weight is trustworthy on these rows, including %g; "
            "falling back to it and ignoring cost",
            blind.cost_weight,
        )
        return blind
    return min(safe, key=lambda p: (p.cost, p.cost_weight))
