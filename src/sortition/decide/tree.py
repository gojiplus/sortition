"""A learned policy: gradient-boosted trees over the logged features.

Ranks arms by predicted outcome minus a weighted, normalized cost. Two models,
not one: quality and price are separately predicted for each eligible arm, and
both are conditioned on the request.

Pricing cost per *request* rather than per arm is what makes the trade-off
truthful. A bill is price-per-token times tokens, so the gap between a budget
arm and a premium one is wide on a long request and nearly nothing on a short
one. Charging every request the arm's global mean -- the earlier design -- prices
the cheap-to-premium ladder and nothing else, and gets the direction of the
trade-off exactly backwards on the short requests where the premium arm is
almost free.

The cost weight is the operator's exchange rate between quality and price, and it
is the one number worth tuning from logs rather than guessing;
:mod:`sortition.train.sweep` does that.

This satisfies the same ``Policy`` protocol as a rules table, so a trained policy
deploys through the same engine, serializes into the same artifact format, and is
evaluated by the same estimators. Replacing rules with a model changes no other
part of the system, which is what the frozen artifact schema was for.

Feature vectors are built through ``sortition.features`` against the spec
recorded in the artifact, so the policy scores live requests exactly as it was
trained.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sortition.features import matrix, vectorize

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

MAX_DESIGN_CELLS = 8_000_000
"""Cells of the batched design matrix to hold at once, about 64 MB of float64.

:meth:`TreePolicy.score_matrix` expands a log to one row per request-arm pair, so
the array grows with the product of the two. Four arms over a day of traffic is a
few megabytes; the hundred-model rosters that make routing worth doing at all are
gigabytes of it, which is an out-of-memory error rather than a slow sweep. The
batch is cut into blocks of this many cells instead."""


@dataclass
class TreePolicy:
    """Scores arms with a boosted-tree outcome model and a boosted cost model.

    One model over all arms, with the arm as a one-hot block, rather than a
    model per arm: arms share most of what makes a request easy or hard, and
    per-arm models discard that exactly where data is thinnest.
    """

    booster_text: str
    """The serialized LightGBM model. Text so an artifact stays diffable and
    needs no pickle, which would make deploying a policy a code-execution
    decision."""

    feature_spec: tuple[str, ...]
    arms: tuple[str, ...]
    cost_booster_text: str | None = None
    """A second model, over the same design, predicting dollars. ``None`` when
    the training log carried no cost column, in which case ``cost_weight`` must
    be zero -- a policy that claims to price cost and cannot is worse than one
    that never claimed to."""

    cost_scale: float = 0.0
    """Dollars that one unit of ``cost_weight`` is worth, fixed when the policy
    was fitted: the mean predicted cost over the training log.

    A fixed scale rather than each request's own min and max. Normalizing within
    the eligible set looks equivalent and destroys the thing this policy exists
    to use: a bill is size times price, so both the numerator and the spread
    scale with the request, they cancel, and every request ends up with the same
    penalties -- the cheapest arm at zero and the dearest at exactly one, whether
    it is a 200-token call or a 20,000-token one. Against a fixed scale the
    penalty grows with the request, which is what lets the premium arm be worth
    taking when it is cheap and not when it is not."""

    cost_weight: float = 0.0
    name: str = "tree"
    _booster: Any = field(default=None, init=False, repr=False, compare=False)
    _cost_booster: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Reject a cost weight the policy has no cost model to apply."""
        if self.cost_weight != 0.0 and self.cost_booster_text is None:
            raise ValueError(
                "cost_weight is nonzero but no cost model was fitted; the "
                "policy would silently rank on quality alone"
            )

    @property
    def booster(self) -> Any:
        """The deserialized outcome model, loaded on first use.

        Returns:
            A LightGBM ``Booster``.
        """
        if self._booster is None:
            import lightgbm as lgb

            self._booster = lgb.Booster(model_str=self.booster_text)
        return self._booster

    @property
    def cost_booster(self) -> Any:
        """The deserialized cost model, or ``None`` if there is none.

        Returns:
            A LightGBM ``Booster``, or ``None``.
        """
        if self.cost_booster_text is None:
            return None
        if self._cost_booster is None:
            import lightgbm as lgb

            self._cost_booster = lgb.Booster(model_str=self.cost_booster_text)
        return self._cost_booster

    def _design(
        self, features: dict[str, Any], eligible: tuple[str, ...]
    ) -> FloatArray:
        """Build the (arm, feature) design rows this policy was fitted on.

        Args:
            features: The request's feature vector.
            eligible: Arms surviving the hard filter.

        Returns:
            One row per eligible arm.
        """
        base = vectorize(features, self.feature_spec)
        index = {arm: i for i, arm in enumerate(self.arms)}
        rows = []
        for arm in eligible:
            onehot = [0.0] * len(self.arms)
            if arm in index:
                onehot[index[arm]] = 1.0
            rows.append(base + onehot)
        return np.array(rows, dtype=np.float64)

    def predict(
        self, features: dict[str, Any], eligible: tuple[str, ...]
    ) -> dict[str, float]:
        """Predicted outcome for each eligible arm.

        Args:
            features: The request's feature vector.
            eligible: Arms surviving the hard filter.

        Returns:
            A predicted outcome per arm.
        """
        predictions = self.booster.predict(self._design(features, eligible))
        return dict(zip(eligible, (float(p) for p in predictions), strict=True))

    def predict_cost(
        self, features: dict[str, Any], eligible: tuple[str, ...]
    ) -> dict[str, float]:
        """Predicted dollar cost of serving *this* request on each eligible arm.

        Args:
            features: The request's feature vector.
            eligible: Arms surviving the hard filter.

        Returns:
            A predicted cost per arm, in USD. Empty when no cost model was
            fitted. Clipped at zero: a booster extrapolating below the cheapest
            row it saw can predict a negative bill, which would read as a credit.
        """
        booster = self.cost_booster
        if booster is None:
            return {}
        predictions = booster.predict(self._design(features, eligible))
        return dict(
            zip(eligible, (max(float(p), 0.0) for p in predictions), strict=True)
        )

    def score(
        self, features: dict[str, Any], eligible: tuple[str, ...]
    ) -> dict[str, float]:
        """Rank arms by predicted outcome, discounted by predicted cost.

        Args:
            features: The request's feature vector.
            eligible: Arms surviving the hard filter.

        Returns:
            A score per eligible arm; higher is preferred.
        """
        if not eligible:
            return {}
        quality = self.predict(features, eligible)
        if self.cost_weight == 0.0:
            return quality

        costs = self.predict_cost(features, eligible)
        return _discount(quality, costs, self.cost_weight, self.cost_scale, eligible)

    def score_matrix(
        self, features: list[dict[str, Any]], eligible: NDArray[np.bool_]
    ) -> tuple[FloatArray, FloatArray]:
        """Predict quality and cost for every row-arm pair in two batched calls.

        The per-row :meth:`score` path pays LightGBM's call overhead once per
        request, which is right for serving and wrong for a sweep that rescores
        the same log at a dozen cost weights. Both routes must agree; a test
        asserts they do, because a fast path that quietly disagrees with the
        deployed one would tune a weight for a policy that never ships.

        Args:
            features: Per-row feature dicts.
            eligible: An ``(n, len(arms))`` mask, used only for its shape here;
                ineligible arms are scored and masked by the caller.

        Returns:
            ``(quality, cost)``, each ``(n, len(arms))``. Cost is all-zero when
            no cost model was fitted.
        """
        n, k = len(features), len(self.arms)
        base = matrix(features, self.feature_spec)
        eye = np.eye(k, dtype=np.float64)
        booster = self.cost_booster

        quality = np.zeros((n, k), dtype=np.float64)
        cost = np.zeros((n, k), dtype=np.float64)
        per_row = k * (base.shape[1] + k)
        step = max(1, MAX_DESIGN_CELLS // per_row)
        for start in range(0, n, step):
            stop = min(start + step, n)
            # One row per (request, arm): repeat the features, cycle the arms.
            block = np.hstack(
                [
                    np.repeat(base[start:stop], k, axis=0),
                    np.tile(eye, (stop - start, 1)),
                ]
            )
            quality[start:stop] = np.asarray(
                self.booster.predict(block), dtype=np.float64
            ).reshape(-1, k)
            if booster is not None:
                cost[start:stop] = np.maximum(
                    np.asarray(booster.predict(block), dtype=np.float64), 0.0
                ).reshape(-1, k)
        return quality, cost


def _discount(
    quality: dict[str, float],
    costs: dict[str, float],
    weight: float,
    scale: float,
    eligible: tuple[str, ...],
) -> dict[str, float]:
    """Subtract cost, measured against a fixed scale, from quality.

    ``weight`` is how much outcome one ``scale``-worth of spend is allowed to
    cost, so at the default scale -- the mean bill over the training log -- a
    weight of 1.0 says an average-priced request may give up a full point of
    outcome. A request costing three times the average is charged three times as
    much, which is the whole reason cost is predicted per request.

    Args:
        quality: Predicted outcome per arm.
        costs: Predicted cost per arm.
        weight: The exchange rate between the two.
        scale: Dollars per unit of weight.
        eligible: Arms surviving the hard filter, in order.

    Returns:
        A score per arm.
    """
    if scale <= 0.0:
        return quality
    cheapest = min(costs.get(arm, 0.0) for arm in eligible)
    # Relative to the cheapest eligible arm, so a request whose every option is
    # expensive is not penalised for having no cheap option.
    return {
        arm: quality[arm] - weight * ((costs.get(arm, 0.0) - cheapest) / scale)
        for arm in eligible
    }


def discount_matrix(
    quality: FloatArray,
    cost: FloatArray,
    eligible: NDArray[np.bool_],
    weight: float,
    scale: float,
) -> FloatArray:
    """The batched form of :func:`_discount`, over a whole log at one weight.

    Args:
        quality: ``(n, K)`` predicted outcomes.
        cost: ``(n, K)`` predicted costs.
        eligible: ``(n, K)`` mask of arms surviving the hard filter.
        weight: The exchange rate between quality and cost.
        scale: Dollars per unit of weight.

    Returns:
        ``(n, K)`` scores, with ineligible entries left at ``-inf``.
    """
    scores = np.where(eligible, quality, -np.inf)
    if weight == 0.0 or scale <= 0.0:
        return scores
    masked = np.where(eligible, cost, np.nan)
    low = np.nanmin(masked, axis=1, keepdims=True)
    return np.where(eligible, quality - weight * ((masked - low) / scale), -np.inf)
