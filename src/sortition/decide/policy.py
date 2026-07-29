"""Policies: what the router prefers, before exploration is layered on.

A policy expresses preference, not probability. It scores the arms that survived
the hard filter and says nothing about how often to try the others -- that is
exploration's job, and keeping the two apart is what lets the same policy be
deployed with an exploration floor and evaluated without one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

Features = dict[str, Any]


@runtime_checkable
class Policy(Protocol):
    """Scores eligible arms for a request. Higher is better."""

    @property
    def name(self) -> str:
        """Identifies the policy in logs and reports."""
        ...

    def score(self, features: Features, eligible: tuple[str, ...]) -> dict[str, float]:
        """Score each eligible arm.

        Args:
            features: The request's feature vector.
            eligible: Arms surviving the hard filter, in a stable order.

        Returns:
            A score per eligible arm. Higher is preferred.
        """
        ...


@dataclass(frozen=True)
class ConstantPolicy:
    """Always prefer one arm; rank the rest arbitrarily but stably.

    The baseline a team is usually already running, and the thing a candidate
    policy has to beat.
    """

    arm: str

    @property
    def name(self) -> str:
        """Identifies the policy in logs and reports."""
        return f"always:{self.arm}"

    def score(self, features: Features, eligible: tuple[str, ...]) -> dict[str, float]:
        """Score the preferred arm above all others.

        Args:
            features: Unused; this policy ignores the request.
            eligible: Arms surviving the hard filter.

        Returns:
            1.0 for the preferred arm, 0.0 elsewhere.
        """
        return {arm: (1.0 if arm == self.arm else 0.0) for arm in eligible}


@dataclass(frozen=True)
class WeightedPolicy:
    """Fixed per-arm preferences, independent of the request.

    Useful as a hand-tuned starting point and as the shape a trained policy's
    output takes once features are in play.
    """

    weights: dict[str, float]
    name: str = "weighted"

    def score(self, features: Features, eligible: tuple[str, ...]) -> dict[str, float]:
        """Look up each eligible arm's fixed weight.

        Args:
            features: Unused; this policy ignores the request.
            eligible: Arms surviving the hard filter.

        Returns:
            The configured weight per arm, defaulting to 0.0.
        """
        return {arm: self.weights.get(arm, 0.0) for arm in eligible}


@dataclass(frozen=True)
class CostAwarePolicy:
    """Trade predicted quality against price.

    The routing decision in its simplest honest form: prefer the better arm
    until the price difference outweighs the quality difference. ``cost_weight``
    is the operator's exchange rate between the two, and it is the one number
    most worth tuning from logs.
    """

    quality: dict[str, float]
    cost_usd: dict[str, float]
    cost_weight: float = 0.0
    name: str = "cost-aware"
    _max_cost: float = field(default=0.0, init=False, repr=False)

    def score(self, features: Features, eligible: tuple[str, ...]) -> dict[str, float]:
        """Score arms by quality minus weighted, normalized cost.

        Args:
            features: Unused; preferences here are per-arm, not per-request.
            eligible: Arms surviving the hard filter.

        Returns:
            A score per eligible arm.
        """
        costs = [self.cost_usd.get(arm, 0.0) for arm in eligible]
        spread = max(costs) - min(costs) if costs else 0.0
        if spread <= 0.0:
            return {arm: self.quality.get(arm, 0.0) for arm in eligible}
        cheapest = min(costs)
        return {
            arm: self.quality.get(arm, 0.0)
            - self.cost_weight * ((self.cost_usd.get(arm, 0.0) - cheapest) / spread)
            for arm in eligible
        }
