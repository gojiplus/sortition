"""Target policies: the "what if we had done this instead?" side of the question.

A target policy takes the features and eligible set of each logged request and
returns a distribution over arms. It never sees an outcome -- it describes a
counterfactual routing rule, and the estimators supply the outcomes.

Every target here respects eligibility. A policy that assigns mass to an arm the
hard filter excluded is asking about a request that could not have happened, and
the diagnostics will say so rather than quietly extrapolating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@runtime_checkable
class TargetPolicy(Protocol):
    """Maps logged requests to counterfactual action probabilities."""

    @property
    def name(self) -> str:
        """Identifies the policy in reports and log rows."""
        ...

    def probabilities(
        self,
        features: list[dict[str, Any]],
        eligible: BoolArray,
        arms: tuple[str, ...],
    ) -> FloatArray:
        """Return an ``(n, len(arms))`` array whose rows sum to 1."""
        ...


def _uniform_over_eligible(eligible: BoolArray) -> FloatArray:
    counts = eligible.sum(axis=1, keepdims=True)
    if (counts == 0).any():
        raise ValueError("some rows have an empty eligible set; nothing could have served them")
    return np.where(eligible, 1.0 / counts, 0.0)


@dataclass(frozen=True)
class AlwaysArm:
    """Send everything to one arm.

    The baseline every routing discussion starts from -- "what would
    always-premium have cost us?" -- and the hardest case for overlap, because a
    logging policy that rarely chose that arm supports the estimate on few rows.

    Where the arm is ineligible it cannot be used, so those rows fall back to
    uniform over what was available. That is a real routing decision, not a
    modeling convenience: the alternative is claiming a counterfactual that the
    hard filter forbids.
    """

    arm: str

    @property
    def name(self) -> str:
        return f"always:{self.arm}"

    def probabilities(
        self,
        features: list[dict[str, Any]],
        eligible: BoolArray,
        arms: tuple[str, ...],
    ) -> FloatArray:
        if self.arm not in arms:
            raise ValueError(f"unknown arm {self.arm!r}; the log contains {list(arms)}")
        index = arms.index(self.arm)
        probs = np.zeros(eligible.shape, dtype=np.float64)
        usable = eligible[:, index]
        probs[usable, index] = 1.0
        if not usable.all():
            probs[~usable] = _uniform_over_eligible(eligible[~usable])
        return probs


@dataclass(frozen=True)
class Uniform:
    """Spread traffic evenly over whatever was eligible. Maximum exploration."""

    name: str = "uniform"

    def probabilities(
        self,
        features: list[dict[str, Any]],
        eligible: BoolArray,
        arms: tuple[str, ...],
    ) -> FloatArray:
        return _uniform_over_eligible(eligible)


@dataclass(frozen=True)
class Mixture:
    """Interpolate between two policies.

    Useful for asking how far toward a candidate policy it is safe to move: a
    10% mixture keeps almost all the overlap of the logging policy while shifting
    a measurable amount of traffic.
    """

    base: TargetPolicy
    other: TargetPolicy
    weight: float

    @property
    def name(self) -> str:
        return f"mix({self.base.name},{self.other.name},{self.weight:g})"

    def probabilities(
        self,
        features: list[dict[str, Any]],
        eligible: BoolArray,
        arms: tuple[str, ...],
    ) -> FloatArray:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"weight must be in [0, 1], got {self.weight}")
        a = self.base.probabilities(features, eligible, arms)
        b = self.other.probabilities(features, eligible, arms)
        return np.asarray((1.0 - self.weight) * a + self.weight * b, dtype=np.float64)


@dataclass(frozen=True)
class EpsilonFloor:
    """Wrap a policy so it always keeps ``epsilon`` of traffic exploring.

    This is what keeps a deployed policy evaluable. A policy with no exploration
    floor produces logs that can only ever confirm what it already does, which is
    how a router ends up unable to justify itself.
    """

    inner: TargetPolicy
    epsilon: float = 0.05

    @property
    def name(self) -> str:
        return f"{self.inner.name}+eps{self.epsilon:g}"

    def probabilities(
        self,
        features: list[dict[str, Any]],
        eligible: BoolArray,
        arms: tuple[str, ...],
    ) -> FloatArray:
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {self.epsilon}")
        greedy = self.inner.probabilities(features, eligible, arms)
        explore = _uniform_over_eligible(eligible)
        return np.asarray((1.0 - self.epsilon) * greedy + self.epsilon * explore, dtype=np.float64)


def parse_target(spec: str) -> TargetPolicy:
    """Build a target policy from a CLI-friendly string.

    ``always:premium-reasoning``, ``uniform``, or either with an exploration
    floor appended: ``always:premium-reasoning+eps0.05``.
    """
    text = spec.strip()
    epsilon: float | None = None
    if "+eps" in text:
        text, _, eps_text = text.partition("+eps")
        try:
            epsilon = float(eps_text)
        except ValueError as exc:
            raise ValueError(f"could not read an epsilon from {spec!r}") from exc

    policy: TargetPolicy
    if text == "uniform":
        policy = Uniform()
    elif text.startswith("always:"):
        arm = text.removeprefix("always:").strip()
        if not arm:
            raise ValueError(f"{spec!r} names no arm")
        policy = AlwaysArm(arm)
    else:
        raise ValueError(
            f"unrecognized target {spec!r}; expected 'uniform', 'always:<arm>', "
            "optionally with '+eps<float>'"
        )

    return EpsilonFloor(policy, epsilon) if epsilon is not None else policy
