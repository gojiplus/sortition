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

from sortition.exploration import apply_epsilon_floor, uniform_over_eligible

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

# Aliased rather than reimplemented: the evaluation side and the decision side
# must compute exploration probabilities with one formula, or a policy stops
# evaluating as itself. See sortition.exploration.
_uniform_over_eligible = uniform_over_eligible


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
        """Identifier used in reports."""
        return f"always:{self.arm}"

    def probabilities(
        self,
        features: list[dict[str, Any]],
        eligible: BoolArray,
        arms: tuple[str, ...],
    ) -> FloatArray:
        """Action probabilities for each logged request.

        Args:
            features: Per-row feature dicts as logged.
            eligible: Boolean mask of arms surviving the hard filter.
            arms: The arm universe, in index order.

        Returns:
            An ``(n, len(arms))`` array whose rows sum to 1.

        Raises:
            ValueError: If ``self.arm`` is not in the arm universe.
        """
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
        """Action probabilities for each logged request.

        Args:
            features: Per-row feature dicts as logged.
            eligible: Boolean mask of arms surviving the hard filter.
            arms: The arm universe, in index order.

        Returns:
            An ``(n, len(arms))`` array whose rows sum to 1.
        """
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
        """Identifier used in reports."""
        return f"mix({self.base.name},{self.other.name},{self.weight:g})"

    def probabilities(
        self,
        features: list[dict[str, Any]],
        eligible: BoolArray,
        arms: tuple[str, ...],
    ) -> FloatArray:
        """Action probabilities for each logged request.

        Args:
            features: Per-row feature dicts as logged.
            eligible: Boolean mask of arms surviving the hard filter.
            arms: The arm universe, in index order.

        Returns:
            An ``(n, len(arms))`` array whose rows sum to 1.

        Raises:
            ValueError: If ``weight`` is outside [0, 1].
        """
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
        """Identifier used in reports."""
        return f"{self.inner.name}+eps{self.epsilon:g}"

    def probabilities(
        self,
        features: list[dict[str, Any]],
        eligible: BoolArray,
        arms: tuple[str, ...],
    ) -> FloatArray:
        """Action probabilities for each logged request.

        Args:
            features: Per-row feature dicts as logged.
            eligible: Boolean mask of arms surviving the hard filter.
            arms: The arm universe, in index order.

        Returns:
            An ``(n, len(arms))`` array whose rows sum to 1. Validation of
            ``epsilon`` happens in the shared implementation.
        """
        greedy = self.inner.probabilities(features, eligible, arms)
        return apply_epsilon_floor(greedy, eligible, self.epsilon)


@dataclass(frozen=True)
class PolicyTarget:
    """A real routing policy, asked what it *would* have done.

    This is the bridge between the two halves of the project. A deployed policy
    scores one request at a time (``decide.Policy.score``); an evaluation target
    produces probabilities for a whole log. Without something joining them you
    can deploy a policy and you can evaluate ``always:premium``, but you cannot
    evaluate the policy you actually run -- which is the question the project
    exists to answer.

    Two things make it correct rather than merely present.

    The exploration mix comes from :func:`~sortition.exploration.apply_epsilon_floor`,
    the same function the decision path uses. A second implementation here would
    let a policy stop evaluating as itself, and the resulting bias would be
    invisible: the weights would still look reasonable.

    Probabilities never leave the logged eligible set. That set is already
    post-hard-filter, so the candidate's own constraints intersect with it rather
    than replacing it. Where a candidate would have reached outside it, the
    support-violation diagnostic is what should say so -- silently renormalising
    would hide exactly the case that cannot be estimated.
    """

    policy: Any
    """A ``sortition.decide.Policy``. Typed loosely so this module does not
    depend on ``decide``, which would make the import graph circular."""

    epsilon: float = 0.05
    name: str = "policy"

    def probabilities(
        self,
        features: list[dict[str, Any]],
        eligible: BoolArray,
        arms: tuple[str, ...],
    ) -> FloatArray:
        """Action probabilities this policy would have produced.

        Args:
            features: Per-row feature dicts as logged.
            eligible: Boolean mask of arms surviving the hard filter.
            arms: The arm universe, in index order.

        Returns:
            An ``(n, len(arms))`` array whose rows sum to 1.
        """
        index = {arm: i for i, arm in enumerate(arms)}
        greedy = np.zeros(eligible.shape, dtype=np.float64)
        # Exploration spreads over the arms this policy would have drawn from,
        # which is its own post-hard-filter set -- not the whole logged mask.
        # The decision path computes the propensity over exactly that set, so
        # using a different denominator here would put the two out of step and
        # a policy would stop evaluating as itself.
        survivor_mask = np.zeros(eligible.shape, dtype=bool)
        hard_filter = getattr(self.policy, "hard_filter", None)

        for row, mask in enumerate(eligible):
            available = tuple(arm for arm in arms if mask[index[arm]])
            if not available:
                continue
            row_features = features[row] if row < len(features) else {}

            survivors = available
            if hard_filter is not None:
                filtered, _ = hard_filter(row_features, available)
                # Intersect rather than replace: an arm the log could not have
                # served has no counterfactual regardless of what this policy
                # thinks of it.
                survivors = (
                    tuple(a for a in filtered if a in set(available)) or available
                )

            for arm in survivors:
                survivor_mask[row, index[arm]] = True
            scores = self.policy.score(row_features, survivors)
            # Ties break toward the arm that sorts first, which is what
            # `DecisionEngine` does when it ranks by (-score, arm). Taking the
            # last one here instead would mean a policy stops evaluating as the
            # policy that serves, on exactly the rows where the model is
            # indifferent.
            best = min(survivors, key=lambda arm: (-scores.get(arm, 0.0), arm))
            greedy[row, index[best]] = 1.0

        # Rows with an empty eligible set keep an all-zero row rather than
        # tripping the uniform helper's empty-support guard.
        empty = ~survivor_mask.any(axis=1)
        if empty.all():
            return greedy
        result = np.zeros(eligible.shape, dtype=np.float64)
        keep = ~empty
        result[keep] = apply_epsilon_floor(
            greedy[keep], survivor_mask[keep], self.epsilon
        )
        return result


def parse_target(spec: str) -> TargetPolicy:
    """Build a target policy from a CLI-friendly string.

    ``uniform``, ``always:<arm>``, or ``policy:<artifact.json>`` to ask what a
    real deployed policy would have done. Any of them may carry an exploration
    floor: ``always:premium+eps0.05``.

    Args:
        spec: The target specification.

    Returns:
        The parsed target policy.

    Raises:
        ValueError: If the spec names no arm, carries an unreadable epsilon, or
            is not one of the recognised forms.
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
    elif text.startswith("policy:"):
        path = text.removeprefix("policy:").strip()
        if not path:
            raise ValueError(f"{spec!r} names no artifact")
        # Imported here so this module does not depend on `decide`, and so an
        # artifact edited without being rebuilt is rejected by exactly the same
        # check that guards the decision path.
        from sortition.decide.artifact import load

        deployed, exploration, meta = load(path)
        return PolicyTarget(
            policy=deployed,
            epsilon=epsilon if epsilon is not None else exploration.epsilon,
            name=meta.policy_version,
        )
    else:
        raise ValueError(
            f"unrecognized target {spec!r}; expected 'uniform', 'always:<arm>' "
            "or 'policy:<artifact.json>', optionally with '+eps<float>'"
        )

    return EpsilonFloor(policy, epsilon) if epsilon is not None else policy
