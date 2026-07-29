"""A YAML rule table: the policy most teams are already running.

It ships first not because it is clever but because it is honest about where
routing starts. A rules file is legible, reviewable, and requires no data to
write — and once it is deployed behind an exploration floor it produces exactly
the logs a trained policy needs. The trained policy replaces it later without
changing the decision API, because both satisfy the same `Policy` protocol.

Rules do two separable things, and the distinction matters downstream. ``exclude``
is a *hard constraint*: the arm could not have served the request, so there is no
counterfactual to estimate and the eligible set shrinks. ``prefer`` is a *soft
ranking*: the arm was available and simply was not favoured, which is exactly the
kind of decision exploration lets us second-guess later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_COMPARATORS = {
    "eq": lambda value, target: value == target,
    "ne": lambda value, target: value != target,
    "gt": lambda value, target: value > target,
    "gte": lambda value, target: value >= target,
    "lt": lambda value, target: value < target,
    "lte": lambda value, target: value <= target,
    "in": lambda value, target: value in target,
}


def _matches(condition: Any, value: Any) -> bool:
    """Test one feature against one condition.

    Args:
        condition: A literal to compare equal, or a mapping of operator to operand.
        value: The feature value from the request.

    Returns:
        Whether the condition holds.

    Raises:
        ValueError: If the condition names an unknown operator.
    """
    if not isinstance(condition, dict):
        return bool(value == condition)
    for operator, target in condition.items():
        comparator = _COMPARATORS.get(operator)
        if comparator is None:
            raise ValueError(
                f"unknown operator {operator!r}; expected one of {sorted(_COMPARATORS)}"
            )
        if value is None:
            return False
        try:
            if not comparator(value, target):
                return False
        except TypeError:
            # A feature of the wrong type is a non-match, not a crash: routing
            # must not fail because one caller sent a string where a number was
            # expected.
            logger.debug(
                "rule condition %s=%r skipped: incomparable value %r",
                operator,
                target,
                value,
            )
            return False
    return True


@dataclass(frozen=True)
class Rule:
    """One row of the rule table."""

    when: dict[str, Any] = field(default_factory=dict)
    prefer: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    label: str = ""

    def applies(self, features: dict[str, Any]) -> bool:
        """Whether every condition in ``when`` holds for this request.

        Args:
            features: The request's feature vector.

        Returns:
            True when all conditions match. An empty ``when`` always matches.
        """
        return all(
            _matches(condition, features.get(key))
            for key, condition in self.when.items()
        )


@dataclass(frozen=True)
class RulesPolicy:
    """Rank arms by the first matching rule, falling back to a default order."""

    arms: tuple[str, ...]
    rules: tuple[Rule, ...] = ()
    default: tuple[str, ...] = ()
    name: str = "rules"

    def hard_filter(
        self, features: dict[str, Any], eligible: tuple[str, ...]
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Drop arms that could not have served this request.

        Every matching rule's exclusions apply, not just the first: a hard
        constraint is a statement about feasibility, and two of them are both
        true at once.

        Args:
            features: The request's feature vector.
            eligible: Arms the caller offered.

        Returns:
            The surviving arms and the labels of the constraints applied. If
            excluding everything would leave nothing, the exclusions are ignored
            and reported, because refusing to route is worse than routing to a
            disfavoured arm.
        """
        excluded: set[str] = set()
        applied: list[str] = []
        for rule in self.rules:
            if rule.exclude and rule.applies(features):
                excluded.update(rule.exclude)
                applied.append(rule.label or f"exclude:{','.join(rule.exclude)}")

        survivors = tuple(arm for arm in eligible if arm not in excluded)
        if not survivors:
            logger.warning(
                "hard constraints %s would exclude every eligible arm %s; "
                "ignoring them",
                applied,
                list(eligible),
            )
            return eligible, ()
        return survivors, tuple(applied)

    def score(
        self, features: dict[str, Any], eligible: tuple[str, ...]
    ) -> dict[str, float]:
        """Score eligible arms by their position in the first matching preference list.

        Args:
            features: The request's feature vector.
            eligible: Arms surviving the hard filter.

        Returns:
            A score per eligible arm; arms named earlier score higher, and arms
            named by no rule score below all of them.
        """
        order = self.default
        for rule in self.rules:
            if rule.prefer and rule.applies(features):
                order = rule.prefer
                break

        ranked = [arm for arm in order if arm in eligible]
        scores = {arm: float(len(ranked) - i) for i, arm in enumerate(ranked)}
        # Arms no rule mentions are still usable, just least preferred; leaving
        # them out entirely would silently shrink the support.
        return {arm: scores.get(arm, 0.0) for arm in eligible}

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> RulesPolicy:
        """Build a policy from a parsed rule table.

        Args:
            config: A mapping with ``arms`` and optionally ``rules``, ``default``
                and ``name``.

        Returns:
            The configured policy.

        Raises:
            ValueError: If ``arms`` is missing, or a rule names an unknown arm.
        """
        arms = tuple(config.get("arms") or ())
        if not arms:
            raise ValueError("a rules policy needs a non-empty 'arms' list")

        rules = []
        for raw in config.get("rules") or ():
            rule = Rule(
                when=dict(raw.get("when") or {}),
                prefer=tuple(raw.get("prefer") or ()),
                exclude=tuple(raw.get("exclude") or ()),
                label=str(raw.get("label") or ""),
            )
            unknown = (set(rule.prefer) | set(rule.exclude)) - set(arms)
            if unknown:
                raise ValueError(f"rule names arms not in 'arms': {sorted(unknown)}")
            rules.append(rule)

        default = tuple(config.get("default") or arms)
        unknown_default = set(default) - set(arms)
        if unknown_default:
            raise ValueError(
                f"'default' names arms not in 'arms': {sorted(unknown_default)}"
            )

        return cls(
            arms=arms,
            rules=tuple(rules),
            default=default,
            name=str(config.get("name") or "rules"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> RulesPolicy:
        """Load a rule table from a YAML file.

        Args:
            path: Path to the YAML rule table.

        Returns:
            The configured policy.

        Raises:
            ValueError: If the file does not contain a mapping.
        """
        import yaml

        content = Path(path).read_text(encoding="utf-8")
        config = yaml.safe_load(content)
        if not isinstance(config, dict):
            raise ValueError(f"{path} does not contain a YAML mapping")
        logger.info("loaded rules policy from %s", path)
        return cls.from_dict(config)
