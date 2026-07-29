"""The decision path: hard filter, rank, explore, record why.

Three stages, in this order, because each depends on the last:

1. **Hard filter.** Which arms could have served this request at all. Anything
   removed here has no counterfactual — the estimators must not be asked what
   would have happened if we had used it.
2. **Rank.** Which of the survivors the policy prefers.
3. **Explore.** How often to try something other than the favourite, and with
   what probability, recorded exactly.

Stage 3 is what makes the log worth keeping. A router with no exploration
produces logs that can only ever confirm what it already does.
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from sortition.decide.policy import Policy
from sortition.decide.thompson import DEFAULT_PROPENSITY_SAMPLES
from sortition.exploration import epsilon_greedy_propensity
from sortition.schema import Decision

logger = logging.getLogger(__name__)

# Below this, a log stops being able to answer questions about the arms the
# policy dislikes -- which are exactly the arms worth asking about.
MIN_USEFUL_EPSILON = 0.01


@dataclass(frozen=True)
class ExplorationConfig:
    """How much traffic to spend learning, and how."""

    strategy: Literal["epsilon_greedy", "none"] = "epsilon_greedy"
    epsilon: float = 0.05
    n_samples: int = DEFAULT_PROPENSITY_SAMPLES

    def as_dict(self) -> dict[str, Any]:
        """Render for the policy artifact and the log row.

        Returns:
            A mapping describing the exploration strategy and its parameters.
        """
        if self.strategy == "none":
            return {"strategy": "none"}
        return {"strategy": self.strategy, "epsilon": self.epsilon}


@dataclass
class DecisionEngine:
    """Turns a request into a logged, reproducible routing decision."""

    policy: Policy
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    policy_version: str | None = None
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        """Derive a policy version and warn about un-evaluable configurations."""
        if self.policy_version is None:
            suffix = (
                f"+eps{self.exploration.epsilon:g}"
                if self.exploration.strategy == "epsilon_greedy"
                else "+noexplore"
            )
            self.policy_version = f"{self.policy.name}{suffix}"

        if (
            self.exploration.strategy == "none"
            or self.exploration.epsilon < MIN_USEFUL_EPSILON
        ):
            logger.warning(
                "policy %s explores %.3f of traffic: its logs will support no "
                "counterfactual claim about the arms it does not already pick",
                self.policy_version,
                0.0
                if self.exploration.strategy == "none"
                else self.exploration.epsilon,
            )

    def decide(
        self,
        *,
        features: dict[str, Any],
        eligible: list[str] | tuple[str, ...],
        decision_id: str | None = None,
    ) -> Decision:
        """Choose an arm and record the probability it was chosen with.

        Args:
            features: The request's feature vector, logged verbatim.
            eligible: Arms the gateway says are available for this request.
            decision_id: Optional caller-supplied id; generated when absent.

        Returns:
            A :class:`~sortition.schema.Decision` carrying the chosen arm, its
            propensity, the eligible set it was drawn from, and an ordered
            fallback chain.

        Raises:
            ValueError: If ``eligible`` is empty.
        """
        offered = tuple(dict.fromkeys(eligible))  # de-duplicate, preserve order
        if not offered:
            raise ValueError("no eligible arms; nothing can serve this request")

        survivors, constraints = self._hard_filter(features, offered)
        scores = self.policy.score(features, survivors)
        ranked = tuple(sorted(survivors, key=lambda arm: (-scores.get(arm, 0.0), arm)))
        greedy = ranked[0]

        chosen, propensity, explored = self._sample(greedy, survivors)

        # The fallback chain is what the gateway should try if the chosen arm
        # fails. It carries no propensities: those arms are instructions, not
        # draws, and a fallback that fires means the served arm was not sampled.
        fallback_chain = tuple(arm for arm in ranked if arm != chosen)

        decision = Decision(
            decision_id=decision_id or f"d-{uuid.uuid4().hex[:16]}",
            policy_version=self.policy_version or self.policy.name,
            chosen_arm=chosen,
            propensity=propensity,
            eligible_set=survivors,
            explore=explored,
            fallback_chain=fallback_chain,
            hard_constraints_applied=constraints,
        )
        logger.debug(
            "decided %s (p=%.4f, explore=%s) from %s",
            chosen,
            propensity,
            explored,
            list(survivors),
        )
        return decision

    def _hard_filter(
        self, features: dict[str, Any], offered: tuple[str, ...]
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Apply the policy's hard constraints, if it has any.

        Args:
            features: The request's feature vector.
            offered: Arms the gateway offered.

        Returns:
            The surviving arms and the labels of the constraints applied.
        """
        hard_filter = getattr(self.policy, "hard_filter", None)
        if hard_filter is None:
            return offered, ()
        survivors, constraints = hard_filter(features, offered)
        return tuple(survivors), tuple(constraints)

    def _sample(
        self, greedy: str, survivors: tuple[str, ...]
    ) -> tuple[str, float, bool]:
        """Draw an arm and compute its exact propensity.

        Args:
            greedy: The policy's preferred arm.
            survivors: The eligible set to draw from.

        Returns:
            The chosen arm, its propensity, and whether exploration fired.
        """
        n = len(survivors)
        if self.exploration.strategy == "none" or n == 1:
            return greedy, 1.0, False

        epsilon = self.exploration.epsilon
        if self.rng.random() < epsilon:
            # Uniform over the whole eligible set, including the greedy arm --
            # which is why the greedy arm's propensity is 1-eps+eps/n and not
            # simply 1-eps. Getting this wrong is the classic epsilon-greedy
            # propensity bug.
            chosen = self.rng.choice(survivors)
            explored = chosen != greedy
        else:
            chosen, explored = greedy, False

        propensity = epsilon_greedy_propensity(
            n_eligible=n, epsilon=epsilon, is_greedy=chosen == greedy
        )
        return chosen, propensity, explored
