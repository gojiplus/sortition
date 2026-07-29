"""A LiteLLM ``RoutingPlugin`` that samples one arm and records the probability.

Registered on a Router, this runs before the routing decision, narrows the
candidate pool to the single arm sortition sampled, and publishes the decision on
``context.signals`` so the logging callback can pick it up.

Narrowing to exactly one arm is the point. A propensity describes the
distribution an arm was drawn from; if the plugin left several candidates and
let LiteLLM's own strategy pick among them, the logged probability would
describe a draw that did not determine the outcome. LiteLLM's ``fallbacks:``
config still provides resilience — and when a fallback fires, that is precisely
the event ``fallback_depth`` records and the estimators exclude.

Arms are deployment model strings within one model group, because that is what
LiteLLM hands a plugin. Routing across capability tiers means defining a model
group that contains both, which is how LiteLLM's own examples are written.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Protocol

from sortition import metrics
from sortition.integrations.features import extract

if TYPE_CHECKING:
    from sortition.decide import DecisionEngine

logger = logging.getLogger(__name__)

SIGNAL_KEY = "sortition"


class _Context(Protocol):
    """The subset of LiteLLM's ``RoutingContext`` this plugin touches."""

    raw_messages: list[dict[str, Any]]
    structured_messages: list[dict[str, Any]]
    candidate_models: list[str]
    metadata: dict[str, Any]
    signals: dict[str, Any]


class SortitionPlugin:
    """Samples a deployment and publishes the decision for the logger.

    Any exception is caught and the context returned untouched. A bug in routing
    logic must not be able to fail a request: the cost of a missed decision is
    one unlogged row, the cost of an exception is a user-visible error. The
    failure is logged at error level so it is not silent.
    """

    def __init__(
        self,
        engine: DecisionEngine,
        *,
        feature_extra: dict[str, Any] | None = None,
    ) -> None:
        """Wire a decision engine into the routing pipeline.

        Args:
            engine: The configured :class:`~sortition.decide.DecisionEngine`.
            feature_extra: Static features merged into every request, e.g. a
                tenant identifier.
        """
        self.engine = engine
        self.feature_extra = feature_extra or {}

    async def run(self, context: _Context) -> _Context:
        """Narrow the candidate pool to one sampled arm.

        Args:
            context: LiteLLM's routing context.

        Returns:
            The context, with ``candidate_models`` narrowed and the decision
            published on ``signals``. Returned unmodified if anything fails.
        """
        try:
            return self._decide(context)
        except Exception:
            # Counted, not just logged: a fail-open plugin looks exactly like a
            # healthy one from outside -- requests succeed, latency is normal --
            # so this counter is the only signal that routing has stopped.
            metrics.plugin_errors.inc()
            logger.exception(
                "sortition plugin failed; leaving routing untouched so the "
                "request still succeeds. This request will not be evaluable."
            )
            return context

    def _decide(self, context: _Context) -> _Context:
        candidates = list(context.candidate_models)
        if not candidates:
            # Another plugin already excluded everything; adding arms back would
            # override its policy, and LiteLLM ignores additions anyway.
            logger.debug("no candidate models left for sortition to choose from")
            return context

        metadata = context.metadata or {}
        features = extract(
            context.structured_messages or context.raw_messages or [],
            tools=metadata.get("tools"),
            extra=self.feature_extra,
        )
        start = time.perf_counter()
        decision = self.engine.decide(features=features, eligible=candidates)
        metrics.observe_decision(
            arm=decision.chosen_arm,
            policy_version=decision.policy_version,
            explore=decision.explore,
            propensity_value=decision.propensity,
            seconds=time.perf_counter() - start,
        )

        context.candidate_models = [decision.chosen_arm]
        context.signals[SIGNAL_KEY] = {
            "schema_version": decision.schema_version,
            "decision_id": decision.decision_id,
            "policy_version": decision.policy_version,
            "chosen_arm": decision.chosen_arm,
            "propensity": decision.propensity,
            "eligible_set": list(decision.eligible_set),
            "explore": decision.explore,
            "fallback_chain": list(decision.fallback_chain),
            "hard_constraints_applied": list(decision.hard_constraints_applied),
            "features": features,
        }
        return context
