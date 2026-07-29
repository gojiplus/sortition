"""Turning LiteLLM callback payloads into evaluable log rows.

It reads from two places, and the split is not arbitrary.

The decision comes from the raw kwargs -- specifically
``litellm_params.metadata.routing_plugin_signals`` -- and not from
``standard_logging_object``. LiteLLM's
``get_standard_logging_metadata`` filters metadata down to the keys declared on
``StandardLoggingMetadata``, so a plugin's signals are stripped from the standard
payload. They survive on the raw kwargs. This is verified by an integration test,
because it is the assumption the whole logger rests on.

Everything else — which deployment actually served, cost, tokens, latency —
comes from ``standard_logging_object``, which is the stable, documented surface.

Comparing the two is what detects a fallback: if the arm that served is not the
arm that was sampled, LiteLLM's retry machinery overrode the decision, the served
arm was not drawn from the sampler, and its logged propensity no longer describes
how it was selected.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sortition.integrations.litellm_plugin import SIGNAL_KEY
from sortition.store.writer import json_safe

logger = logging.getLogger(__name__)


def read_decision(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """Pull sortition's decision out of a LiteLLM callback's kwargs.

    Args:
        kwargs: The kwargs LiteLLM passes to a ``CustomLogger`` callback.

    Returns:
        The decision published by the plugin, or None when this request was not
        routed by sortition.
    """
    params = kwargs.get("litellm_params") or {}
    metadata = params.get("metadata") or {}
    signals = metadata.get("routing_plugin_signals") or {}
    decision = signals.get(SIGNAL_KEY) if isinstance(signals, dict) else None
    return decision if isinstance(decision, dict) else None


def to_log_row(kwargs: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    """Combine a decision and an execution into one flat log row.

    Args:
        kwargs: The callback kwargs, carrying ``standard_logging_object``.
        decision: The decision published by the plugin.

    Returns:
        A row in the shape ``sortition.frame.to_arrays`` reads.
    """
    payload = kwargs.get("standard_logging_object") or {}
    served = payload.get("model")
    chosen = decision.get("chosen_arm")

    # A served arm that differs from the sampled one means the gateway's own
    # failover overrode the decision after it was made.
    fallback_depth = 0
    if served and chosen and served != chosen:
        chain = decision.get("fallback_chain") or []
        fallback_depth = chain.index(served) + 1 if served in chain else 1

    started = payload.get("startTime")
    ended = payload.get("endTime")
    latency_ms = None
    if isinstance(started, int | float) and isinstance(ended, int | float):
        latency_ms = (ended - started) * 1000.0

    return {
        "request_id": payload.get("id") or decision.get("decision_id"),
        "ts": datetime.now(UTC),
        "tenant": (payload.get("metadata") or {}).get("user_api_key_team_id"),
        "session_id": None,
        "policy_version": decision.get("policy_version"),
        "decision_id": decision.get("decision_id"),
        "chosen_arm": chosen,
        "propensity": decision.get("propensity"),
        "explore": decision.get("explore"),
        "eligible_set": list(decision.get("eligible_set") or ()),
        "features": json_safe(decision.get("features")),
        "served_arm": served,
        "model_group": payload.get("model_group"),
        "fallback_depth": fallback_depth,
        "status": payload.get("status") or "success",
        "cost_usd": payload.get("response_cost"),
        "latency_ms": latency_ms,
        "tokens_in": payload.get("prompt_tokens"),
        "tokens_out": payload.get("completion_tokens"),
    }
