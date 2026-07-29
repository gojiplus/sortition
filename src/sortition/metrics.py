"""Prometheus metrics, or nothing at all.

Every metric here degrades to a no-op when ``prometheus_client`` is not
installed, so the rest of the package calls them unconditionally and the base
install stays free of an observability dependency it may not want.

The metric worth having most is ``sortition_plugin_errors_total``. The plugin
catches its own exceptions and lets the request through untouched, which is the
right behaviour and also means a broken policy looks exactly like a healthy one:
requests succeed, latency is normal, and the log quietly stops being evaluable.
A counter is the only thing that distinguishes those two states from outside.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

NAMESPACE = "sortition"

_FACTORIES: dict[str, Any] = {}
try:  # pragma: no cover - exercised by whether the extra is installed
    from prometheus_client import Counter, Gauge, Histogram

    _FACTORIES = {"counter": Counter, "gauge": Gauge, "histogram": Histogram}
    AVAILABLE = True
except ImportError:  # pragma: no cover
    AVAILABLE = False


class _NoOp:
    """Stands in for a metric when prometheus_client is absent."""

    def labels(self, *args: Any, **kwargs: Any) -> _NoOp:
        """Return self, so chained calls keep working.

        Args:
            *args: Ignored label values.
            **kwargs: Ignored label values.

        Returns:
            This same no-op.
        """
        return self

    def inc(self, *args: Any, **kwargs: Any) -> None:
        """Ignore an increment."""

    def observe(self, *args: Any, **kwargs: Any) -> None:
        """Ignore an observation."""

    def set(self, *args: Any, **kwargs: Any) -> None:
        """Ignore a gauge write."""

    def info(self, *args: Any, **kwargs: Any) -> None:
        """Ignore an info write."""


def _metric(
    kind: str, name: str, doc: str, labels: list[str] | None = None, **kw: Any
) -> Any:
    if not AVAILABLE:
        return _NoOp()
    factory = _FACTORIES[kind]
    try:
        return factory(f"{NAMESPACE}_{name}", doc, labels or [], **kw)
    except ValueError:
        # Duplicate registration: the module was imported twice, typically under
        # a test reload. A no-op is better than crashing the proxy at import.
        logger.debug("metric %s already registered", name)
        return _NoOp()


# -- decisions -------------------------------------------------------------

decisions = _metric(
    "counter",
    "decisions_total",
    "Routing decisions, by arm and policy.",
    ["arm", "policy_version", "explore"],
)
decision_seconds = _metric(
    "histogram",
    "decision_seconds",
    "Time spent choosing an arm.",
    buckets=(0.0001, 0.00025, 0.0005, 0.001, 0.005, 0.01, 0.05),
)
propensity = _metric(
    "histogram",
    "propensity",
    "Logged selection probability. Mass at 1.0 means exploration has collapsed.",
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0),
)
plugin_errors = _metric(
    "counter",
    "plugin_errors_total",
    "Times the routing plugin failed and let the request through unrouted.",
)

# -- execution -------------------------------------------------------------

fallbacks = _metric(
    "counter",
    "fallbacks_total",
    "Requests the gateway rerouted after the decision, making them unevaluable.",
    ["chosen_arm", "served_arm"],
)
cost_usd = _metric("counter", "cost_usd_total", "Spend, by arm.", ["arm"])
tokens = _metric("counter", "tokens_total", "Tokens, by arm.", ["arm", "direction"])

# -- sink ------------------------------------------------------------------

sink_rows = _metric("counter", "sink_rows_total", "Log rows written.", ["table"])
sink_dropped = _metric(
    "counter",
    "sink_dropped_total",
    "Log rows dropped, from a full buffer or a failed write.",
    ["table"],
)
sink_buffered = _metric("gauge", "sink_buffered", "Log rows waiting to be written.")

# -- policy ----------------------------------------------------------------

policy_reloads = _metric(
    "counter", "policy_reloads_total", "Successful policy reloads."
)
policy_reload_errors = _metric(
    "counter",
    "policy_reload_errors_total",
    "Failed policy reloads. The previous policy keeps serving.",
)

# -- health ----------------------------------------------------------------

health_ok = _metric("gauge", "health_ok", "1 when the log can support estimates.")
exploration_rate = _metric(
    "gauge", "exploration_rate", "Share of recent requests that explored."
)
leakage_rate = _metric(
    "gauge", "leakage_rate", "Share of recent requests the gateway rerouted."
)
effective_sample_fraction = _metric(
    "gauge",
    "effective_sample_fraction",
    "Effective sample size as a fraction of rows, against a uniform target.",
)


def observe_decision(
    *,
    arm: str,
    policy_version: str,
    explore: bool,
    propensity_value: float,
    seconds: float,
) -> None:
    """Record one routing decision.

    Args:
        arm: The arm chosen.
        policy_version: The policy that chose it.
        explore: Whether exploration fired.
        propensity_value: The logged selection probability.
        seconds: Time spent deciding.
    """
    decisions.labels(arm=arm, policy_version=policy_version, explore=str(explore)).inc()
    propensity.observe(propensity_value)
    decision_seconds.observe(seconds)


def observe_execution(
    *,
    chosen_arm: str | None,
    served_arm: str | None,
    fallback_depth: int,
    cost: float | None,
    tokens_in: int | None,
    tokens_out: int | None,
) -> None:
    """Record what the gateway actually did.

    Args:
        chosen_arm: The arm sortition sampled.
        served_arm: The arm that actually served.
        fallback_depth: 0 when the chosen arm served.
        cost: Request cost in USD, if known.
        tokens_in: Prompt tokens, if known.
        tokens_out: Completion tokens, if known.
    """
    if fallback_depth > 0:
        fallbacks.labels(
            chosen_arm=chosen_arm or "unknown", served_arm=served_arm or "none"
        ).inc()
    arm = served_arm or chosen_arm or "unknown"
    if cost:
        cost_usd.labels(arm=arm).inc(cost)
    if tokens_in:
        tokens.labels(arm=arm, direction="in").inc(tokens_in)
    if tokens_out:
        tokens.labels(arm=arm, direction="out").inc(tokens_out)
