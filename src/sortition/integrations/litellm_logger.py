"""The LiteLLM callback that persists routing decisions.

Unlike the rest of the package, this module imports litellm at module level. It
has to: ``litellm_logging.py`` dispatches callbacks with
``isinstance(callback, CustomLogger)``, so an object that merely exposes the
right method names is silently never called. That was found by an integration
test rather than by reading the docs, which is why the test exists.

The pure payload-to-row conversion lives in ``adapters`` and stays importable
without litellm, so logs exported from a gateway can be converted anywhere.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

from sortition import metrics
from sortition.integrations.adapters import read_decision, to_log_row
from sortition.store.writer import LogStore

logger = logging.getLogger(__name__)


class SortitionLogger(CustomLogger):  # type: ignore[misc]
    """Writes one evaluable row per resolved request."""

    def __init__(self, store: LogStore) -> None:
        """Point the logger at a store.

        Args:
            store: Where rows are appended.
        """
        super().__init__()
        self.store = store

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Record a served request.

        Args:
            kwargs: LiteLLM's callback kwargs.
            response_obj: The provider response. Unused.
            start_time: Call start. Unused; timings come from the payload.
            end_time: Call end. Unused.
        """
        self._record(kwargs, status="success")

    async def async_log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Record a request that failed outright.

        Failures are logged rather than dropped: a policy whose chosen arms fail
        more often is a worse policy, and discarding those rows would hide it.

        Args:
            kwargs: LiteLLM's callback kwargs.
            response_obj: The exception or error payload. Unused.
            start_time: Call start. Unused.
            end_time: Call end. Unused.
        """
        self._record(kwargs, status="failure")

    def _record(self, kwargs: dict[str, Any], *, status: str) -> None:
        try:
            decision = read_decision(kwargs)
            if decision is None:
                return
            row = to_log_row(kwargs, decision)
            row["status"] = status
            self.store.write_decision(row)
            metrics.observe_execution(
                chosen_arm=row.get("chosen_arm"),
                served_arm=row.get("served_arm"),
                fallback_depth=int(row.get("fallback_depth") or 0),
                cost=row.get("cost_usd"),
                tokens_in=row.get("tokens_in"),
                tokens_out=row.get("tokens_out"),
            )
        except Exception:
            # Same contract as the plugin: logging must never fail a request.
            logger.exception("sortition failed to record a routing decision")

    def record_outcome(
        self,
        request_id: str,
        value: float,
        *,
        name: str = "outcome",
        bounded: bool = True,
    ) -> None:
        """Attach a late-arriving outcome to a logged request.

        Args:
            request_id: The request the outcome belongs to.
            value: The outcome value.
            name: Which outcome this is, when several are collected.
            bounded: Whether the value lies in [0, 1], which selects the
                interval method at evaluation time.
        """
        self.store.write_outcome(
            {
                "request_id": request_id,
                "ts": datetime.now(UTC),
                "name": name,
                "value": float(value),
                "bounded": bounded,
            }
        )
