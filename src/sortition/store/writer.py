"""Append-only parquet, in two tables that are joined at read time.

Decisions and outcomes live apart because they arrive apart. A thumbs-up lands
in seconds, a resolution flag or CSAT score in hours, and an upsert into the
decision row would make a historical estimate irreproducible: re-running last
month's report would silently give a different answer than it did last month.

Parquet has no true append, so each flush writes a new part file and readers glob
the directory. That is the standard partitioned-dataset pattern, and it means a
crashed process loses at most one buffer rather than corrupting the log.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)

DECISIONS = "decisions"
OUTCOMES = "outcomes"


class LogStore:
    """Buffers routing rows and flushes them to a partitioned parquet dataset."""

    def __init__(self, directory: str | Path, *, flush_every: int = 200) -> None:
        """Open a store rooted at ``directory``.

        Args:
            directory: Root of the dataset. Subdirectories are created per table.
            flush_every: Rows to buffer before writing a part file. Smaller means
                less data at risk on a crash and more, smaller files.
        """
        self.directory = Path(directory)
        self.flush_every = max(1, flush_every)
        self._buffers: dict[str, list[dict[str, Any]]] = {DECISIONS: [], OUTCOMES: []}
        self._lock = threading.Lock()

    def write_decision(self, row: dict[str, Any]) -> None:
        """Append one resolved request.

        Args:
            row: A flat log row: the decision, its propensity, and what the
                gateway actually did.
        """
        self._append(DECISIONS, row)

    def write_outcome(self, row: dict[str, Any]) -> None:
        """Append one outcome, which may arrive long after the decision.

        Args:
            row: A row carrying at least ``request_id``, ``name`` and ``value``.
        """
        self._append(OUTCOMES, row)

    def _append(self, table: str, row: dict[str, Any]) -> None:
        with self._lock:
            buffer = self._buffers[table]
            buffer.append(row)
            ready = len(buffer) >= self.flush_every
        if ready:
            self.flush(table)

    def flush(self, table: str | None = None) -> None:
        """Write buffered rows to disk.

        Args:
            table: Flush only this table, or all tables when omitted.
        """
        for name in (table,) if table else (DECISIONS, OUTCOMES):
            with self._lock:
                rows, self._buffers[name] = self._buffers[name], []
            if rows:
                self._write_part(name, rows)

    def _write_part(self, table: str, rows: list[dict[str, Any]]) -> None:
        import polars as pl

        target = self.directory / table
        target.mkdir(parents=True, exist_ok=True)
        # A unique name per part: several processes may write the same dataset,
        # and a collision would silently drop one of their logs.
        path = target / f"part-{os.getpid()}-{uuid.uuid4().hex[:12]}.parquet"
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(path)
        logger.debug("wrote %d rows to %s", len(rows), path)

    def read(self, table: str = DECISIONS) -> pl.DataFrame:
        """Read a table back, flushing anything still buffered.

        Args:
            table: Which table to read.

        Returns:
            Every row written so far.

        Raises:
            FileNotFoundError: If nothing has ever been written to the table.
        """
        import polars as pl

        self.flush(table)
        target = self.directory / table
        parts = sorted(target.glob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"no {table} written under {target}")
        return pl.concat([pl.read_parquet(p) for p in parts], how="diagonal_relaxed")

    def load(self) -> pl.DataFrame:
        """Read decisions with any outcomes joined on.

        Outcomes are pivoted so each named outcome becomes its own column, which
        is the shape ``sortition.frame.to_arrays`` expects. Requests whose
        outcome has not arrived keep a null, and the estimators drop those rows
        for that metric rather than imputing them.

        Returns:
            The joined log table.
        """
        decisions = self.read(DECISIONS)
        try:
            outcomes = self.read(OUTCOMES)
        except FileNotFoundError:
            return decisions

        # Last write wins for a given (request_id, name): outcomes are corrected
        # more often than they are duplicated.
        latest = outcomes.group_by(["request_id", "name"]).last()
        wide = latest.pivot(on="name", index="request_id", values="value")
        return decisions.join(wide, on="request_id", how="left")

    def __enter__(self) -> LogStore:
        """Enter a context that flushes on exit.

        Returns:
            This store.
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """Flush buffered rows when leaving the context."""
        self.flush()


def json_safe(value: Any) -> Any:
    """Coerce a value into something parquet can hold.

    Args:
        value: Any logged value.

    Returns:
        The value unchanged when it is a scalar, otherwise a JSON string.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return json.dumps(value, default=str)
