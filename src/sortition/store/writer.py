"""Reading routing logs back, and the store that ties writing to reading.

Writes go through a :class:`~sortition.store.sink.Sink`; this module owns the
query side.

Reads go through duckdb rather than concatenating every parquet file. The dataset
is partitioned by date, so ``since``/``until`` become partition pruning: asking
about last week reads last week, not the whole history. The previous
implementation globbed and concatenated every part ever written, which was fine
for a test fixture and would not survive a month of production traffic.

Decisions and outcomes stay separate tables, joined at read time. They arrive
apart -- a thumbs-up in seconds, a resolution flag in hours -- and mutating a
decision row when its outcome lands would make a historical estimate
irreproducible: re-running last month's report would quietly give a different
answer than it did last month.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sortition.store.sink import DECISIONS, OUTCOMES, ParquetSink, Sink

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)

TimeBound = str | date | datetime | None


def _as_date(value: TimeBound) -> str | None:
    """Normalize a time bound to the ``dt=`` partition format.

    Args:
        value: A date, datetime, ``YYYY-MM-DD`` string, or None.

    A non-ISO string propagates ``ValueError`` from ``date.fromisoformat``,
    which is the right failure: a mistyped bound would otherwise silently widen
    or narrow the window.

    Returns:
        A ``YYYY-MM-DD`` string, or None when unbounded.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return date.fromisoformat(str(value)[:10]).isoformat()


def _pad(day: str | None, days: int) -> str | None:
    """Widen a partition bound by a day.

    Partitions are stamped in UTC, but callers naturally pass local dates --
    ``date.today()`` can be a day behind the partition a row landed in. Since
    the ``dt`` filter is only a coarse pre-filter over directories, widening it
    by one day on each side costs at most two extra partitions and removes any
    chance of a timezone edge silently dropping rows. Filter on ``ts`` for an
    exact window.

    Args:
        day: A ``YYYY-MM-DD`` bound, or None.
        days: Offset to apply, negative to widen a lower bound.

    Returns:
        The widened bound, or None when unbounded.
    """
    if day is None:
        return None
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


# One literal query per bound combination. Written out in full rather than
# assembled from fragments: every value is already a bound parameter, and a
# constant string is both verifiable by eye and unambiguous to duckdb's
# partition pruner.
_READ = "SELECT * FROM read_parquet(?, hive_partitioning = 1, union_by_name = 1)"
_RANGE_QUERIES: dict[tuple[bool, bool], str] = {
    (False, False): _READ,
    (True, False): _READ + " WHERE dt >= ?",
    (False, True): _READ + " WHERE dt <= ?",
    (True, True): _READ + " WHERE dt >= ? AND dt <= ?",
}


class LogStore:
    """Writes through a sink, reads through duckdb."""

    def __init__(
        self,
        directory: str | Path,
        *,
        sink: Sink | None = None,
        **sink_kwargs: Any,
    ) -> None:
        """Open a store rooted at ``directory``.

        Args:
            directory: Root of the dataset.
            sink: Where writes go. Defaults to a :class:`ParquetSink` over the
                same directory.
            **sink_kwargs: Passed to the default sink, e.g. ``flush_interval``.
        """
        self.directory = Path(directory)
        self.sink: Sink = (
            sink if sink is not None else ParquetSink(directory, **sink_kwargs)
        )

    def write_decision(self, row: dict[str, Any]) -> None:
        """Record one resolved request.

        Args:
            row: A flat log row: the decision, its propensity, and what the
                gateway actually did.
        """
        self.sink.submit(DECISIONS, row)

    def write_outcome(self, row: dict[str, Any]) -> None:
        """Record one outcome, which may arrive long after the decision.

        Args:
            row: A row carrying at least ``request_id``, ``name`` and ``value``.
        """
        self.sink.submit(OUTCOMES, row)

    def flush(self) -> None:
        """Write everything buffered, blocking until it is durable."""
        self.sink.flush()

    def close(self) -> None:
        """Flush and stop the sink."""
        self.sink.close()

    # -- read side ---------------------------------------------------------

    def _glob(self, table: str) -> str:
        return str(self.directory / table / "**" / "*.parquet")

    def has(self, table: str) -> bool:
        """Whether anything has been written to a table.

        Args:
            table: Which table to check.

        Returns:
            True if at least one part file exists.
        """
        return any((self.directory / table).glob("**/*.parquet"))

    def read(
        self,
        table: str = DECISIONS,
        *,
        since: TimeBound = None,
        until: TimeBound = None,
    ) -> pl.DataFrame:
        """Read one table, pruned to a date range.

        Args:
            table: Which table to read.
            since: Inclusive lower bound on the partition date.
            until: Inclusive upper bound on the partition date.

        Returns:
            The matching rows.

        Raises:
            FileNotFoundError: If nothing has ever been written to the table.
        """
        import duckdb

        self.flush()
        if not self.has(table):
            raise FileNotFoundError(
                f"no {table} written under {self.directory / table}"
            )

        # hive_partitioning exposes dt as a column, which is what lets duckdb
        # skip whole directories instead of reading and filtering them.
        # The four queries are written out as constants rather than assembled
        # from fragments: the bounds are already parameterized, and a literal
        # query is easier to confirm by eye than a string built at runtime.
        low, high = _pad(_as_date(since), -1), _pad(_as_date(until), +1)
        sql = _RANGE_QUERIES[(low is not None, high is not None)]
        params = [b for b in (low, high) if b is not None]
        with duckdb.connect() as conn:
            return conn.execute(sql, [self._glob(table), *params]).pl()

    def load(
        self,
        *,
        since: TimeBound = None,
        until: TimeBound = None,
    ) -> pl.DataFrame:
        """Read decisions with any outcomes joined on.

        Each named outcome becomes its own column, which is the shape
        ``sortition.frame.to_arrays`` reads. A request whose outcome has not
        arrived keeps a null, and the estimators drop it for that metric rather
        than imputing a value.

        Args:
            since: Inclusive lower bound on the partition date.
            until: Inclusive upper bound on the partition date.

        Returns:
            The joined log table.
        """
        decisions = self.read(DECISIONS, since=since, until=until)
        if not self.has(OUTCOMES):
            return decisions

        # Outcomes are not date-pruned: one that arrives on Tuesday may belong to
        # Monday's request, and pruning them to the decision window would drop
        # exactly the late labels this schema exists to accommodate.
        outcomes = self.read(OUTCOMES)
        if outcomes.is_empty():
            return decisions

        # Last write wins per (request_id, name): outcomes get corrected more
        # often than they get duplicated.
        latest = outcomes.sort("ts").group_by(["request_id", "name"]).last()
        wide = latest.pivot(on="name", index="request_id", values="value")
        return decisions.join(wide, on="request_id", how="left")

    def __enter__(self) -> LogStore:
        """Enter a context that closes the sink on exit.

        Returns:
            This store.
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """Flush and stop the sink."""
        self.close()


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
