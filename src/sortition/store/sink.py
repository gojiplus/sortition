"""Durable, non-blocking log writing.

Two constraints shape this module, and both come from where it runs: inside
someone else's proxy, on the path of every request.

**It must never block the caller.** ``submit`` appends to an in-memory buffer and
returns. All disk work happens on a dedicated writer thread, so a parquet write
cannot stall the event loop of the gateway sortition is embedded in.

**It must not lose the log.** LiteLLM's ``CustomLogger`` exposes no shutdown hook
-- verified against the source, there is nothing to register a graceful drain
with -- so a periodic flush is not a refinement here, it is the entire durability
story. The contract is bounded loss: a hard kill loses at most ``flush_interval``
of buffered rows, which is the same guarantee the Datadog and OpenTelemetry
agents give. A write-ahead log would be a bigger machine than routing telemetry
warrants.

A thread rather than an asyncio task, deliberately. LiteLLM's own logging worker
binds its queue to the running event loop and silently stops draining when that
loop is replaced; a thread has no such failure mode, and the same code then works
in synchronous scripts and tests without a loop at all.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from sortition import metrics

logger = logging.getLogger(__name__)

DECISIONS = "decisions"
OUTCOMES = "outcomes"
TABLES = (DECISIONS, OUTCOMES)

DEFAULT_FLUSH_INTERVAL = 5.0
DEFAULT_FLUSH_EVERY = 200
# Telemetry must not be able to exhaust the memory of the process it observes.
DEFAULT_MAX_BUFFERED = 50_000


class Sink(Protocol):
    """Where log rows go. Implementations must not block ``submit``."""

    def submit(self, table: str, row: dict[str, Any]) -> None:
        """Accept a row for eventual durable storage.

        Must never block and must never raise: it is called from a request path.

        Args:
            table: Which table the row belongs to.
            row: The row itself.
        """
        ...

    def flush(self) -> None:
        """Write everything buffered, blocking until it is durable."""
        ...

    def close(self) -> None:
        """Stop accepting rows and flush what remains."""
        ...


class BufferedSink:
    """Buffering, backpressure and a writer thread, shared by every sink.

    Subclasses implement :meth:`_write_part` and inherit the durability
    contract: ``submit`` never blocks or raises, a flush happens on whichever
    comes first of an interval or a full buffer, and overflow is dropped and
    counted rather than growing without bound.

    The contract exists once here because three copies of it would drift, and
    the failure mode of a drifted copy is silent log loss.
    """

    def __init__(
        self,
        *,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        flush_every: int = DEFAULT_FLUSH_EVERY,
        max_buffered: int = DEFAULT_MAX_BUFFERED,
        on_write: Callable[[str, int], None] | None = None,
        on_drop: Callable[[str, int], None] | None = None,
    ) -> None:
        """Configure buffering and the writer thread.

        Args:
            flush_interval: Seconds between periodic flushes. This is what bounds
                data loss on a hard kill, so it is the number to tune.
            flush_every: Buffered rows that trigger an immediate flush.
            max_buffered: Cap across all tables. Rows beyond it are dropped and
                counted rather than growing the buffer without limit.
            on_write: Called with (table, n_rows) after a successful write.
            on_drop: Called with (table, n_rows) when rows are dropped.
        """
        self.flush_interval = max(0.1, flush_interval)
        self.flush_every = max(1, flush_every)
        self.max_buffered = max(1, max_buffered)
        self._on_write = on_write or (
            lambda table, n: metrics.sink_rows.labels(table=table).inc(n)
        )
        self._on_drop = on_drop or (
            lambda table, n: metrics.sink_dropped.labels(table=table).inc(n)
        )

        self._buffers: dict[str, list[dict[str, Any]]] = {t: [] for t in TABLES}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()

        self.rows_written = 0
        self.rows_dropped = 0
        self.flushes = 0

        atexit.register(self.close)

    # -- producer side -----------------------------------------------------

    def submit(self, table: str, row: dict[str, Any]) -> None:
        """Buffer a row. Never blocks on disk, never raises.

        Args:
            table: Which table the row belongs to.
            row: The row itself.
        """
        try:
            with self._lock:
                if self._buffered_unlocked() >= self.max_buffered:
                    self.rows_dropped += 1
                    dropped = self.rows_dropped
                    should_warn = dropped == 1 or dropped % 1000 == 0
                    if self._on_drop is not None:
                        self._on_drop(table, 1)
                    if should_warn:
                        logger.warning(
                            "sortition sink buffer is full (%d rows); dropped %d "
                            "rows so far. The writer thread is not keeping up.",
                            self.max_buffered,
                            dropped,
                        )
                    return
                self._buffers.setdefault(table, []).append(row)
                ready = len(self._buffers[table]) >= self.flush_every

            self._ensure_worker()
            if ready:
                self._wake.set()
        except Exception:
            # A telemetry sink may never take down the request it is observing.
            logger.exception("sortition sink failed to buffer a row")

    def _buffered_unlocked(self) -> int:
        return sum(len(buffer) for buffer in self._buffers.values())

    @property
    def buffered(self) -> int:
        """Rows currently waiting to be written."""
        with self._lock:
            return self._buffered_unlocked()

    # -- writer thread -----------------------------------------------------

    def _ensure_worker(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        with self._thread_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._stopping.is_set():
                return
            # Daemon so it can never keep a process alive; atexit does the
            # final drain, and it runs before daemon threads are killed.
            self._thread = threading.Thread(
                target=self._run,
                name="sortition-sink",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        while not self._stopping.is_set():
            # Whichever comes first: the interval elapses, or a full buffer
            # wakes us early.
            self._wake.wait(self.flush_interval)
            self._wake.clear()
            self._drain()
        self._drain()

    def _drain(self) -> None:
        for table in list(self._buffers):
            with self._lock:
                rows = self._buffers.get(table) or []
                if not rows:
                    continue
                self._buffers[table] = []
            try:
                self._write_part(table, rows)
                self.rows_written += len(rows)
                self.flushes += 1
                if self._on_write is not None:
                    self._on_write(table, len(rows))
            except Exception:
                self.rows_dropped += len(rows)
                if self._on_drop is not None:
                    self._on_drop(table, len(rows))
                # Deliberately not re-buffered: a failing disk would otherwise
                # fill memory retrying rows that will keep failing.
                logger.exception(
                    "sortition sink failed to write %d %s rows; they are lost",
                    len(rows),
                    table,
                )

    def _write_part(self, table: str, rows: list[dict[str, Any]]) -> None:
        """Persist one batch. Subclasses implement this; it runs on the thread.

        Args:
            table: Which table the rows belong to.
            rows: The batch to persist.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError

    # -- lifecycle ---------------------------------------------------------

    def flush(self) -> None:
        """Write everything buffered, blocking until it is on disk."""
        self._drain()

    def close(self) -> None:
        """Stop the writer thread and flush what remains. Idempotent."""
        if self._stopping.is_set():
            return
        self._stopping.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(5.0, self.flush_interval * 2))
        # Drain again regardless: if the join timed out, these rows would
        # otherwise be lost silently.
        self._drain()

    async def aflush(self) -> None:
        """Flush without blocking the event loop."""
        import asyncio

        await asyncio.to_thread(self.flush)

    async def aclose(self) -> None:
        """Close without blocking the event loop."""
        import asyncio

        await asyncio.to_thread(self.close)

    def __enter__(self) -> BufferedSink:
        """Enter a context that closes on exit.

        Returns:
            This sink.
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """Flush and stop the writer thread."""
        self.close()


class ParquetSink(BufferedSink):
    """Date-partitioned parquet on a local filesystem. The default.

    Parts are named per process and per flush, so several proxy replicas can
    write one dataset without coordinating.
    """

    def __init__(self, directory: str | Path, **kwargs: Any) -> None:
        """Open a sink rooted at ``directory``.

        Args:
            directory: Root of the dataset; one subdirectory per table.
            **kwargs: Buffering options, see :class:`BufferedSink`.
        """
        super().__init__(**kwargs)
        self.directory = Path(directory)

    def _write_part(self, table: str, rows: list[dict[str, Any]]) -> None:
        """Write one parquet part.

        Args:
            table: Which table the rows belong to.
            rows: The batch to persist.
        """
        import polars as pl

        # Partitioned by date so a query for last week does not read last year.
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        target = self.directory / table / f"dt={day}"
        target.mkdir(parents=True, exist_ok=True)
        name = f"part-{os.getpid()}-{uuid.uuid4().hex[:12]}.parquet"
        # Written beside the final name and renamed, so a reader globbing the
        # directory can never observe a half-written file.
        tmp = target / f".{name}.tmp"
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(tmp)
        tmp.rename(target / name)
        logger.debug("wrote %d %s rows to %s", len(rows), table, target / name)


class S3Sink(BufferedSink):
    """Date-partitioned parquet in object storage.

    What a fleet needs: several proxy replicas writing one dataset that outlives
    any of them. Keys mirror the local layout, so the same duckdb read path works
    against ``s3://bucket/prefix/decisions/**/*.parquet``.

    Requires the ``s3`` extra. Credentials come from the environment via the
    usual boto3 chain -- sortition holds none of its own, here as everywhere.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        *,
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        """Open a sink writing under ``s3://bucket/prefix``.

        Args:
            bucket: Destination bucket.
            prefix: Key prefix, typically an environment name.
            client: A preconfigured boto3 S3 client. Built from the default
                session when omitted; injectable for tests.
            **kwargs: Buffering options, see :class:`BufferedSink`.
        """
        super().__init__(**kwargs)
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = client

    @property
    def client(self) -> Any:
        """The S3 client, created on first use.

        Returns:
            A boto3 S3 client.
        """
        if self._client is None:
            import boto3

            self._client = boto3.client("s3")
        return self._client

    def _key(self, table: str) -> str:
        day = datetime.now(UTC).strftime("%Y-%m-%d")
        name = f"part-{os.getpid()}-{uuid.uuid4().hex[:12]}.parquet"
        parts = [p for p in (self.prefix, table, f"dt={day}", name) if p]
        return "/".join(parts)

    def _write_part(self, table: str, rows: list[dict[str, Any]]) -> None:
        """Upload one parquet part.

        Args:
            table: Which table the rows belong to.
            rows: The batch to persist.
        """
        import io

        import polars as pl

        # Serialized in memory and uploaded in one call: an object appears whole
        # or not at all, so a reader never sees a partial part.
        buffer = io.BytesIO()
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(buffer)
        key = self._key(table)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=buffer.getvalue())
        logger.debug(
            "wrote %d %s rows to s3://%s/%s", len(rows), table, self.bucket, key
        )


class PostgresSink(BufferedSink):
    """Rows in Postgres, for deployments that already run one.

    Useful when routing logs want to sit beside the spend logs a LiteLLM proxy
    is already writing, and when someone would rather query them with SQL than
    reach for parquet.

    Nested values -- the feature dict, the eligible set -- are stored as JSONB,
    and ``sortition.frame`` already reads a features column that arrives as
    JSON, so nothing downstream changes.

    Requires the ``postgres`` extra.
    """

    DDL = """
    CREATE TABLE IF NOT EXISTS {table} (
        id BIGSERIAL PRIMARY KEY,
        written_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        row JSONB NOT NULL
    )
    """

    def __init__(
        self,
        dsn: str,
        *,
        schema: str = "sortition",
        connect: Any = None,
        **kwargs: Any,
    ) -> None:
        """Open a sink writing to Postgres.

        Args:
            dsn: libpq connection string.
            schema: Schema holding the tables. Created if absent.
            connect: Callable returning a DB-API connection. Injectable for
                tests; defaults to ``psycopg.connect``.
            **kwargs: Buffering options, see :class:`BufferedSink`.
        """
        super().__init__(**kwargs)
        self.dsn = dsn
        self.schema = schema
        self._connect = connect
        self._ready: set[str] = set()

    def _connection(self) -> Any:
        if self._connect is not None:
            return self._connect(self.dsn)
        import psycopg

        return psycopg.connect(self.dsn)

    def _ensure(self, cursor: Any, table: str) -> str:
        qualified = f"{self.schema}.{table}"
        if table not in self._ready:
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            cursor.execute(self.DDL.format(table=qualified))
            self._ready.add(table)
        return qualified

    def _write_part(self, table: str, rows: list[dict[str, Any]]) -> None:
        """Insert one batch.

        Args:
            table: Which table the rows belong to.
            rows: The batch to persist.
        """
        import json

        # One transaction per batch: the batch lands whole or not at all, which
        # matches the parquet sink's all-or-nothing part file.
        with self._connection() as conn, conn.cursor() as cursor:
            qualified = self._ensure(cursor, table)
            cursor.executemany(
                f"INSERT INTO {qualified} (row) VALUES (%s)",  # noqa: S608
                [(json.dumps(row, default=str),) for row in rows],
            )
            conn.commit()
        logger.debug("wrote %d %s rows to %s", len(rows), table, table)
