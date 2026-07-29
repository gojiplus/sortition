"""Durability and non-blocking behaviour of the log sink.

The first class is the one that matters. Before this, ``LogStore`` flushed only
at 200 buffered rows or on ``__exit__``, and a proxy never leaves that context --
so every restart silently dropped up to 200 rows and a quiet tenant could flush
never. A test that only checked "rows written then read back in one process"
passes happily against that bug, which is why the durability test kills a real
subprocess instead.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sortition.store import DECISIONS, LogStore, ParquetSink


def _row(i: int) -> dict[str, object]:
    return {
        "request_id": f"r-{i:06d}",
        "chosen_arm": "premium" if i % 2 else "cheap",
        "propensity": 0.9 if i % 2 else 0.1,
        "eligible_set": ["cheap", "premium"],
        "outcome": float(i % 2),
        "cost_usd": 0.001 * (i % 5 + 1),
    }


class TestDurability:
    def test_periodic_flush_writes_without_reaching_the_size_threshold(
        self, tmp_path: Path
    ) -> None:
        # The regression: a low-traffic tenant sits below flush_every forever.
        # The interval, not the row count, is what has to save them.
        sink = ParquetSink(tmp_path, flush_interval=0.2, flush_every=10_000)
        try:
            for i in range(5):
                sink.submit(DECISIONS, _row(i))
            assert sink.buffered == 5
            deadline = time.monotonic() + 5.0
            while sink.rows_written < 5 and time.monotonic() < deadline:
                time.sleep(0.05)
            assert sink.rows_written == 5
            assert sink.buffered == 0
        finally:
            sink.close()

    def test_survives_a_hard_kill(self, tmp_path: Path) -> None:
        """SIGKILL a real process and assert the flushed rows are on disk.

        No atexit, no context manager, no graceful shutdown -- exactly what a
        proxy pod gets when it is evicted.
        """
        script = textwrap.dedent(f"""
            import time
            from sortition.store import ParquetSink, DECISIONS
            sink = ParquetSink(
                {str(tmp_path)!r}, flush_interval=0.2, flush_every=10_000
            )
            for i in range(50):
                sink.submit(DECISIONS, {{"request_id": f"r-{{i}}", "chosen_arm": "a",
                                         "propensity": 0.5, "eligible_set": ["a"]}})
            print("submitted", flush=True)
            time.sleep(30)
        """)
        proc = subprocess.Popen(  # noqa: S603 -- our own interpreter, literal script
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert proc.stdout is not None
            assert proc.stdout.readline().strip() == "submitted"
            # Long enough for at least one periodic flush.
            time.sleep(1.5)
        finally:
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)

        store = LogStore(tmp_path, flush_interval=0.2)
        try:
            assert store.read(DECISIONS).height == 50
        finally:
            store.close()

    def test_close_flushes_what_remains(self, tmp_path: Path) -> None:
        sink = ParquetSink(tmp_path, flush_interval=3600.0, flush_every=10_000)
        for i in range(7):
            sink.submit(DECISIONS, _row(i))
        sink.close()
        assert sink.rows_written == 7

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        sink = ParquetSink(tmp_path, flush_interval=3600.0)
        sink.submit(DECISIONS, _row(0))
        sink.close()
        sink.close()
        assert sink.rows_written == 1

    def test_partial_files_are_never_visible(self, tmp_path: Path) -> None:
        # Parts are written to a dotfile and renamed, so a reader globbing the
        # directory cannot pick up a half-written parquet.
        with LogStore(tmp_path, flush_interval=3600.0) as store:
            for i in range(20):
                store.write_decision(_row(i))
            store.flush()
            parts = list((tmp_path / DECISIONS).glob("**/*.parquet"))
            assert parts
            assert not any(p.name.startswith(".") for p in parts)


class TestNonBlocking:
    def test_submit_does_not_touch_disk(self, tmp_path: Path) -> None:
        # submit() is called from a request path. If it writes, it blocks the
        # gateway's event loop.
        sink = ParquetSink(tmp_path, flush_interval=3600.0, flush_every=10_000)
        try:
            start = time.perf_counter()
            for i in range(2_000):
                sink.submit(DECISIONS, _row(i))
            elapsed = time.perf_counter() - start
            # Buffering 2,000 dicts is pure memory work; a disk write would be
            # orders of magnitude slower than this bound.
            assert elapsed < 0.5
            assert not list((tmp_path / DECISIONS).glob("**/*.parquet"))
        finally:
            sink.close()

    def test_latency_stays_flat_across_a_flush_boundary(self, tmp_path: Path) -> None:
        # The old code wrote parquet inline every flush_every rows, which showed
        # up as a spike on that request. Nothing should spike now.
        sink = ParquetSink(tmp_path, flush_interval=3600.0, flush_every=50)
        try:
            timings = []
            for i in range(500):
                start = time.perf_counter()
                sink.submit(DECISIONS, _row(i))
                timings.append(time.perf_counter() - start)
            typical = sorted(timings)[len(timings) // 2]
            worst = max(timings)
            # Generous: this is asserting the absence of a synchronous parquet
            # write, which costs milliseconds, not that scheduling is uniform.
            assert worst < max(typical * 200, 0.02)
        finally:
            sink.close()

    def test_is_safe_under_concurrent_writers(self, tmp_path: Path) -> None:
        sink = ParquetSink(tmp_path, flush_interval=0.1, flush_every=100)
        try:

            def writer(offset: int) -> None:
                for i in range(200):
                    sink.submit(DECISIONS, _row(offset + i))

            threads = [
                threading.Thread(target=writer, args=(t * 1000,)) for t in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            sink.close()
            assert sink.rows_written + sink.rows_dropped == 1_600
        finally:
            sink.close()


class TestBackpressure:
    def test_drops_and_counts_when_the_buffer_is_full(self, tmp_path: Path) -> None:
        # Telemetry must not be able to exhaust the memory of the process it
        # observes, so overflow drops rather than grows.
        sink = ParquetSink(
            tmp_path, flush_interval=3600.0, flush_every=10_000, max_buffered=100
        )
        try:
            for i in range(250):
                sink.submit(DECISIONS, _row(i))
            assert sink.buffered == 100
            assert sink.rows_dropped == 150
        finally:
            sink.close()

    def test_submit_never_raises(self, tmp_path: Path) -> None:
        sink = ParquetSink(tmp_path, flush_interval=3600.0)
        try:
            # An unserializable value must not propagate out of a request path.
            sink.submit(DECISIONS, {"request_id": "r-1", "bad": object()})
            sink.flush()
        finally:
            sink.close()


class TestReadPath:
    @pytest.fixture
    def store(self, tmp_path: Path) -> LogStore:
        store = LogStore(tmp_path, flush_interval=3600.0)
        for i in range(30):
            store.write_decision(_row(i))
        store.flush()
        return store

    def test_round_trips(self, store: LogStore) -> None:
        try:
            assert store.read(DECISIONS).height == 30
        finally:
            store.close()

    def test_date_bounds_select_partitions(self, store: LogStore) -> None:
        try:
            today = datetime.now(UTC).date()
            assert store.read(DECISIONS, since=today).height == 30
            assert store.read(DECISIONS, since=today - timedelta(days=7)).height == 30
            # A window that closed well before today must return nothing. The
            # bound is a week back rather than a day: read() deliberately widens
            # the partition filter by a day on each side to survive a caller
            # passing a local date against UTC-stamped partitions, so a
            # yesterday bound legitimately still reaches today.
            assert store.read(DECISIONS, until=today - timedelta(days=7)).height == 0
        finally:
            store.close()

    def test_writes_are_date_partitioned(self, store: LogStore) -> None:
        try:
            # Partitions are stamped in UTC, matching the ts column.
            day = datetime.now(UTC).strftime("%Y-%m-%d")
            assert (store.directory / DECISIONS / f"dt={day}").is_dir()
        finally:
            store.close()

    def test_outcomes_join_on_and_late_ones_are_kept(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime

        with LogStore(tmp_path, flush_interval=3600.0) as store:
            for i in range(4):
                store.write_decision(_row(i))
            store.write_outcome(
                {
                    "request_id": "r-000001",
                    "ts": datetime.now(UTC),
                    "name": "csat",
                    "value": 0.8,
                }
            )
            joined = store.load()
            assert joined.height == 4
            row = joined.filter(joined["request_id"] == "r-000001").row(0, named=True)
            assert row["csat"] == 0.8
            # Requests whose outcome has not arrived keep a null; the estimators
            # drop them for that metric rather than imputing a value.
            assert joined["csat"].null_count() == 3

    def test_reading_an_empty_table_is_a_clear_error(self, tmp_path: Path) -> None:
        with (
            LogStore(tmp_path, flush_interval=3600.0) as store,
            pytest.raises(FileNotFoundError, match="no decisions"),
        ):
            store.read(DECISIONS)


class FakeS3:
    """Records put_object calls instead of talking to AWS."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.objects[f"{Bucket}/{Key}"] = Body


class FakeCursor:
    def __init__(self, log: list[tuple[str, object]]) -> None:
        self.log = log

    def execute(self, sql: str, params: object = None) -> None:
        self.log.append((sql.strip().split()[0].upper(), params))

    def executemany(self, sql: str, rows: list[tuple[object, ...]]) -> None:
        self.log.append(("INSERTMANY", len(rows)))

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakeConnection:
    def __init__(self, log: list[tuple[str, object]]) -> None:
        self.log = log
        self.commits = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.log)

    def commit(self) -> None:
        self.commits += 1
        self.log.append(("COMMIT", None))

    def __enter__(self) -> FakeConnection:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class TestAlternativeSinks:
    """S3 and Postgres inherit the durability contract rather than restating it.

    The point of the shared base is that a second copy of the buffering and
    backpressure logic would drift, and a drifted copy loses logs silently. These
    check that each sink actually gets the inherited behaviour.
    """

    def test_s3_writes_partitioned_keys(self) -> None:
        from sortition.store import S3Sink

        fake = FakeS3()
        sink = S3Sink("logs-bucket", prefix="prod", client=fake, flush_interval=3600.0)
        try:
            for i in range(5):
                sink.submit(DECISIONS, _row(i))
            sink.flush()
        finally:
            sink.close()

        assert len(fake.objects) == 1
        key = next(iter(fake.objects))
        # Same layout as the local sink, so one duckdb read path serves both.
        assert key.startswith("logs-bucket/prod/decisions/dt=")
        assert key.endswith(".parquet")

    def test_s3_uploads_a_whole_object(self) -> None:
        # One put per batch: an object appears whole or not at all, so a reader
        # can never see a partial part.
        import polars as pl

        from sortition.store import S3Sink

        fake = FakeS3()
        with S3Sink("b", client=fake, flush_interval=3600.0) as sink:
            for i in range(12):
                sink.submit(DECISIONS, _row(i))
            sink.flush()
        body = next(iter(fake.objects.values()))
        assert pl.read_parquet(body).height == 12

    def test_s3_inherits_backpressure(self) -> None:
        from sortition.store import S3Sink

        sink = S3Sink("b", client=FakeS3(), flush_interval=3600.0, max_buffered=10)
        try:
            for i in range(40):
                sink.submit(DECISIONS, _row(i))
            assert sink.buffered == 10
            assert sink.rows_dropped == 30
        finally:
            sink.close()

    def test_s3_submit_never_raises_on_a_broken_client(self) -> None:
        # A telemetry sink must not take down the request it is observing, even
        # when the far end is down.
        from sortition.store import S3Sink

        class Broken:
            def put_object(self, **kwargs: object) -> None:
                raise RuntimeError("no credentials")

        sink = S3Sink("b", client=Broken(), flush_interval=3600.0)
        try:
            sink.submit(DECISIONS, _row(0))
            sink.flush()
            assert sink.rows_dropped == 1
            assert sink.rows_written == 0
        finally:
            sink.close()

    def test_postgres_creates_schema_once_and_commits(self) -> None:
        from sortition.store import PostgresSink

        log: list[tuple[str, object]] = []
        sink = PostgresSink(
            "postgresql://x",
            connect=lambda dsn: FakeConnection(log),
            flush_interval=3600.0,
        )
        try:
            for i in range(3):
                sink.submit(DECISIONS, _row(i))
            sink.flush()
            for i in range(3):
                sink.submit(DECISIONS, _row(i))
            sink.flush()
        finally:
            sink.close()

        verbs = [entry[0] for entry in log]
        assert verbs.count("INSERTMANY") == 2
        assert verbs.count("COMMIT") == 2
        # DDL is issued once, not on every batch.
        assert verbs.count("CREATE") == 2  # schema + table, first batch only

    def test_postgres_batches_in_one_transaction(self) -> None:
        from sortition.store import PostgresSink

        log: list[tuple[str, object]] = []
        with PostgresSink(
            "postgresql://x",
            connect=lambda dsn: FakeConnection(log),
            flush_interval=3600.0,
        ) as sink:
            for i in range(25):
                sink.submit(DECISIONS, _row(i))
            sink.flush()
        # All 25 rows land in a single executemany, matching the parquet sink's
        # all-or-nothing part file.
        assert ("INSERTMANY", 25) in log
