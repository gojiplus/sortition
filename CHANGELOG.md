# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `S3Sink` and `PostgresSink`, behind the `s3` and `postgres` extras. Both
  inherit the durability contract from a shared `BufferedSink` rather than
  restating it.
- `sortition.train`: a boosted-tree policy fitted from logs, emitted as a
  `kind="tree"` artifact so it deploys and is evaluated exactly like a rules
  table. `sortition train` reports the held-out comparison and says when the
  candidate is not measurably better.
- `sortition.features`: the feature vectoriser, shared by training and the
  decision path so a deployed policy scores as it was fitted.
- `sortition dashboard`: a self-contained HTML page, no server and no CDN.

### Changed

- **Every coverage gate's tolerance now comes from the replicate count.** The
  five statistical assertions in `tests/test_ci.py` and `tests/test_estimators.py`
  were one-sided floors written by hand -- `coverage >= 0.95`, `>= 0.88`,
  `>= 0.90` -- and an interval that is vacuously wide passes every one. They come
  from [simcheck](https://github.com/finite-sample/simcheck) now: a two-sided
  `assert_count_rate` where the method claims coverage equal to nominal, and
  `binomial_band`'s lower edge where it claims coverage *at least* nominal, which
  is what `betting_ci` guarantees as a confidence sequence.
- `assert_intervals_informative` gates the betting interval's *width* against the
  spread the study measured, which is the thing a one-sided floor cannot see.
- Replicate counts come from `simcheck.reps_for` and the `SIMCHECK_DEEP` /
  `SIMCHECK_REPS` variables, replacing the repo-local `SORTITION_FULL_SIMS`, so
  the replicate count and the bands derived from it move together.

## [0.1.0] - 2026-07-29

First release. The loop is closed: a candidate policy can be evaluated on
historical logs, deployed as a versioned artifact without a restart, watched, and
reported on.

### Evaluation

- IPS, SNIPS, direct method, doubly robust, switch-DR and DR with optimistic
  shrinkage, over logged bandit feedback.
- Cross-fitted outcome models. The folds are walked explicitly rather than via
  `cross_val_predict`, which returns predictions only at observed rows while DR
  needs them at counterfactual arms.
- Waudby-Smith–Ramdas hedged betting intervals for bounded outcomes, valid under
  repeated peeking; bootstrap for cost and latency. Implemented here because
  `confseq` last shipped in January 2023 with wheels up to cp310.
- Overlap, effective-sample-size, support-violation and leakage diagnostics that
  mark an estimate untrustworthy rather than returning a number the data cannot
  support.
- `evaluate`, `compare` and `doctor`, plus a CLI over parquet, CSV and ndjson.
  Differences between policies and their intervals come from the same per-row
  scores, so a reported difference cannot fall outside its own interval.

### Deciding

- Rules policies from a YAML table, distinguishing hard constraints that shrink
  the eligible set from soft preferences that remain explorable.
- Epsilon-greedy with exact analytic propensities; Thompson sampling with Monte
  Carlo ones, written in pure Python so it is portable to gateways that do not
  depend on numpy.
- Versioned policy artifacts with a content-hashed `policy_version`, and hot
  reload. A corrupt artifact leaves the running policy serving.
- `PolicyTarget` bridges the two halves: a deployed policy can be evaluated as
  a target, which is what makes "would this candidate have done better?"
  answerable. The deployed and evaluated probabilities come from one shared
  implementation, asserted by a round trip returning importance weights of
  exactly 1.

### Operating

- LiteLLM routing plugin and logging callback. The plugin narrows the candidate
  pool to the single sampled arm, fails open on error, and counts that it did.
- Durable, non-blocking logging: rows are buffered and written from a dedicated
  thread, bounding loss to one flush interval on a hard kill.
- Date-partitioned parquet with a duckdb read path, so a query for last week
  does not read last year.
- Prometheus metrics, degrading to no-ops when `prometheus_client` is absent.
- `sortition.health`, which catches a log that has gone blind — the failure that
  leaves every operational signal green while the router quietly loses the
  ability to justify itself.
- `sortition report` for a document someone forwards, leading with the health
  verdict when the log cannot support the numbers in it.

### Known limits

Validated against a simulator with known ground truth and a mocked LiteLLM, not
against real production traffic. Sink throughput under sustained load and parquet
part-count growth over weeks are untested.

[Unreleased]: https://github.com/gojiplus/sortition/compare/main...main
[0.1.0]: https://github.com/gojiplus/sortition/commits/main
