# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-08-24

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
- `sortition.train.sweep`: traces the quality/price frontier over a grid of cost
  weights and selects a point by a non-inferiority test against ignoring cost
  entirely, so the exchange rate comes from the log instead of from a constant
  somebody typed. `sortition train --tune-cost-weight` prints the whole frontier
  and splits three ways -- fit, tune, report -- because choosing a weight on the
  rows the result is quoted on is the same error the train/test split prevents,
  one level up. On the simulator, where the true cost of every arm on every
  request is known, the tuned policy bills 39% less than the router that produced
  the logs and scores better on quality; ignoring cost would have billed more
  than the incumbent.
- `tests/test_clean_cases.py`: problems whose optimum is stated in advance --
  identical quality at ten times the price, a dominated arm, a noiseless price
  ladder. Every other test measures against a simulator optimum nobody hits
  exactly, so its assertions are inequalities loose enough to hide a real defect.
  These found two.
- `tests/test_conftest_gating.py`: asserts that a test module importing an
  optional extra is declared in `conftest._NEEDS`. An undeclared module does not
  skip on the minimal install, it fails collection and takes the whole job with
  it, which is a one-line omission with no local symptom.
- `TreePolicy.predict_cost`, and a second Tweedie-loss booster behind it, so what
  a request costs is predicted per request instead of charged at the arm's global
  mean.

### Changed

- **The default `tolerance` for the cost-weight sweep is 0.01, not zero.** A
  margin of exactly zero looks like the safe choice and is instead a rule that
  never fires: proving a difference is no worse than zero needs an interval
  excluding everything negative, and the interval around a true difference of
  zero straddles it at every sample size. Two arms of identical quality where one
  costs ten times more -- the one trade nobody would argue with -- was refused.
  Zero is still accepted and now means "do not trade".
- **A trained policy now prices cost per request, not per arm.** `TreePolicy`
  charged every request the arm's mean cost over the whole training log, so a
  200-token call and a 20k-token call were priced identically and the cost term
  could only ever express the cheap-to-premium ladder. It fits a second booster
  over the same design instead. On the simulator this cuts the error against the
  true expected cost by a factor of 3.7 (mean absolute error 0.0018 against the
  per-arm mean's 0.0067). `cost_usd` is gone from the `kind="tree"` payload and
  `cost_booster_text` replaces it; a `cost_weight` with no cost column to learn
  from is now an error rather than a silently inert setting.
- `sim.BanditProblem.cost` is `(n_contexts, n_arms)`, not `(n_arms,)`. A bill is
  price-per-token times tokens, and a per-arm constant left nothing for a cost
  model to learn.
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

The learned policy reaches about 2.7x the cheapest bill that would buy its own
quality on the contextual simulator. Substituting exact costs for the fitted ones
moves that by 0.2%, so the distance is the quality model rather than the pricing:
it is estimation error on a Bernoulli outcome, not a loss function or a capacity
setting, and squared error against binary logloss, weighted against unweighted,
and four tree sizes all land within a percent of each other. More logs narrow it
to about 1.9x by 240k rows and then it flattens, short of the 1.22x the logged
features permit. Early stopping closes the gap where features carry no signal and
widens it where they do.

Validated against a simulator with known ground truth and a mocked LiteLLM, not
against real production traffic. Sink throughput under sustained load and parquet
part-count growth over weeks are untested.

[Unreleased]: https://github.com/gojiplus/sortition/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/gojiplus/sortition/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/gojiplus/sortition/commits/main
