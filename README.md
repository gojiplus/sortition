# sortition

**Counterfactual evaluation for LLM routing policies.**

[![CI](https://github.com/gojiplus/sortition/actions/workflows/ci.yml/badge.svg)](https://github.com/gojiplus/sortition/actions/workflows/ci.yml)
[![Docs](https://github.com/gojiplus/sortition/actions/workflows/docs.yml/badge.svg)](https://gojiplus.github.io/sortition/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Every LLM router picks a model. None of them can tell you whether it picked well,
because answering that requires the counterfactual: *what would the other model
have cost and scored?* Static benchmarks don't resemble production traffic, and
production logs record only the outcome of the model that was actually called.

Sortition treats routing as policy learning. It records the probability with
which each decision was made, explores on a slice of traffic, and ships the
estimators that turn those logs into valid claims about policies that were never
deployed.

## The loop

```bash
# 1. A candidate policy, compiled to a versioned artifact.
sortition policy build rules.yaml -o candidate.json --epsilon 0.05

# 2. What would it have done to last month's traffic?
sortition compare logs.parquet --a policy:current.json --b policy:candidate.json

# 3. If it wins, deploy it. The engine hot-reloads; no restart.
cp candidate.json /etc/sortition/policy.json

# 4. Watch it. Exits non-zero when the log stops being able to justify itself.
sortition doctor logs.parquet --check

# 5. Report on it.
sortition report logs.parquet --baseline policy:current.json --out report.md
sortition dashboard logs.parquet --out dashboard.html
```

Or skip writing the rules and learn them:

```bash
sortition train logs.parquet -o learned.json --holdout 0.3
# trained learned-bf2d5e3d9577 on 20712 rows
# on 8842 held-out rows:
#   what actually happened: 0.465
#   this policy would have: 0.711  [0.677, 0.744]
```

The holdout is not optional politeness: a policy chosen to look good on a set of
logs will look good on those logs.

Step 2 is the one no other open-source router can do.

## Try it in thirty seconds

```bash
pip install 'sortition[cli]'
sortition demo --out logs.parquet          # synthetic log, ground truth printed
sortition doctor logs.parquet              # can this log answer anything?
sortition eval logs.parquet --target always:arm-3 --metric cost_usd
```

`demo` prints the true value of every policy, so you can check the estimates
against answers a real log would never give you.

## From Python

```python
import polars as pl
from sortition.eval import compare, evaluate

logs = pl.read_parquet("logs.parquet")

# What would always-premium have cost?
evaluate(logs, "always:arm-3", metric="cost_usd")

# Two real policies, head to head, with intervals on the difference.
compare(logs, a="policy:current.json", b="policy:candidate.json")
```

## Producing the logs

The estimators work on any log carrying a propensity. To generate one, sortition
ships an in-process LiteLLM routing plugin:

```python
import litellm
from sortition.decide import ReloadingEngine
from sortition.integrations import SortitionLogger, SortitionPlugin
from sortition.store import LogStore

store = LogStore("routing-logs")
litellm.callbacks = [SortitionLogger(store)]

router = litellm.Router(
    model_list=[...],
    plugins=[SortitionPlugin(ReloadingEngine("policy.json"))],
)
```

The plugin holds no credentials and executes no calls. It samples one arm,
narrows LiteLLM's candidate pool to it, and records the probability. If it
raises, it fails open and the request is served normally — and increments
`sortition_plugin_errors_total`, because a silently broken router is
indistinguishable from a working one otherwise.

## Why propensities

Randomization without recorded propensities is wasted entropy. A router that
samples across models perturbs production traffic and buys no inferential value
unless it records the probability with which it made each choice. That number
cannot be reconstructed afterwards, and it is what makes a log answerable.

Two consequences run through the design:

- **A policy with no exploration produces logs that can only confirm what it
  already does.** `doctor` says so, loudly, rather than returning a
  confident-looking number.
- **A gateway fallback breaks the propensity.** When LiteLLM's own retry
  machinery overrides a decision, the served arm was not drawn from the sampler,
  so those rows are excluded and reported as a leakage rate rather than quietly
  averaged in.

## What's in the box

| module | what it does |
|---|---|
| `sortition.eval` | IPS, SNIPS, DM, DR, switch-DR, shrinkage-DR; cross-fitted outcome models; Waudby-Smith–Ramdas betting intervals for bounded outcomes, bootstrap for cost and latency; overlap and support diagnostics that refuse rather than guess |
| `sortition.decide` | rules policies, epsilon-greedy with exact propensities, Thompson with Monte Carlo ones, versioned artifacts, hot reload |
| `sortition.train` | a boosted-tree policy fitted from logs, deployable as the same artifact a rules table produces |
| `sortition.integrations` | LiteLLM routing plugin and logging callback |
| `sortition.store` | durable, non-blocking, date-partitioned parquet; duckdb read path; S3 and Postgres sinks |
| `sortition.health` | is this log still able to justify the router? |
| `sortition.metrics` | Prometheus, or nothing at all if it isn't installed |
| `sortition.sim` | synthetic bandits with exactly known policy values — the oracle the estimators are tested against |

## Install

```bash
pip install 'sortition[eval]'          # estimators only; no gateway dependency
pip install 'sortition[eval,litellm]'  # + the routing plugin and log adapter
pip install 'sortition[train]'          # + the learned policy
pip install 'sortition[s3]'             # + object-storage sink
pip install 'sortition[all]'
```

The eval path deliberately installs without LiteLLM or LightGBM — it has to run
on a laptop against a parquet file. CI proves it by building the wheel and
importing it in a clean environment.

## Status and limits

Early. The log schema is the stable contract; everything else may change.

Worth saying plainly rather than letting a version number imply otherwise: this
has been validated against a simulator with known ground truth and a mocked
LiteLLM, **not against real production traffic**. The first things likely to
surface in a real deployment are sink throughput under sustained load and
parquet part-count growth over weeks, and no test suite will reveal either.

The durability contract is bounded loss: a hard kill costs at most one flush
interval (5s by default), the same guarantee the Datadog and OpenTelemetry agents
give. LiteLLM's `CustomLogger` exposes no shutdown hook, so the periodic flush is
what bounds it.

## Documentation

<https://gojiplus.github.io/sortition/>

## License

MIT
