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

## What a request is worth

A policy that ignores price sends everything to the best model. One that prices
it needs an exchange rate between quality and dollars -- and that number is not
in the log. What *is* in the log is what each rate would have bought, so
`--tune-cost-weight` measures the whole curve and you say how much you will
spend:

```bash
sortition train logs.parquet -o learned.json --tune-cost-weight --tolerance 0.05

# cost-weight frontier, on 7759 tuning rows:
#    weight    outcome   cost_usd/req            vs ignoring cost
#         0     0.7776        0.02661  +0.0000 [+0.0000, +0.0000]
# *   0.125     0.7825        0.02163  +0.0049 [-0.0114, +0.0225]
#      0.25     0.7490        0.01511  -0.0286 [-0.0513, -0.0064]
#       0.5     0.6764       0.008416  -0.1012 [-0.1309, -0.0720]
#         4     0.5918       0.003227  -0.1858 [-0.2241, -0.1501]
#
# chose cost_weight=0.125: saves 0.00498 of cost_usd per request, and is +0.0049 on outcome.
```

Nineteen percent off the bill, provably inside the 0.05 of outcome you said you
would spend, and a cliff immediately after it. The whole table is printed because
the shape is the finding: an operator who sees the drop between 0.125 and 0.5
knows something a single chosen number would have hidden.

The test is one-sided on purpose. `--tolerance 0.05` asks for the cheapest weight
the log can *prove* stays within 0.05 of ignoring cost, not the cheapest one that
is not provably worse -- which passes everything when the log is too small to
measure anything, and hands back the most aggressive weight on the grid exactly
when there is least reason to trust it. At the default of 0.01 this same log
declines to trade at all, because the interval at 0.125 reaches -0.0114.

A margin of exactly zero is not the safe choice it looks like. Proving a
difference is no worse than zero takes an interval that excludes everything
negative, and the interval around a true difference of zero straddles it at any
sample size -- so `--tolerance 0` refuses every trade, including two arms of
identical quality where one costs ten times more. It is available and it means
"do not trade"; the default is a point of outcome instead.

There are three splits here, not two. The weight is chosen on rows the boosters
never saw, and reported on rows neither the boosters nor the sweep saw. Picking
a point from seven and then quoting that point's own number is the train/test
error one level up.

Cost is predicted per request, not per arm. A bill is price-per-token times
tokens, so the premium model is nearly free on a short request and ruinous on a
long one; charging every request the arm's average gets that backwards on both
ends. The cost model is a Tweedie-loss booster over the same features -- spend is
non-negative, right-skewed, and has a point mass at exactly zero when a call is
served from cache.

### Does it actually save money

On a real log this is unanswerable, which is what the simulator is for: it knows
the true quality and the true price of every arm on every request, so a claim
about dollars can be checked rather than estimated. Sixty thousand logged
requests, five arms, a fifty-fold price ladder:

| policy | true quality | $ per 1M requests |
|---|---|---|
| the incumbent router | 0.5878 | 18,506 |
| always the cheapest arm | 0.4440 | 1,156 |
| always the dearest arm | 0.5441 | 57,793 |
| learned, cost ignored | 0.7217 | 23,955 |
| learned, tuned at `--tolerance 0.05` | 0.7097 | **11,348** |

Thirty-nine percent cheaper than the router that produced the logs, and better on
quality, which is the combination worth having. Ignoring cost would have been
*more* expensive than the incumbent.

Optimality is the harder question, and the same ground truth answers it. Sweeping
`argmax_a [q(x,a) - lambda*cost(x,a)]` over lambda traces the best curve any
router could achieve, so the tuned policy can be priced against it at its own
quality:

| what the policy ranks by | $ per 1M | vs the best possible |
|---|---|---|
| true quality, true cost | 4,324 | 1.00x |
| true quality, **learned** cost | 4,331 | 1.00x |
| **learned** quality, true cost | 11,753 | 2.72x |
| best any model could do from the four logged features | 5,254 | 1.22x |

Replacing the learned prices with perfect ones moves the bill by 0.2%, so the
cost model is worth what knowing the costs exactly is worth. The gap that remains
is the quality model, and it is not a loss function or a capacity problem --
squared error against binary logloss, weighted against unweighted, and four sizes
of tree all land within a percent of each other. It is estimation error on a
Bernoulli outcome: choosing the wrong arm is nearly free when arms cost the same
and expensive against a fifty-fold ladder, which is why a 53% top-one rate costs
2.7x.

More logs help, up to a point. Repeating the whole exercise at five sizes takes
the gap from 4.7x at 30k rows to about 1.9x at 240k, where it flattens rather
than continuing toward the 1.22x that the logged features permit. So a thin log
is the first thing to fix and not the last; what is left after that is worth
understanding before anyone promises it away. These are single runs at one seed,
so read the trend and not the third digit.

Exploration is not free either. The same frontier without the 5% floor reaches
that quality for $3,032 per 1M rather than $4,324, so keeping the log answerable
costs about 43% of the bill here. That is the price of being able to run any of
this next quarter, and it is a number rather than a principle.

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
| `sortition.train` | a boosted-tree policy fitted from logs, with a second model for what each request costs, and a sweep that picks the quality/price exchange rate instead of asking you for it |
| `sortition.integrations` | LiteLLM routing plugin and logging callback |
| `sortition.store` | durable, non-blocking, date-partitioned parquet; duckdb read path; S3 and Postgres sinks |
| `sortition.health` | is this log still able to justify the router? |
| `sortition.metrics` | Prometheus, or nothing at all if it isn't installed |
| `sortition.sim` | synthetic bandits with exactly known policy values — the oracle the estimators are tested against |

## Install

```bash
pip install 'sortition[eval]'          # estimators only; no gateway dependency
pip install 'sortition[cli]'           # + the sortition command
pip install 'sortition[eval,litellm]'  # + the routing plugin and log adapter
pip install 'sortition[train]'         # + the learned policy and the cost sweep
pip install 'sortition[s3]'            # + object-storage sink
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
