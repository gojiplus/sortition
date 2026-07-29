# sortition

**Counterfactual evaluation for LLM routing policies.**

Every LLM router picks a model. None of them can tell you whether it picked well — because
answering that requires the counterfactual: *what would the other model have cost and
scored?* Static benchmarks (MT-Bench, MMLU) don't resemble production traffic, and
production logs record only the outcome of the model that was actually called.

Sortition treats routing as policy learning. It logs propensities, explores on a small
traffic slice, and ships the estimators that turn those logs into valid claims about
policies that were never deployed — including policies it didn't make.

```python
import polars as pl
from sortition.eval import evaluate, compare

logs = pl.read_parquet("routing_logs/*.parquet")

# What would "always premium" have cost, versus what we actually ran?
evaluate(logs, target="always:premium-reasoning", metric="cost_usd")

# Head-to-head on last month's traffic.
compare(logs, a="rules-v3", b="tree-2026-09-04", metrics=["outcome", "cost_usd"])
```

## Why this exists

Randomization without propensities is wasted entropy. A router that samples across models
— as several now do — perturbs production traffic and buys no inferential value unless it
records the probability with which it made each choice. That one number is what makes a
log answerable.

Sortition is two things, and they are separable:

- **`sortition.eval`** — the estimators, diagnostics, and confidence intervals. Works on
  any log that carries a propensity, from any gateway. This is the part you can use
  without changing your router.
- **`sortition.decide`** — a reference policy that produces such logs, shipped as an
  in-process LiteLLM `RoutingPlugin`. Sub-millisecond, holds no credentials, executes no
  calls.

## Install

```bash
pip install 'sortition[eval]'          # estimators only; no gateway dependency
pip install 'sortition[eval,litellm]'  # + the routing plugin and log adapter
pip install 'sortition[all]'
```

The eval path deliberately installs without LiteLLM, FastAPI, or LightGBM — it has to run
on a laptop against a parquet file.

## Status

Early development. The log schema is the stable contract; everything else may change.

## License

MIT
