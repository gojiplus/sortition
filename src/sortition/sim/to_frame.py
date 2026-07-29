"""Turn simulated logs into the canonical flat table.

Lets the DataFrame path, the CLI, and the docs all exercise the same code a real
gateway feeds, without needing a gateway.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from sortition.sim.generator import BanditProblem, LoggedData

if TYPE_CHECKING:
    import polars as pl


def to_frame(
    logs: LoggedData,
    problem: BanditProblem,
    *,
    policy_version: str = "sim-v1",
    tenant: str = "sim",
    start: datetime | None = None,
    fallback_rate: float = 0.0,
    seed: int = 0,
) -> pl.DataFrame:
    """Render simulated logs as a log table.

    ``fallback_rate`` injects rows where the gateway served an arm other than
    the one drawn -- the leakage the estimators must exclude. Real logs always
    have some; a test fixture with none would never exercise that path.
    """
    import polars as pl

    rng = np.random.default_rng(seed)
    n = logs.n
    # Ending at "now" rather than a fixed past date, so that a window like
    # --since 7d selects something on a freshly generated demo log.
    base = start or (datetime.now(UTC) - timedelta(seconds=n))
    arms = np.array(problem.arms)

    chosen = arms[logs.action]
    served = chosen.copy()
    fallback_depth = np.zeros(n, dtype=np.int64)
    if fallback_rate > 0.0:
        hit = rng.random(n) < fallback_rate
        fallback_depth[hit] = 1
        # The gateway retried onto some other arm, not the one the sampler drew.
        shifted = (logs.action[hit] + 1) % problem.n_arms
        served[hit] = arms[shifted]

    eligible_sets = [
        [problem.arms[a] for a in np.flatnonzero(row)] for row in logs.eligible
    ]
    features = [
        {
            "n_tokens": float(x[0] * 100.0 + 300.0),
            "code_fraction": float(1.0 / (1.0 + np.exp(-x[1]))),
            "context_tokens": float(abs(x[2]) * 1000.0),
            "tools_required": bool(x[3] > 0.0),
        }
        for x in logs.contexts
    ]

    return pl.DataFrame(
        {
            "request_id": [f"r-{i:08d}" for i in range(n)],
            "ts": [base + timedelta(seconds=int(i)) for i in range(n)],
            "tenant": [tenant] * n,
            "session_id": [None] * n,
            "policy_version": [policy_version] * n,
            "chosen_arm": chosen.tolist(),
            "propensity": logs.propensity.tolist(),
            "explore": (logs.propensity < 0.9).tolist(),
            "eligible_set": eligible_sets,
            "features": features,
            "served_arm": served.tolist(),
            "fallback_depth": fallback_depth.tolist(),
            "status": ["success"] * n,
            "outcome": logs.reward.tolist(),
            "cost_usd": logs.cost.tolist(),
            "latency_ms": (500.0 + logs.cost * 20_000.0).tolist(),
            "tokens_in": [int(f["n_tokens"]) for f in features],
            "tokens_out": [int(f["n_tokens"] * 0.3) for f in features],
        }
    )
