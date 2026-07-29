"""The flat log table, and how it becomes arrays an estimator can use.

The pydantic models in ``schema`` are the contract for writing a log. This module
is the contract for reading one: a single flat table, one row per resolved
request, which any gateway can produce and which parquet stores natively.

Turning that table into estimator inputs is where two decisions get enforced.
Rows the gateway overrode via its own fallback machinery are dropped and counted,
because the arm that served them was not drawn from the sampler. And the arm
universe is derived from the union of every logged eligible set, so that an arm
index means the same thing on every row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import polars as pl

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

REQUIRED_COLUMNS = ("request_id", "chosen_arm", "propensity", "eligible_set")
"""Without these there is no decision to evaluate."""

OPTIONAL_COLUMNS = (
    "ts",
    "tenant",
    "session_id",
    "policy_version",
    "explore",
    "features",
    "served_arm",
    "fallback_depth",
    "status",
    "outcome",
    "cost_usd",
    "latency_ms",
    "tokens_in",
    "tokens_out",
)


@dataclass(frozen=True)
class EvalArrays:
    """A log table, reduced to what the estimators actually consume."""

    arms: tuple[str, ...]
    action: IntArray
    propensity: FloatArray
    eligible: BoolArray
    features: list[dict[str, Any]]
    contexts: FloatArray
    """Numeric feature matrix for the outcome model. Zero-width when the log
    carries no numeric features, which reduces DR to plain IPS rather than
    failing -- a log without features can still answer the cost question."""

    metrics: dict[str, FloatArray]
    n_excluded_leakage: int
    n_excluded_missing: int

    @property
    def n(self) -> int:
        """Number of evaluable rows."""
        return int(self.action.shape[0])


def _check_columns(df: pl.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"log is missing required column(s) {missing}. A log without "
            "propensities cannot support counterfactual evaluation -- see the "
            "schema docs for what a gateway must emit."
        )


def arm_universe(df: pl.DataFrame) -> tuple[str, ...]:
    """Every arm that appears in any eligible set, in a stable order.

    Taken from ``eligible_set`` rather than ``chosen_arm``: an arm that was
    always available but never chosen still needs an index, or a target policy
    could not ask about it.
    """
    arms: set[str] = set()
    for row in df.get_column("eligible_set").to_list():
        arms.update(row or [])
    arms.update(x for x in df.get_column("chosen_arm").to_list() if x is not None)
    if not arms:
        raise ValueError("no arms found in the log")
    return tuple(sorted(arms))


def _numeric_contexts(features: list[dict[str, Any]]) -> FloatArray:
    """Stack whatever numeric features every row shares.

    Keys present on some rows but not others are skipped rather than imputed:
    a silently zero-filled feature is worse than a missing one, because the
    outcome model treats it as a real measurement.
    """
    if not features:
        return np.zeros((0, 0), dtype=np.float64)

    shared: set[str] | None = None
    for row in features:
        # bool is deliberately included: tool-required flags are real features.
        numeric = {
            k for k, v in (row or {}).items() if isinstance(v, bool | int | float)
        }
        shared = numeric if shared is None else (shared & numeric)
    keys = sorted(shared or ())
    if not keys:
        return np.zeros((len(features), 0), dtype=np.float64)
    return np.array(
        [[float((row or {}).get(k, 0.0)) for k in keys] for row in features],
        dtype=np.float64,
    )


def to_arrays(
    df: pl.DataFrame,
    *,
    metrics: tuple[str, ...] = ("outcome", "cost_usd", "latency_ms"),
    drop_fallbacks: bool = True,
) -> EvalArrays:
    """Reduce a log table to estimator inputs.

    ``drop_fallbacks`` removes rows the gateway rerouted after the decision was
    made. Keeping them would attribute an outcome to a draw that never happened.
    The count survives as ``n_excluded_leakage`` and is reported as a leakage
    rate rather than silently absorbed.
    """
    import polars as pl

    _check_columns(df)
    total = df.height
    if total == 0:
        raise ValueError("log is empty")

    n_leakage = 0
    if drop_fallbacks and "fallback_depth" in df.columns:
        depth = df.get_column("fallback_depth").fill_null(0)
        keep = depth == 0
        n_leakage = int((~keep).sum())
        df = df.filter(keep)
    if "status" in df.columns:
        ok = df.get_column("status").is_null() | (df.get_column("status") == "success")
        df = df.filter(ok)

    before_missing = df.height
    df = df.filter(
        pl.col("propensity").is_not_null() & pl.col("chosen_arm").is_not_null()
    )
    n_missing = before_missing - df.height
    if df.height == 0:
        raise ValueError(
            f"no evaluable rows remain out of {total}: "
            f"{n_leakage} lost to gateway fallback, {n_missing} missing a "
            "propensity or chosen arm"
        )

    arms = arm_universe(df)
    index = {arm: i for i, arm in enumerate(arms)}

    chosen = df.get_column("chosen_arm").to_list()
    action = np.array([index[a] for a in chosen], dtype=np.int64)
    propensity = df.get_column("propensity").to_numpy().astype(np.float64)

    eligible = np.zeros((df.height, len(arms)), dtype=bool)
    for i, row in enumerate(df.get_column("eligible_set").to_list()):
        for arm in row or ():
            eligible[i, index[arm]] = True
    # A log that omitted the eligible set is treated as "everything was allowed",
    # which is the only assumption that does not invent a constraint.
    empty = ~eligible.any(axis=1)
    eligible[empty] = True
    eligible[np.arange(df.height), action] = True

    features = (
        [r or {} for r in df.get_column("features").to_list()]
        if "features" in df.columns
        else [{} for _ in range(df.height)]
    )

    collected: dict[str, FloatArray] = {}
    for name in metrics:
        if name in df.columns:
            values = df.get_column(name)
            if values.null_count() < df.height:
                collected[name] = values.fill_null(np.nan).to_numpy().astype(np.float64)

    return EvalArrays(
        arms=arms,
        action=action,
        propensity=propensity,
        eligible=eligible,
        features=features,
        contexts=_numeric_contexts(features),
        metrics=collected,
        n_excluded_leakage=n_leakage,
        n_excluded_missing=n_missing,
    )
