"""Whether the log can still answer questions, checked continuously.

This exists for one failure mode, and it is the one most likely to go unnoticed:
a router whose log has gone blind. Exploration gets turned down, or a policy
becomes deterministic, and everything continues to look healthy -- requests
succeed, latency is fine, cost is fine -- while the logs quietly stop being able
to justify any of it. Nothing in a normal dashboard shows that.

``HealthReport.ok`` is meant for a readiness probe or an alert, so it answers
"could this log support an estimate?", not "is the router good?".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from sortition import metrics
from sortition.eval.diagnostics import MIN_ESS_FRACTION, effective_sample_size
from sortition.frame import to_arrays
from sortition.targets import Uniform

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)

# A propensity this close to 1 means the policy had no real choice on that row.
DETERMINISTIC_THRESHOLD = 0.999
# Above this share of deterministic rows, the log can only confirm what the
# policy already prefers.
MAX_DETERMINISTIC_RATE = 0.99
MAX_LEAKAGE_RATE = 0.05
MIN_ROWS = 30


@dataclass(frozen=True)
class HealthReport:
    """Whether a recent slice of log can support counterfactual claims."""

    n: int
    deterministic_rate: float
    """Share of rows whose propensity was effectively 1. These rows carry no
    information about any arm other than the one that served them."""

    exploration_rate: float
    leakage_rate: float
    ess_fraction: float
    """Effective sample size against a uniform target, as a fraction of rows.
    A general-purpose measure of how much the log can be re-weighted at all."""

    arm_share: dict[str, float]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """Whether the log is healthy enough to estimate from."""
        return not self.warnings

    def explain(self) -> str:
        """Render the report as human-readable lines.

        Returns:
            A multi-line summary.
        """
        shares = ", ".join(
            f"{arm}={share:.1%}" for arm, share in sorted(self.arm_share.items())
        )
        lines = [
            f"rows={self.n}  exploration={self.exploration_rate:.1%}  "
            f"deterministic={self.deterministic_rate:.1%}",
            f"leakage={self.leakage_rate:.1%}  ESS={self.ess_fraction:.1%} of n",
            f"arm share: {shares}",
        ]
        lines.extend(f"WARNING: {w}" for w in self.warnings)
        lines.append("HEALTHY" if self.ok else "NOT HEALTHY")
        return "\n".join(lines)


def assess(logs: pl.DataFrame) -> HealthReport:
    """Judge whether a slice of log can support estimates.

    Args:
        logs: A log table, typically a recent window.

    Returns:
        The report, with ``ok`` false when something needs attention.
    """
    data = to_arrays(logs, drop_fallbacks=True)
    n = data.n

    deterministic = float((data.propensity >= DETERMINISTIC_THRESHOLD).mean())
    explored = float("nan")
    if "explore" in logs.columns:
        flags = logs.get_column("explore").fill_null(False).cast(bool).to_numpy()
        explored = float(flags.mean()) if flags.size else 0.0
    total = n + data.n_excluded_leakage
    leakage = data.n_excluded_leakage / total if total else 0.0

    counts = np.bincount(data.action, minlength=len(data.arms))
    arm_share = {arm: float(counts[i] / n) for i, arm in enumerate(data.arms)}

    # Uniform is the reference target because it is the most demanding thing a
    # log is routinely asked about, and it needs no policy to be configured.
    probs = Uniform().probabilities(data.features, data.eligible, data.arms)
    weights = probs[np.arange(n), data.action] / data.propensity
    ess_fraction = effective_sample_size(weights) / n if n else 0.0

    warnings: list[str] = []
    if n < MIN_ROWS:
        warnings.append(f"only {n} rows in this window; too few to judge anything")
    if deterministic > MAX_DETERMINISTIC_RATE:
        warnings.append(
            f"{deterministic:.1%} of requests had no real choice (propensity 1.0). "
            "This log can confirm what the policy already prefers and nothing "
            "else. Set an exploration floor."
        )
    if leakage > MAX_LEAKAGE_RATE:
        warnings.append(
            f"{leakage:.1%} of requests were rerouted by the gateway after the "
            "decision, above the 5% threshold; those rows cannot be evaluated"
        )
    if n >= MIN_ROWS and ess_fraction < MIN_ESS_FRACTION:
        warnings.append(
            f"effective sample size is {ess_fraction:.1%} of rows; the log is "
            "too concentrated on one arm to re-weight"
        )

    report = HealthReport(
        n=n,
        deterministic_rate=deterministic,
        exploration_rate=explored,
        leakage_rate=leakage,
        ess_fraction=ess_fraction,
        arm_share=arm_share,
        warnings=tuple(warnings),
    )
    publish(report)
    return report


def publish(report: HealthReport) -> None:
    """Export a report as Prometheus gauges.

    Args:
        report: The report to publish.
    """
    metrics.health_ok.set(1 if report.ok else 0)
    metrics.leakage_rate.set(report.leakage_rate)
    metrics.effective_sample_fraction.set(report.ess_fraction)
    if report.exploration_rate == report.exploration_rate:  # not NaN
        metrics.exploration_rate.set(report.exploration_rate)
