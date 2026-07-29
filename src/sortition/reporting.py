"""The periodic report: what the router did, and whether to believe it.

Distinct from ``sortition.eval.report``, which is the estimator API. This module
assembles those estimates into a document someone forwards.

One rule governs the layout: **the health verdict comes first when it is bad.**
A savings figure computed from a blind log renders exactly like a savings figure
computed from a good one, so a report that leads with the number and buries the
caveat is worse than no report. When the log cannot support the estimates, the
estimates are still shown -- suppressing them would invite someone to recompute
them by hand -- but they are shown underneath a statement that they should not be
acted on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sortition.eval.report import Comparison, compare
from sortition.health import HealthReport, assess

if TYPE_CHECKING:
    import polars as pl

_DURATION = re.compile(r"^(\d+)\s*([dhw])$", re.IGNORECASE)
_UNITS = {"h": "hours", "d": "days", "w": "weeks"}


def parse_since(text: str) -> datetime:
    """Turn ``7d``, ``24h`` or ``2w`` into an absolute cutoff.

    Args:
        text: A duration such as ``7d``.

    Returns:
        The corresponding UTC datetime.

    Raises:
        ValueError: If the duration cannot be parsed.
    """
    match = _DURATION.match(text.strip())
    if not match:
        raise ValueError(f"could not read a duration from {text!r}; try 7d, 24h or 2w")
    amount, unit = int(match.group(1)), match.group(2).lower()
    return datetime.now(UTC) - timedelta(**{_UNITS[unit]: amount})


@dataclass(frozen=True)
class Report:
    """A rendered view of one window of routing traffic."""

    window: str
    generated_at: datetime
    health: HealthReport
    comparisons: dict[str, list[Comparison]]
    """Baseline name to its comparison against the logged policy."""

    policy_versions: tuple[str, ...] = field(default_factory=tuple)
    observed: dict[str, float] = field(default_factory=dict)
    """What actually happened, per metric. Not an estimate."""

    @property
    def trustworthy(self) -> bool:
        """Whether the estimates in this report should be acted on."""
        return self.health.ok


def build(
    logs: pl.DataFrame,
    *,
    baselines: tuple[str, ...] = ("always:premium",),
    metrics: tuple[str, ...] = ("outcome", "cost_usd"),
    window: str = "all time",
    alpha: float = 0.05,
) -> Report:
    """Assemble a report from a window of logs.

    Args:
        logs: The log table.
        baselines: Policies to compare the logged traffic against.
        metrics: Metric columns to report.
        window: Human-readable description of the window, for the header.
        alpha: 1 minus the confidence level.

    Returns:
        The assembled report.
    """
    health = assess(logs)

    observed: dict[str, float] = {}
    for metric in metrics:
        if metric in logs.columns:
            column = logs.get_column(metric).drop_nulls()
            if len(column):
                observed[metric] = float(column.to_numpy().mean())

    # "uniform" stands in for the logged policy: comparing the deployed policy
    # against a baseline needs the deployed policy expressed as a target, which
    # a log alone does not provide. Reporting the observed mean alongside keeps
    # the actual number visible either way.
    comparisons: dict[str, list[Comparison]] = {}
    for baseline in baselines:
        try:
            comparisons[baseline] = compare(
                logs, baseline, "uniform", metrics=metrics, alpha=alpha
            )
        except ValueError:
            continue

    versions: tuple[str, ...] = ()
    if "policy_version" in logs.columns:
        versions = tuple(
            sorted(v for v in logs.get_column("policy_version").unique().to_list() if v)
        )

    return Report(
        window=window,
        generated_at=datetime.now(UTC),
        health=health,
        comparisons=comparisons,
        policy_versions=versions,
        observed=observed,
    )


def to_markdown(report: Report) -> str:
    """Render a report as markdown.

    Args:
        report: The report to render.

    Returns:
        The markdown document.
    """
    lines = [
        "# Routing report",
        "",
        f"Window: {report.window}  ",
        f"Generated: {report.generated_at:%Y-%m-%d %H:%M UTC}  ",
        f"Rows: {report.health.n:,}",
        "",
    ]

    if not report.trustworthy:
        # First, before any number anyone might act on.
        lines += [
            "## These numbers should not be acted on",
            "",
            "The log cannot support counterfactual estimates:",
            "",
        ]
        lines += [f"- {w}" for w in report.health.warnings]
        lines += [
            "",
            "The estimates below are shown so they are not recomputed by hand, "
            "not because they are usable.",
            "",
        ]

    if report.policy_versions:
        lines += [
            "## Policies in this window",
            "",
        ]
        lines += [f"- `{v}`" for v in report.policy_versions]
        if len(report.policy_versions) > 1:
            lines += [
                "",
                "More than one policy served this window. Estimates below pool "
                "them, which is only meaningful if you intended to compare the "
                "mixture rather than either one.",
            ]
        lines.append("")

    if report.observed:
        lines += [
            "## What actually happened",
            "",
            "| metric | observed mean |",
            "|---|---|",
        ]
        lines += [f"| {m} | {v:.6g} |" for m, v in sorted(report.observed.items())]
        lines.append("")

    for baseline, results in report.comparisons.items():
        lines += [
            f"## Against `{baseline}`",
            "",
            "| metric | baseline | alternative | difference "
            "| 95% interval | significant |",
            "|---|---|---|---|---|---|",
        ]
        for r in results:
            low, high = r.difference_interval
            lines.append(
                f"| {r.metric} | {r.a.value:.6g} | {r.b.value:.6g} | "
                f"{r.difference:+.6g} | [{low:+.6g}, {high:+.6g}] | "
                f"{'yes' if r.significant else 'no'} |"
            )
        lines.append("")

    lines += ["## Log health", "", "```", report.health.explain(), "```", ""]
    return "\n".join(lines)


def to_terminal(report: Report) -> None:
    """Print a report to the terminal.

    Args:
        report: The report to render.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    if not report.trustworthy:
        console.print(
            Panel(
                "\n".join(f"- {w}" for w in report.health.warnings),
                title="[bold red]These numbers should not be acted on[/bold red]",
                border_style="red",
            )
        )

    header = Table(title=f"Routing report - {report.window}", show_header=False)
    header.add_row("rows", f"{report.health.n:,}")
    header.add_row("exploration", f"{report.health.exploration_rate:.1%}")
    header.add_row("leakage", f"{report.health.leakage_rate:.1%}")
    header.add_row("policies", ", ".join(report.policy_versions) or "unknown")
    console.print(header)

    if report.observed:
        observed = Table(title="What actually happened")
        observed.add_column("metric")
        observed.add_column("observed mean", justify="right")
        for metric, value in sorted(report.observed.items()):
            observed.add_row(metric, f"{value:.6g}")
        console.print(observed)

    for baseline, results in report.comparisons.items():
        table = Table(title=f"Against {baseline}")
        table.add_column("metric")
        table.add_column("baseline", justify="right")
        table.add_column("alternative", justify="right")
        table.add_column("difference", justify="right")
        table.add_column("95% interval", justify="right")
        for r in results:
            low, high = r.difference_interval
            table.add_row(
                r.metric,
                f"{r.a.value:.6g}",
                f"{r.b.value:.6g}",
                f"[bold]{r.difference:+.6g}[/bold]"
                if r.significant
                else f"{r.difference:+.6g}",
                f"[{low:+.4g}, {high:+.4g}]",
            )
        console.print(table)

    arms = Table(title="Arm share")
    arms.add_column("arm")
    arms.add_column("share", justify="right")
    for arm, share in sorted(report.health.arm_share.items()):
        arms.add_row(arm, f"{share:.1%}")
    console.print(arms)
