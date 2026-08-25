"""Command line interface.

    sortition doctor  logs.parquet
    sortition eval    logs.parquet --target always:premium --metric cost_usd
    sortition compare logs.parquet --a uniform --b always:premium
    sortition report  logs.parquet --baseline always:premium --out report.md
    sortition policy  build rules.yaml -o policy.json
    sortition demo    --out logs.parquet

``doctor`` comes first on purpose. Whether a log can support a counterfactual
claim is prior to what the claim is, and it is the question people skip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

app = typer.Typer(
    name="sortition",
    help="Counterfactual evaluation for LLM routing policies.",
    no_args_is_help=True,
    add_completion=False,
)
policy_app = typer.Typer(
    name="policy",
    help="Build and inspect versioned routing policies.",
    no_args_is_help=True,
)
app.add_typer(policy_app)


def _load(path: Path) -> object:
    import polars as pl

    if not path.exists():
        raise typer.BadParameter(f"no such file: {path}")
    if path.suffix in (".parquet", ".pq"):
        return pl.read_parquet(path)
    if path.suffix in (".csv", ".tsv"):
        return pl.read_csv(path, separator="\t" if path.suffix == ".tsv" else ",")
    if path.suffix in (".ndjson", ".jsonl"):
        return pl.read_ndjson(path)
    raise typer.BadParameter(
        f"unsupported format {path.suffix!r}; use parquet, csv, or ndjson"
    )


@app.command()
def doctor(
    log: Annotated[Path, typer.Argument(help="Log file to inspect.")],
    target: Annotated[
        str, typer.Option(help="Target policy to assess overlap against.")
    ] = "uniform",
    check: Annotated[
        bool,
        typer.Option(
            help="Exit non-zero when the log cannot support estimates. For alerts."
        ),
    ] = False,
) -> None:
    """Report whether a log can support counterfactual claims."""
    from sortition.eval.report import doctor as run_doctor
    from sortition.health import assess

    frame = _load(log)
    typer.echo(run_doctor(frame, target))  # type: ignore[arg-type]

    report = assess(frame)  # type: ignore[arg-type]
    typer.echo("")
    typer.echo(report.explain())
    if check and not report.ok:
        raise typer.Exit(code=1)


@app.command()
def eval(
    log: Annotated[Path, typer.Argument(help="Log file to evaluate.")],
    target: Annotated[
        str, typer.Option(help="e.g. always:premium-reasoning, uniform.")
    ],
    metric: Annotated[str, typer.Option(help="Column to evaluate.")] = "outcome",
    estimator: Annotated[
        str, typer.Option(help="ips, snips, dm, dr, switch_dr, dr_os.")
    ] = "dr",
    alpha: Annotated[float, typer.Option(help="1 - confidence level.")] = 0.05,
    anytime: Annotated[
        bool, typer.Option(help="Interval valid under repeated peeking.")
    ] = False,
) -> None:
    """Estimate what a metric would have been under a target policy."""
    from sortition.eval.report import evaluate

    result = evaluate(
        _load(log),  # type: ignore[arg-type]
        target,
        metric=metric,
        estimator=estimator,  # type: ignore[arg-type]
        alpha=alpha,
        anytime=anytime,
    )
    typer.echo(str(result))
    if not result.trustworthy:
        raise typer.Exit(code=1)


@app.command()
def compare(
    log: Annotated[Path, typer.Argument(help="Log file to compare over.")],
    a: Annotated[str, typer.Option(help="Baseline policy.")],
    b: Annotated[str, typer.Option(help="Candidate policy.")],
    metrics: Annotated[str, typer.Option(help="Comma-separated metric columns.")] = (
        "outcome,cost_usd"
    ),
    estimator: Annotated[str, typer.Option()] = "dr",
    alpha: Annotated[float, typer.Option()] = 0.05,
) -> None:
    """Compare two policies head to head on the same logged traffic."""
    from sortition.eval.report import compare as run_compare

    results = run_compare(
        _load(log),  # type: ignore[arg-type]
        a,
        b,
        metrics=tuple(m.strip() for m in metrics.split(",") if m.strip()),
        estimator=estimator,  # type: ignore[arg-type]
        alpha=alpha,
    )
    for result in results:
        typer.echo(str(result))
        typer.echo("")
    if any(not r.trustworthy for r in results):
        raise typer.Exit(code=1)


@app.command()
def demo(
    out: Annotated[Path, typer.Option(help="Where to write the generated log.")] = Path(
        "sortition-demo.parquet"
    ),
    n: Annotated[int, typer.Option(help="Number of requests.")] = 20_000,
    epsilon: Annotated[
        float, typer.Option(help="Exploration rate of the logging policy.")
    ] = 0.2,
    fallback_rate: Annotated[
        float, typer.Option(help="Share of gateway fallbacks.")
    ] = 0.02,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Write a synthetic but realistic log, for trying the other commands."""
    import numpy as np

    from sortition.sim import epsilon_greedy_policy, make_problem, sample_logs
    from sortition.sim.to_frame import to_frame

    problem = make_problem(n_contexts=500, n_arms=4, seed=seed)
    weights = np.random.default_rng(seed).standard_normal((6, problem.n_arms))
    policy = epsilon_greedy_policy(weights, epsilon=epsilon)
    logs = sample_logs(problem, policy, n, seed=seed + 1)
    frame = to_frame(
        logs,
        problem,
        policy_version=f"sim-eps{epsilon:g}",
        fallback_rate=fallback_rate,
        seed=seed + 2,
    )
    frame.write_parquet(out)

    typer.echo(f"wrote {frame.height} rows to {out}")
    typer.echo(f"arms: {list(problem.arms)}")
    typer.echo("\nground truth (unknowable from a real log, exact here):")
    for arm_index, arm in enumerate(problem.arms):
        from sortition.sim import constant_policy

        constant = constant_policy(arm_index)
        typer.echo(
            f"  always:{arm:<8} outcome={problem.value(constant):.4f} "
            f"cost={problem.cost_value(constant):.5f}"
        )
    typer.echo(f"  {'logged policy':<15} outcome={problem.value(policy):.4f} ")
    typer.echo(f"\ntry:  sortition doctor {out}")


@policy_app.command("build")
def policy_build(
    rules: Annotated[Path, typer.Argument(help="YAML rule table.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Artifact to write.")] = Path(
        "policy.json"
    ),
    epsilon: Annotated[
        float, typer.Option(help="Share of traffic kept exploring.")
    ] = 0.05,
    name: Annotated[
        str | None, typer.Option(help="Label prefixed to the content hash.")
    ] = None,
) -> None:
    """Compile a rule table into a versioned, deployable policy artifact."""
    from sortition.decide import ExplorationConfig, RulesPolicy, build, save

    if not rules.exists():
        raise typer.BadParameter(f"no such file: {rules}")

    artifact = build(
        RulesPolicy.from_yaml(rules),
        ExplorationConfig(epsilon=epsilon),
        name=name,
        git_sha=_git_sha(),
    )
    save(artifact, out)
    typer.echo(f"wrote {artifact.policy_version} to {out}")
    if epsilon < 0.01:
        typer.echo(
            "WARNING: this policy explores less than 1% of traffic. Its logs "
            "will confirm what it already prefers and nothing else."
        )


@policy_app.command("show")
def policy_show(
    artifact: Annotated[Path, typer.Argument(help="Policy artifact to inspect.")],
) -> None:
    """Print what a deployed policy actually does."""
    from sortition.decide.artifact import load

    if not artifact.exists():
        raise typer.BadParameter(f"no such file: {artifact}")

    policy, exploration, meta = load(artifact)
    typer.echo(f"version:     {meta.policy_version}")
    typer.echo(f"kind:        {meta.kind}")
    typer.echo(f"created:     {meta.created_at:%Y-%m-%d %H:%M:%S %Z}")
    typer.echo(f"arms:        {', '.join(meta.arms)}")
    typer.echo(f"exploration: {exploration.strategy} eps={exploration.epsilon:g}")
    if meta.git_sha:
        typer.echo(f"git:         {meta.git_sha}")

    rules = getattr(policy, "rules", ())
    if rules:
        typer.echo("\nrules, first match wins:")
        for rule in rules:
            label = rule.label or "(unlabelled)"
            action = (
                f"exclude {', '.join(rule.exclude)}"
                if rule.exclude
                else f"prefer {', '.join(rule.prefer)}"
            )
            typer.echo(f"  {label}: when {rule.when or '(always)'} -> {action}")
    default = getattr(policy, "default", ())
    if default:
        typer.echo(f"\ndefault order: {', '.join(default)}")


def _git_sha() -> str | None:
    """Return the current commit, when the working directory is a repo.

    Returns:
        The short SHA, or None outside a repository.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


@app.command()
def report(
    log: Annotated[Path, typer.Argument(help="Log file to report on.")],
    baseline: Annotated[
        list[str] | None,
        typer.Option(help="Policy to compare against. Repeatable."),
    ] = None,
    metrics: Annotated[str, typer.Option(help="Comma-separated metric columns.")] = (
        "outcome,cost_usd"
    ),
    since: Annotated[str | None, typer.Option(help="Window, e.g. 7d, 24h, 2w.")] = None,
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="Write markdown here.")
    ] = None,
    check: Annotated[
        bool, typer.Option(help="Exit non-zero when the log cannot support estimates.")
    ] = False,
) -> None:
    """Summarise a window of routing traffic, for forwarding."""
    import polars as pl

    from sortition.reporting import build, parse_since, to_markdown, to_terminal

    frame = _load(log)
    window = "all time"
    if since:
        cutoff = parse_since(since)
        window = f"last {since}"
        if "ts" in frame.columns:  # type: ignore[union-attr]
            frame = frame.filter(pl.col("ts") >= cutoff)  # type: ignore[union-attr]
            if frame.height == 0:  # type: ignore[union-attr]
                # A quiet window is a normal answer, not a crash.
                typer.echo(f"no requests in the last {since}.")
                raise typer.Exit(code=0)
        else:
            typer.echo("note: log has no ts column; reporting on all rows")

    result = build(
        frame,  # type: ignore[arg-type]
        baselines=tuple(baseline) if baseline else ("always:premium",),
        metrics=tuple(m.strip() for m in metrics.split(",") if m.strip()),
        window=window,
    )

    if out is not None:
        out.write_text(to_markdown(result), encoding="utf-8")
        typer.echo(f"wrote {out}")
    to_terminal(result)

    if check and not result.trustworthy:
        raise typer.Exit(code=1)


@app.command()
def train(
    log: Annotated[Path, typer.Argument(help="Log file to learn from.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Artifact to write.")] = Path(
        "learned.json"
    ),
    metric: Annotated[str, typer.Option(help="Outcome column to predict.")] = "outcome",
    cost_weight: Annotated[
        float, typer.Option(help="How much predicted quality to trade for price.")
    ] = 0.0,
    tune_cost_weight: Annotated[
        bool,
        typer.Option(
            "--tune-cost-weight",
            help="Choose the cost weight from the logs instead of taking it.",
        ),
    ] = False,
    tolerance: Annotated[
        float | None,
        typer.Option(help="Most quality to spend on price, when tuning."),
    ] = None,
    budget: Annotated[
        float | None,
        typer.Option(help="Ceiling on cost per request. Overrides --tolerance."),
    ] = None,
    epsilon: Annotated[
        float, typer.Option(help="Exploration floor for the trained policy.")
    ] = 0.05,
    holdout: Annotated[
        float, typer.Option(help="Share of rows kept back to evaluate on.")
    ] = 0.3,
    tune: Annotated[
        float, typer.Option(help="Share of rows the cost-weight sweep may see.")
    ] = 0.2,
    name: Annotated[str | None, typer.Option(help="Label for the artifact.")] = None,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Fit a policy from logs and report whether it beats what produced them."""
    from sortition.decide.artifact import save as save_artifact
    from sortition.eval import evaluate
    from sortition.targets import PolicyTarget
    from sortition.train import (
        DEFAULT_TOLERANCE,
        split_three_ways,
        sweep,
        train_test_split,
    )
    from sortition.train import train as fit

    if not tune_cost_weight and (
        budget is not None or tolerance is not None or tune != 0.2
    ):
        raise typer.BadParameter(
            "--budget, --tolerance and --tune only apply with --tune-cost-weight; "
            "without it the cost weight is whatever --cost-weight says, and a "
            "budget you asked for would be silently dropped"
        )
    if tune_cost_weight and cost_weight != 0.0:
        raise typer.BadParameter(
            "--cost-weight sets the exchange rate by hand and --tune-cost-weight "
            "chooses it from the log; pass one or the other"
        )

    frame = _load(log)
    swept = None

    if tune_cost_weight:
        # Three splits, not two. Choosing the weight on the rows the result is
        # quoted on would make the reported gain partly the luck of whichever
        # grid point suited that split.
        fit_rows, tune_rows, held_out = split_three_ways(
            frame,  # type: ignore[arg-type]
            tune=tune,
            holdout=holdout,
            seed=seed,
        )
        swept = sweep(
            fit_rows,
            tune_rows,
            metric=metric,
            epsilon=epsilon,
            budget=budget,
            tolerance=DEFAULT_TOLERANCE if tolerance is None else tolerance,
            name=name,
            seed=seed,
        )
        artifact = swept.artifact(name=name)
        policy, n_rows = swept.policy, swept.n_fit
        feature_spec = swept.feature_spec
        _echo_frontier(swept)
    else:
        result = fit(
            (rows := train_test_split(frame, holdout=holdout, seed=seed))[0],  # type: ignore[arg-type]
            metric=metric,
            cost_weight=cost_weight,
            epsilon=epsilon,
            name=name,
            seed=seed,
        )
        held_out = rows[1]
        artifact, policy = result.artifact, result.policy
        n_rows, feature_spec = result.n_rows, result.feature_spec

    save_artifact(artifact, out)
    typer.echo(f"trained {artifact.policy_version} on {n_rows} rows")
    typer.echo(f"features: {', '.join(feature_spec)}")
    typer.echo(f"wrote {out}")

    # Measured on rows the policy has not seen. Training and evaluating on the
    # same logs would flatter any candidate, which is the whole reason for the
    # holdout.
    target = PolicyTarget(policy=policy, epsilon=epsilon, name=artifact.policy_version)
    estimate = evaluate(held_out, target, metric=metric, estimator="dr")
    observed = float(held_out.get_column(metric).drop_nulls().to_numpy().mean())

    if swept is not None and swept.chosen.cost_weight != 0.0:
        # The saving printed with the frontier was measured on the rows the
        # winning weight was selected on, so it is the number most likely to be
        # flattered by the search. Re-measure the same trade-off here, against
        # the cost-blind policy, on rows neither the boosters nor the sweep saw.
        from sortition.eval import compare

        blind = PolicyTarget(
            policy=swept.policy_at(0.0), epsilon=epsilon, name="cost_weight=0"
        )
        for outcome in compare(
            held_out,
            a=blind,
            b=target,
            metrics=(metric, swept.cost_metric),
            seed=seed,
        ):
            low, high = outcome.difference_interval
            typer.echo(
                f"  held-out {outcome.metric} vs ignoring cost: "
                f"{outcome.difference:+.6g} [{low:+.6g}, {high:+.6g}]"
            )

    typer.echo("")
    typer.echo(f"on {held_out.height} held-out rows:")
    typer.echo(f"  what actually happened: {observed:.6g}")
    typer.echo(f"  this policy would have: {estimate.value:.6g}", nl=False)
    if estimate.interval is not None:
        typer.echo(f"  [{estimate.interval.low:.6g}, {estimate.interval.high:.6g}]")
    else:
        typer.echo("")

    if not estimate.trustworthy:
        typer.echo("\nNOT TRUSTWORTHY -- the held-out log cannot support this estimate")
        raise typer.Exit(code=1)
    if estimate.interval is not None and estimate.interval.low <= observed:
        typer.echo(
            "\nThe interval overlaps what already happened, so this policy is "
            "not measurably better. Deploying it would be a coin flip."
        )


@app.command()
def dashboard(
    log: Annotated[Path, typer.Argument(help="Log file to visualise.")],
    out: Annotated[
        Path, typer.Option("--out", "-o", help="HTML file to write.")
    ] = Path("dashboard.html"),
    baseline: Annotated[
        list[str] | None, typer.Option(help="Policy to compare against. Repeatable.")
    ] = None,
    metrics: Annotated[str, typer.Option(help="Comma-separated metric columns.")] = (
        "outcome,cost_usd"
    ),
    since: Annotated[str | None, typer.Option(help="Window, e.g. 7d, 24h, 2w.")] = None,
) -> None:
    """Write a self-contained HTML dashboard: one file, no server, no CDN."""
    import polars as pl

    from sortition.dashboard import write as write_dashboard
    from sortition.reporting import build, parse_since

    frame = _load(log)
    window = "all time"
    if since:
        cutoff = parse_since(since)
        window = f"last {since}"
        if "ts" in frame.columns:  # type: ignore[union-attr]
            frame = frame.filter(pl.col("ts") >= cutoff)  # type: ignore[union-attr]
            if frame.height == 0:  # type: ignore[union-attr]
                typer.echo(f"no requests in the last {since}.")
                raise typer.Exit(code=0)

    result = build(
        frame,  # type: ignore[arg-type]
        baselines=tuple(baseline) if baseline else ("uniform",),
        metrics=tuple(m.strip() for m in metrics.split(",") if m.strip()),
        window=window,
    )
    write_dashboard(result, frame, out)  # type: ignore[arg-type]
    typer.echo(f"wrote {out}")


def main() -> None:
    """Entry point for the ``sortition`` console script."""
    app()


def _echo_frontier(swept: Any) -> None:
    """Print the measured quality/price frontier and the point taken from it.

    The whole table is shown, not only the winner: the shape of the frontier is
    what tells an operator whether the choice was close, and a single chosen
    number hides a curve that was flat over three grid points.

    Args:
        swept: A :class:`~sortition.train.sweep.SweepResult`.
    """
    typer.echo("")
    typer.echo(f"cost-weight frontier, on {swept.n_tune} tuning rows:")
    # The cost column is named for the metric rather than labelled "per
    # request": these are dollars only if the log's cost column holds dollars,
    # and a bare number under a generic heading invites the reader to assume.
    typer.echo(
        f"  {'weight':>7}  {swept.metric:>9}  {swept.cost_metric + '/req':>13}  "
        f"{'vs ignoring cost':>26}"
    )
    for point in swept.frontier:
        low, high = point.quality_difference_interval
        mark = "*" if point.cost_weight == swept.chosen.cost_weight else " "
        typer.echo(
            f"{mark} {point.cost_weight:>7g}  {point.quality:>9.4f}  "
            f"{point.cost:>13.4g}  {point.quality_difference:>+7.4f} "
            f"[{low:+.4f}, {high:+.4f}]"
        )

    chosen = swept.chosen
    if chosen.cost_weight == 0.0:
        why = (
            f"no cheaper weight stays within {swept.tolerance:g} of ignoring cost"
            if swept.budget is None
            else "the budget is met without trading any quality"
        )
        typer.echo(f"\nchose cost_weight=0: {why}.")
        # A bare refusal cannot be acted on: it looks the same whether the dear
        # arm is genuinely worth it or the log is too thin to prove otherwise.
        # The margin that would have cleared, and what it buys, is the number
        # the operator actually needs.
        alternative = swept.next_best
        if alternative is not None and swept.budget is None:
            typer.echo(
                f"  --tolerance {alternative.tolerance_required:.3g} would take "
                f"cost_weight={alternative.cost_weight:g} and save "
                f"{alternative.saving:.3g} of {swept.cost_metric} per request."
            )
    else:
        spent = (
            f"gives up {-chosen.quality_difference:.4f} of {swept.metric}"
            if chosen.quality_difference < 0.0
            else f"and is {chosen.quality_difference:+.4f} on {swept.metric}"
        )
        typer.echo(
            f"\nchose cost_weight={chosen.cost_weight:g}: saves "
            f"{chosen.saving:.3g} of {swept.cost_metric} per request, {spent}."
        )


if __name__ == "__main__":
    main()
