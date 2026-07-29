"""Command line interface.

    sortition doctor  logs.parquet
    sortition eval    logs.parquet --target always:premium --metric cost_usd
    sortition compare logs.parquet --a uniform --b always:premium
    sortition demo    --out logs.parquet

``doctor`` comes first on purpose. Whether a log can support a counterfactual
claim is prior to what the claim is, and it is the question people skip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="sortition",
    help="Counterfactual evaluation for LLM routing policies.",
    no_args_is_help=True,
    add_completion=False,
)


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
    ] = ("uniform"),
) -> None:
    """Report whether a log can support counterfactual claims."""
    from sortition.eval.report import doctor as run_doctor

    typer.echo(run_doctor(_load(log), target))  # type: ignore[arg-type]


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


def main() -> None:
    """Entry point for the ``sortition`` console script."""
    app()


if __name__ == "__main__":
    main()
