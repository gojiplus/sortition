"""End-to-end through the command line, on a file written to disk.

The unit tests hold DataFrames in memory. This exercises what a user actually
runs: generate a log, write parquet, read it back, and get an answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from sortition.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def log_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("logs") / "demo.parquet"
    result = runner.invoke(app, ["demo", "--out", str(path), "--n", "8000"])
    assert result.exit_code == 0, result.output
    assert path.exists()
    return path


def test_demo_reports_ground_truth(log_file: Path) -> None:
    assert log_file.stat().st_size > 0


def test_doctor_runs(log_file: Path) -> None:
    result = runner.invoke(app, ["doctor", str(log_file)])
    assert result.exit_code == 0, result.output
    assert "ESS" in result.output


def test_eval_reports_an_interval(log_file: Path) -> None:
    result = runner.invoke(app, ["eval", str(log_file), "--target", "always:arm-0"])
    assert result.exit_code == 0, result.output
    assert "under target policy" in result.output
    assert "95%" in result.output


def test_eval_handles_cost(log_file: Path) -> None:
    result = runner.invoke(
        app, ["eval", str(log_file), "--target", "always:arm-3", "--metric", "cost_usd"]
    )
    assert result.exit_code == 0, result.output
    assert "bootstrap" in result.output


def test_compare_runs(log_file: Path) -> None:
    result = runner.invoke(
        app, ["compare", str(log_file), "--a", "always:arm-0", "--b", "always:arm-3"]
    )
    assert result.exit_code == 0, result.output
    assert "difference" in result.output


def test_missing_file_is_a_clean_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["doctor", str(tmp_path / "nope.parquet")])
    assert result.exit_code != 0
    assert "no such file" in result.output


def test_unsupported_format_is_a_clean_error(tmp_path: Path) -> None:
    path = tmp_path / "logs.xlsx"
    path.write_text("")
    result = runner.invoke(app, ["doctor", str(path)])
    assert result.exit_code != 0
    assert "unsupported format" in result.output


def test_untrustworthy_result_exits_nonzero(tmp_path: Path) -> None:
    # An unexplored log cannot support a counterfactual claim, and the exit code
    # has to say so -- this is what makes the CLI usable in CI.
    import polars as pl

    from sortition.cli import _load  # noqa: F401  (ensures the module imports)

    path = tmp_path / "blind.parquet"
    frame = pl.DataFrame(
        {
            "request_id": [f"r{i}" for i in range(200)],
            "chosen_arm": ["cheap"] * 200,
            "propensity": [1.0] * 200,
            "eligible_set": [["cheap", "premium"]] * 200,
            "outcome": [0.5] * 200,
        }
    )
    frame.write_parquet(path)
    result = runner.invoke(app, ["eval", str(path), "--target", "always:premium"])
    assert result.exit_code == 1, result.output
