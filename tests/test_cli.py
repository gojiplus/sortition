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


RULES_YAML = """
arms: [cheap, standard, premium]
default: [standard, cheap, premium]
rules:
  - label: tools_required
    when: {tools_required: true}
    exclude: [cheap]
  - label: long_context
    when: {context_tokens: {gte: 32000}}
    prefer: [premium]
"""


class TestPolicyCommands:
    def _rules(self, tmp_path: Path) -> Path:
        path = tmp_path / "rules.yaml"
        path.write_text(RULES_YAML, encoding="utf-8")
        return path

    def test_build_then_show_then_serve(self, tmp_path: Path) -> None:
        """The whole point of an artifact: what `show` prints is what serves.

        If the version `show` reports and the version a running engine stamps on
        its log rows can drift apart, then reading a log tells you nothing about
        which policy produced it.
        """
        from sortition.decide import ReloadingEngine

        artifact = tmp_path / "policy.json"
        built = runner.invoke(
            app,
            ["policy", "build", str(self._rules(tmp_path)), "-o", str(artifact)],
        )
        assert built.exit_code == 0, built.output
        version = built.output.split()[1]

        shown = runner.invoke(app, ["policy", "show", str(artifact)])
        assert shown.exit_code == 0, shown.output
        assert version in shown.output
        assert "tools_required" in shown.output
        assert "eps=0.05" in shown.output

        engine = ReloadingEngine(artifact, poll_interval=0.0)
        decision = engine.decide(
            features={"tools_required": True}, eligible=["cheap", "standard", "premium"]
        )
        assert decision.policy_version == version
        # The hard constraint in the rule table is in force, not just printed.
        assert "cheap" not in decision.eligible_set

    def test_identical_rules_produce_an_identical_version(self, tmp_path: Path) -> None:
        rules = self._rules(tmp_path)
        versions = set()
        for i in range(2):
            result = runner.invoke(
                app, ["policy", "build", str(rules), "-o", str(tmp_path / f"p{i}.json")]
            )
            assert result.exit_code == 0
            versions.add(result.output.split()[1])
        assert len(versions) == 1

    def test_epsilon_alone_changes_the_version(self, tmp_path: Path) -> None:
        # Same preferences, different sampling distribution: pooling their logs
        # would be wrong, so they must not share a version.
        rules = self._rules(tmp_path)
        out = []
        for eps in ("0.05", "0.30"):
            result = runner.invoke(
                app,
                [
                    "policy",
                    "build",
                    str(rules),
                    "-o",
                    str(tmp_path / f"p{eps}.json"),
                    "--epsilon",
                    eps,
                ],
            )
            out.append(result.output.split()[1])
        assert out[0] != out[1]

    def test_build_warns_when_exploration_is_off(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "policy",
                "build",
                str(self._rules(tmp_path)),
                "-o",
                str(tmp_path / "blind.json"),
                "--epsilon",
                "0",
            ],
        )
        assert result.exit_code == 0
        assert "explores less than 1%" in result.output

    def test_missing_files_are_clean_errors(self, tmp_path: Path) -> None:
        for args in (
            ["policy", "build", str(tmp_path / "absent.yaml")],
            ["policy", "show", str(tmp_path / "absent.json")],
        ):
            assert runner.invoke(app, args).exit_code != 0


class TestDoctorHealth:
    def test_doctor_reports_health(self, log_file: Path) -> None:
        result = runner.invoke(app, ["doctor", str(log_file)])
        assert result.exit_code == 0, result.output
        assert "arm share" in result.output

    def test_check_flag_fails_on_a_blind_log(self, tmp_path: Path) -> None:
        # Regression guard. The health wiring was written once, silently failed
        # to apply, and shipped in a commit that claimed it worked -- because
        # nothing exercised it.
        import polars as pl

        source = tmp_path / "blind.parquet"
        assert (
            runner.invoke(app, ["demo", "--out", str(source), "--n", "2000"]).exit_code
            == 0
        )
        pl.read_parquet(source).with_columns(
            pl.lit(1.0).alias("propensity")
        ).write_parquet(source)

        assert runner.invoke(app, ["doctor", str(source)]).exit_code == 0
        checked = runner.invoke(app, ["doctor", str(source), "--check"])
        assert checked.exit_code == 1
        assert "NOT HEALTHY" in checked.output
