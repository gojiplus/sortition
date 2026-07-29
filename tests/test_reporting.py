"""The operator report.

The rule this file exists to enforce: a savings figure computed from a blind log
renders exactly like one computed from a good log. A report that leads with the
number and buries the caveat is worse than no report, so the health verdict must
come first when it is bad.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from sortition.reporting import build, parse_since, to_markdown
from sortition.sim import epsilon_greedy_policy, make_problem, sample_logs
from sortition.sim.to_frame import to_frame


def _logs(*, epsilon: float = 0.3, n: int = 3_000) -> pl.DataFrame:
    problem = make_problem(n_contexts=200, n_arms=4, seed=0)
    weights = np.random.default_rng(0).standard_normal((6, 4))
    policy = epsilon_greedy_policy(weights, epsilon=epsilon)
    logs = sample_logs(problem, policy, n, seed=1)
    return to_frame(logs, problem, fallback_rate=0.02, seed=2)


class TestSince:
    @pytest.mark.parametrize(
        ("text", "delta"),
        [
            ("7d", timedelta(days=7)),
            ("24h", timedelta(hours=24)),
            ("2w", timedelta(weeks=2)),
        ],
    )
    def test_parses_durations(self, text: str, delta: timedelta) -> None:
        cutoff = parse_since(text)
        assert abs((datetime.now(UTC) - cutoff) - delta) < timedelta(seconds=5)

    @pytest.mark.parametrize("text", ["", "7", "d", "soon", "-3d", "7 months"])
    def test_rejects_nonsense(self, text: str) -> None:
        with pytest.raises(ValueError, match="could not read a duration"):
            parse_since(text)


class TestReport:
    def test_a_healthy_log_is_trustworthy(self) -> None:
        report = build(_logs(), baselines=("always:arm-3",))
        assert report.trustworthy
        assert report.health.n > 2_000
        assert "always:arm-3" in report.comparisons

    def test_observed_means_are_measured_not_estimated(self) -> None:
        logs = _logs()
        report = build(logs, baselines=("always:arm-3",))
        # These are what happened, so they must match the column exactly rather
        # than come from any estimator.
        served = logs.filter(pl.col("fallback_depth") == 0)
        assert report.observed["outcome"] == pytest.approx(
            float(logs.get_column("outcome").drop_nulls().to_numpy().mean()), rel=1e-9
        )
        assert served.height > 0

    def test_policy_versions_are_listed(self) -> None:
        report = build(_logs(), baselines=("always:arm-3",))
        assert report.policy_versions

    def test_unknown_baseline_is_skipped_not_fatal(self) -> None:
        # One bad baseline must not lose the rest of the report.
        report = build(_logs(), baselines=("always:nonexistent", "always:arm-0"))
        assert "always:arm-0" in report.comparisons
        assert "always:nonexistent" not in report.comparisons


class TestBlindLogLeadsWithTheWarning:
    @pytest.fixture
    def blind(self) -> pl.DataFrame:
        return _logs().with_columns(pl.lit(1.0).alias("propensity"))

    def test_report_is_marked_untrustworthy(self, blind: pl.DataFrame) -> None:
        assert not build(blind, baselines=("always:arm-3",)).trustworthy

    def test_warning_precedes_every_number(self, blind: pl.DataFrame) -> None:
        markdown = to_markdown(build(blind, baselines=("always:arm-3",)))
        warning_at = markdown.index("should not be acted on")
        # Any heading that carries an actionable figure must come after it.
        for heading in ("What actually happened", "Against `always:arm-3`"):
            assert markdown.index(heading) > warning_at, heading

    def test_the_reason_is_named(self, blind: pl.DataFrame) -> None:
        markdown = to_markdown(build(blind, baselines=("always:arm-3",)))
        assert "no real choice" in markdown
        assert "exploration floor" in markdown

    def test_estimates_are_still_shown(self, blind: pl.DataFrame) -> None:
        # Hiding them would invite someone to recompute them by hand, with no
        # warning attached at all.
        markdown = to_markdown(build(blind, baselines=("always:arm-3",)))
        assert "Against `always:arm-3`" in markdown

    def test_a_healthy_report_has_no_warning_banner(self) -> None:
        markdown = to_markdown(build(_logs(), baselines=("always:arm-3",)))
        assert "should not be acted on" not in markdown


class TestMarkdown:
    def test_has_the_expected_sections(self) -> None:
        markdown = to_markdown(build(_logs(), baselines=("always:arm-0",)))
        for section in ("# Routing report", "What actually happened", "Log health"):
            assert section in markdown

    def test_tables_are_well_formed(self) -> None:
        markdown = to_markdown(build(_logs(), baselines=("always:arm-0",)))
        for line in markdown.splitlines():
            if line.startswith("|") and "---" not in line:
                # Leading and trailing pipes plus one per column boundary.
                assert line.endswith("|"), line

    def test_pooled_policies_are_called_out(self) -> None:
        # Two policies in one window make a single estimate a statement about
        # the mixture, which is rarely what someone means.
        mixed = pl.concat(
            [
                _logs(n=1_500).with_columns(pl.lit("rules-a").alias("policy_version")),
                _logs(n=1_500).with_columns(pl.lit("rules-b").alias("policy_version")),
            ],
            how="diagonal_relaxed",
        )
        markdown = to_markdown(build(mixed, baselines=("always:arm-0",)))
        assert "More than one policy served this window" in markdown
