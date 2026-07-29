"""The HTML dashboard.

Two properties are worth enforcing in tests rather than trusting.

It must be **self-contained**: one file that opens from ``file://`` and survives
being emailed. A single CDN reference turns it into something that breaks offline
and leaks a request to whoever hosts the asset.

And the health verdict must come **before** any number, for the same reason it
does in the markdown report: a savings figure computed from a blind log renders
exactly like one computed from a good log.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sortition.dashboard import render, write
from sortition.reporting import build
from sortition.sim import epsilon_greedy_policy, make_problem, sample_logs
from sortition.sim.to_frame import to_frame


def _logs(*, epsilon: float = 0.3, n: int = 3_000) -> pl.DataFrame:
    problem = make_problem(n_contexts=200, n_arms=4, seed=0)
    weights = np.random.default_rng(0).standard_normal((6, 4))
    policy = epsilon_greedy_policy(weights, epsilon=epsilon)
    logs = sample_logs(problem, policy, n, seed=1)
    return to_frame(logs, problem, fallback_rate=0.01, seed=2)


@pytest.fixture(scope="module")
def page() -> str:
    logs = _logs()
    return render(build(logs, baselines=("always:arm-3",)), logs)


class TestSelfContained:
    def test_no_external_references(self, page: str) -> None:
        # A CDN reference would break the file offline and leak a request to
        # whoever hosts the asset.
        for pattern in ("http://", "https://", "@import", "<script", "src="):
            assert pattern not in page, pattern

    def test_is_a_complete_document(self, page: str) -> None:
        assert page.startswith("<!doctype html>")
        assert page.rstrip().endswith("</html>")
        assert "<title>" in page

    def test_styles_are_inline(self, page: str) -> None:
        assert "<style>" in page
        assert 'rel="stylesheet"' not in page

    def test_writes_to_disk(self, tmp_path: Path) -> None:
        logs = _logs(n=1_000)
        target = write(build(logs, baselines=("uniform",)), logs, tmp_path / "d.html")
        assert target.exists()
        assert target.read_text(encoding="utf-8").startswith("<!doctype html>")


class TestVerdictComesFirst:
    def test_a_blind_log_is_warned_about_before_any_number(self) -> None:
        logs = _logs().with_columns(pl.lit(1.0).alias("propensity"))
        page = render(build(logs, baselines=("always:arm-3",)), logs)

        warning_at = page.index("should not be acted on")
        # Every heading carrying an actionable figure must come after it.
        for heading in ("Share of requests by arm", "Against always:arm-3"):
            assert page.index(heading) > warning_at, heading

    def test_a_healthy_log_has_no_alert(self, page: str) -> None:
        assert "should not be acted on" not in page

    def test_the_reason_is_named(self) -> None:
        logs = _logs().with_columns(pl.lit(1.0).alias("propensity"))
        page = render(build(logs, baselines=("uniform",)), logs)
        assert "no real choice" in page


class TestAccessibility:
    def test_every_chart_has_a_table_view(self, page: str) -> None:
        # Identity must never be carried by colour alone.
        charts = len(re.findall(r"<svg ", page))
        tables = len(re.findall(r"<details>", page))
        assert charts >= 3
        assert tables >= charts - 1  # the share bar chart and each comparison

    def test_every_chart_is_labelled(self, page: str) -> None:
        assert len(re.findall(r"aria-label=", page)) == len(re.findall(r"<svg ", page))

    def test_marks_carry_hover_titles(self, page: str) -> None:
        # Native <title> needs no script and is read out by screen readers,
        # which a custom div tooltip is not.
        assert page.count("<title>") > 5

    def test_multiple_series_get_a_legend(self, page: str) -> None:
        assert 'class="legend"' in page

    def test_dark_mode_is_selected_not_flipped(self, page: str) -> None:
        # Its own steps from the same hues, and the theme toggle must win over
        # the OS setting in both directions.
        assert "prefers-color-scheme: dark" in page
        assert '[data-theme="dark"]' in page
        assert ':not([data-theme="light"])' in page


class TestGeometry:
    def test_nothing_is_negative_or_overflows(self, page: str) -> None:
        problems = []
        for match in re.finditer(
            r'<svg viewBox="0 0 (\d+) (\d+)"(.*?)</svg>', page, re.S
        ):
            width, height, body = (
                int(match.group(1)),
                int(match.group(2)),
                match.group(3),
            )
            for attr in ("width", "height", "r"):
                problems += [
                    f"negative {attr}={v}"
                    for v in re.findall(rf'\b{attr}="(-?[\d.]+)"', body)
                    if float(v) < 0
                ]
            problems += [
                f"y={v} outside 0..{height}"
                for v in re.findall(r'\by="(-?[\d.]+)"', body)
                if not -4 <= float(v) <= height + 8
            ]
            problems += [
                f"x={v} far outside 0..{width}"
                for v in re.findall(r'\bx="(-?[\d.]+)"', body)
                if float(v) < 0
            ]
        assert not problems, problems

    def test_survives_a_single_arm(self) -> None:
        # A degenerate log must not produce a broken chart.
        problem = make_problem(n_contexts=50, n_arms=1, seed=3)
        logs = sample_logs(problem, lambda c, e: np.ones((len(c), 1)), 600, seed=4)
        frame = to_frame(logs, problem, seed=5)
        page = render(build(frame, baselines=("uniform",)), frame)
        assert "<svg" in page

    def test_survives_a_log_with_no_timestamps(self) -> None:
        logs = _logs(n=1_000).drop("ts")
        page = render(build(logs, baselines=("uniform",)), logs)
        # The over-time chart is skipped rather than rendered empty.
        assert "Where traffic went" not in page
        assert "Share of requests by arm" in page

    def test_escapes_arm_names(self) -> None:
        # An arm name is a model string from someone's config; it reaches the
        # page as text, not as markup.
        logs = _logs(n=800).with_columns(
            pl.lit("<img src=x onerror=alert(1)>").alias("chosen_arm")
        )
        page = render(build(logs, baselines=("uniform",)), logs)
        assert "<img src=x" not in page
        assert "&lt;img" in page
