"""Metrics and the live health signal.

The health check exists for one failure: a log that has gone blind. Exploration
gets turned down or a policy becomes deterministic, and everything still looks
fine -- requests succeed, latency is normal, cost is normal -- while the logs
quietly stop being able to justify any of it. ``ok`` is what makes that visible.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from sortition import health, metrics
from sortition.sim import epsilon_greedy_policy, make_problem, sample_logs
from sortition.sim.to_frame import to_frame


def _logs(
    *, epsilon: float = 0.3, n: int = 2_000, fallback_rate: float = 0.0
) -> pl.DataFrame:
    problem = make_problem(n_contexts=200, n_arms=4, seed=0)
    weights = np.random.default_rng(0).standard_normal((6, 4))
    policy = epsilon_greedy_policy(weights, epsilon=epsilon)
    logs = sample_logs(problem, policy, n, seed=1)
    return to_frame(logs, problem, fallback_rate=fallback_rate, seed=2)


class TestHealth:
    def test_a_well_explored_log_is_healthy(self) -> None:
        report = health.assess(_logs(epsilon=0.3))
        assert report.ok, report.explain()
        assert report.n > 1_000
        assert sum(report.arm_share.values()) == pytest.approx(1.0)

    def test_a_blind_log_is_not_healthy(self) -> None:
        # The failure this whole module exists for: nothing about these rows
        # looks wrong, and no counterfactual claim can be made from them.
        blind = _logs().with_columns(
            pl.lit(1.0).alias("propensity"), pl.lit(False).alias("explore")
        )
        report = health.assess(blind)
        assert not report.ok
        assert report.deterministic_rate == 1.0
        assert any("no real choice" in w for w in report.warnings)
        assert any("exploration floor" in w for w in report.warnings)

    def test_heavy_gateway_fallback_is_flagged(self) -> None:
        report = health.assess(_logs(fallback_rate=0.25))
        assert not report.ok
        assert report.leakage_rate > 0.05
        assert any("rerouted by the gateway" in w for w in report.warnings)

    def test_too_few_rows_refuses_to_judge(self) -> None:
        report = health.assess(_logs(n=10))
        assert not report.ok
        assert any("too few to judge" in w for w in report.warnings)

    def test_explain_is_readable(self) -> None:
        text = health.assess(_logs()).explain()
        assert "arm share" in text
        assert "HEALTHY" in text

    def test_exploration_rate_tracks_epsilon(self) -> None:
        # explore=True is recorded only when the draw actually differs from the
        # greedy arm, so the observed rate sits below epsilon by eps/n_arms.
        low = health.assess(_logs(epsilon=0.1)).exploration_rate
        high = health.assess(_logs(epsilon=0.6)).exploration_rate
        assert low < high


class TestMetrics:
    def test_publishing_health_sets_gauges(self) -> None:
        if not metrics.AVAILABLE:
            pytest.skip("prometheus_client not installed")
        from prometheus_client import REGISTRY

        health.assess(_logs(epsilon=0.3))
        assert REGISTRY.get_sample_value("sortition_health_ok") == 1.0

        blind = _logs().with_columns(pl.lit(1.0).alias("propensity"))
        health.assess(blind)
        assert REGISTRY.get_sample_value("sortition_health_ok") == 0.0

    def test_decisions_are_counted(self) -> None:
        if not metrics.AVAILABLE:
            pytest.skip("prometheus_client not installed")
        from prometheus_client import REGISTRY

        def total() -> float:
            return (
                REGISTRY.get_sample_value(
                    "sortition_decisions_total",
                    {"arm": "premium", "policy_version": "v1", "explore": "False"},
                )
                or 0.0
            )

        before = total()
        metrics.observe_decision(
            arm="premium",
            policy_version="v1",
            explore=False,
            propensity_value=0.9,
            seconds=0.0001,
        )
        assert total() == before + 1

    def test_fallbacks_are_counted_only_when_they_happen(self) -> None:
        if not metrics.AVAILABLE:
            pytest.skip("prometheus_client not installed")
        from prometheus_client import REGISTRY

        def total() -> float:
            return (
                REGISTRY.get_sample_value(
                    "sortition_fallbacks_total",
                    {"chosen_arm": "premium", "served_arm": "cheap"},
                )
                or 0.0
            )

        before = total()
        metrics.observe_execution(
            chosen_arm="premium",
            served_arm="premium",
            fallback_depth=0,
            cost=0.01,
            tokens_in=10,
            tokens_out=5,
        )
        assert total() == before

        metrics.observe_execution(
            chosen_arm="premium",
            served_arm="cheap",
            fallback_depth=1,
            cost=0.001,
            tokens_in=10,
            tokens_out=5,
        )
        assert total() == before + 1

    def test_metrics_are_a_no_op_without_the_extra(self) -> None:
        # The whole package calls these unconditionally, so they must be safe
        # when prometheus_client is absent.
        noop = metrics._NoOp()
        noop.labels(arm="x").inc()
        noop.observe(1.0)
        noop.set(0)
