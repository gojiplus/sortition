"""The log-table path: frame -> arrays -> estimate -> comparison.

These run against a simulated log rendered into the same flat table a gateway
produces, so the ground truth stays known all the way through the public API.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from sortition.eval import compare, doctor, evaluate, to_dicts
from sortition.frame import arm_universe, to_arrays
from sortition.sim import (
    BanditProblem,
    constant_policy,
    epsilon_greedy_policy,
    make_problem,
    sample_logs,
)
from sortition.sim.to_frame import to_frame
from sortition.targets import AlwaysArm, EpsilonFloor, Uniform, parse_target


@pytest.fixture(scope="module")
def scenario() -> tuple[BanditProblem, pl.DataFrame]:
    """A 20k-row log from an epsilon-greedy policy, with 2% gateway fallbacks."""
    problem = make_problem(n_contexts=500, n_arms=4, seed=0)
    weights = np.random.default_rng(0).standard_normal((6, 4))
    policy = epsilon_greedy_policy(weights, epsilon=0.2)
    logs = sample_logs(problem, policy, 20_000, seed=1)
    return problem, to_frame(logs, problem, fallback_rate=0.02, seed=2)


class TestFrame:
    def test_arm_universe_is_stable_and_complete(
        self, scenario: tuple[BanditProblem, pl.DataFrame]
    ) -> None:
        problem, frame = scenario
        assert arm_universe(frame) == tuple(sorted(problem.arms))

    def test_fallback_rows_are_dropped_and_counted(
        self, scenario: tuple[BanditProblem, pl.DataFrame]
    ) -> None:
        _, frame = scenario
        arrays = to_arrays(frame)
        expected = int((frame.get_column("fallback_depth") > 0).sum())
        # Those rows served an arm the sampler never drew, so their outcome
        # cannot be attributed to the logged propensity.
        assert arrays.n_excluded_leakage == expected
        assert arrays.n == frame.height - expected

    def test_keeping_fallbacks_is_opt_in(
        self, scenario: tuple[BanditProblem, pl.DataFrame]
    ) -> None:
        _, frame = scenario
        assert to_arrays(frame, drop_fallbacks=False).n == frame.height

    def test_propensities_survive_the_round_trip(
        self, scenario: tuple[BanditProblem, pl.DataFrame]
    ) -> None:
        _, frame = scenario
        arrays = to_arrays(frame)
        kept = frame.filter(pl.col("fallback_depth") == 0)
        np.testing.assert_allclose(
            arrays.propensity, kept.get_column("propensity").to_numpy(), rtol=1e-12
        )

    def test_missing_propensity_column_is_a_clear_error(self) -> None:
        frame = pl.DataFrame({"request_id": ["a"], "chosen_arm": ["x"], "eligible_set": [["x"]]})
        with pytest.raises(ValueError, match="propensity"):
            to_arrays(frame)

    def test_numeric_features_become_a_context_matrix(
        self, scenario: tuple[BanditProblem, pl.DataFrame]
    ) -> None:
        _, frame = scenario
        arrays = to_arrays(frame)
        # n_tokens, code_fraction, context_tokens, tools_required
        assert arrays.contexts.shape == (arrays.n, 4)


class TestEvaluate:
    @pytest.mark.parametrize("arm_index", [0, 1, 2, 3])
    def test_recovers_true_outcome_of_a_constant_policy(
        self, scenario: tuple[BanditProblem, pl.DataFrame], arm_index: int
    ) -> None:
        problem, frame = scenario
        truth = problem.value(constant_policy(arm_index))
        est = evaluate(frame, f"always:{problem.arms[arm_index]}", metric="outcome")
        assert est.interval is not None
        assert est.interval.covers(truth)

    @pytest.mark.parametrize("arm_index", [0, 2, 3])
    def test_recovers_true_cost_of_a_constant_policy(
        self, scenario: tuple[BanditProblem, pl.DataFrame], arm_index: int
    ) -> None:
        problem, frame = scenario
        truth = problem.cost_value(constant_policy(arm_index))
        est = evaluate(frame, f"always:{problem.arms[arm_index]}", metric="cost_usd")
        assert est.interval is not None
        assert est.interval.covers(truth)

    def test_unbounded_metrics_use_the_bootstrap(
        self, scenario: tuple[BanditProblem, pl.DataFrame]
    ) -> None:
        _, frame = scenario
        assert evaluate(frame, "uniform", metric="cost_usd").interval.method == "bootstrap"  # type: ignore[union-attr]
        assert evaluate(frame, "uniform", metric="outcome").interval.method == "betting"  # type: ignore[union-attr]

    def test_late_arriving_outcomes_are_dropped_not_imputed(
        self, scenario: tuple[BanditProblem, pl.DataFrame]
    ) -> None:
        problem, frame = scenario
        # Half the outcomes have not come back yet, as with CSAT or resolution.
        partial = frame.with_columns(
            pl.when(pl.int_range(pl.len()) % 2 == 0)
            .then(pl.col("outcome"))
            .otherwise(None)
            .alias("outcome")
        )
        est = evaluate(partial, "uniform", metric="outcome")
        assert est.n == pytest.approx(to_arrays(frame).n / 2, rel=0.05)
        assert est.interval is not None
        assert est.interval.covers(problem.value(lambda c, e: np.where(e, 0.25, 0.0)))

    def test_unknown_metric_names_what_is_available(
        self, scenario: tuple[BanditProblem, pl.DataFrame]
    ) -> None:
        _, frame = scenario
        with pytest.raises(ValueError, match="available"):
            evaluate(frame, "uniform", metric="csat")

    def test_unknown_arm_is_rejected(self, scenario: tuple[BanditProblem, pl.DataFrame]) -> None:
        _, frame = scenario
        with pytest.raises(ValueError, match="unknown arm"):
            evaluate(frame, "always:gpt-9", metric="outcome")


class TestCompare:
    def test_difference_lies_inside_its_own_interval(
        self, scenario: tuple[BanditProblem, pl.DataFrame]
    ) -> None:
        # Regression guard. The point estimate came from the DR scores while the
        # interval was bootstrapped from IPS scores, so the reported difference
        # fell outside its own confidence interval. Both must come from the same
        # per-row scores.
        problem, frame = scenario
        for result in compare(frame, f"always:{problem.arms[0]}", f"always:{problem.arms[3]}"):
            low, high = result.difference_interval
            assert low <= result.difference <= high, result

    def test_difference_equals_the_gap_between_the_two_estimates(
        self, scenario: tuple[BanditProblem, pl.DataFrame]
    ) -> None:
        problem, frame = scenario
        for result in compare(frame, f"always:{problem.arms[1]}", f"always:{problem.arms[2]}"):
            assert result.difference == pytest.approx(result.b.value - result.a.value, rel=1e-9)

    def test_recovers_the_true_gap(self, scenario: tuple[BanditProblem, pl.DataFrame]) -> None:
        problem, frame = scenario
        results = compare(frame, f"always:{problem.arms[0]}", f"always:{problem.arms[3]}")
        truths = {
            "outcome": problem.value(constant_policy(3)) - problem.value(constant_policy(0)),
            "cost_usd": problem.cost_value(constant_policy(3))
            - problem.cost_value(constant_policy(0)),
        }
        for result in results:
            low, high = result.difference_interval
            assert low <= truths[result.metric] <= high, result

    def test_a_real_gap_is_called_significant(
        self, scenario: tuple[BanditProblem, pl.DataFrame]
    ) -> None:
        problem, frame = scenario
        # Cheapest against most expensive: a 50x cost gap is not a close call.
        results = compare(frame, f"always:{problem.arms[0]}", f"always:{problem.arms[3]}")
        assert all(r.significant for r in results)

    def test_a_policy_against_itself_shows_no_difference(
        self, scenario: tuple[BanditProblem, pl.DataFrame]
    ) -> None:
        problem, frame = scenario
        target = f"always:{problem.arms[2]}"
        for result in compare(frame, target, target):
            assert result.difference == pytest.approx(0.0, abs=1e-12)
            assert not result.significant

    def test_to_dicts_is_dataframe_ready(
        self, scenario: tuple[BanditProblem, pl.DataFrame]
    ) -> None:
        problem, frame = scenario
        rows = to_dicts(compare(frame, "uniform", f"always:{problem.arms[0]}"))
        assert pl.DataFrame(rows).height == len(rows)
        assert {"metric", "difference", "ci_low", "ci_high", "significant"} <= set(rows[0])


class TestDoctor:
    def test_reports_leakage_and_overlap(
        self, scenario: tuple[BanditProblem, pl.DataFrame]
    ) -> None:
        _, frame = scenario
        report = doctor(frame)
        assert "ESS" in report
        assert "gateway fallback overrode the sampler" in report

    def test_warns_when_the_log_has_no_exploration(
        self, scenario: tuple[BanditProblem, pl.DataFrame]
    ) -> None:
        # A deterministic policy produces logs that can only confirm what it
        # already does. Saying so loudly is the whole point of this command.
        _, frame = scenario
        blind = frame.with_columns(pl.lit(1.0).alias("propensity"))
        assert "effectively unexplored" in doctor(blind)


class TestTargets:
    def test_parses_the_supported_forms(self) -> None:
        assert isinstance(parse_target("uniform"), Uniform)
        assert parse_target("always:premium") == AlwaysArm("premium")
        floored = parse_target("always:premium+eps0.05")
        assert isinstance(floored, EpsilonFloor)
        assert floored.epsilon == 0.05
        assert floored.name == "always:premium+eps0.05"

    def test_rejects_nonsense(self) -> None:
        for spec in ("", "always:", "greedy", "always:x+epsabc"):
            with pytest.raises(ValueError):
                parse_target(spec)

    def test_epsilon_floor_keeps_every_arm_reachable(self) -> None:
        problem = make_problem(n_contexts=50, n_arms=4, seed=7)
        probs = parse_target("always:arm-0+eps0.1").probabilities(
            [{}] * problem.n_contexts, problem.eligible, problem.arms
        )
        # Without a floor an evaluable log is impossible; assert the floor holds.
        assert (probs > 0).all()
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-12)
