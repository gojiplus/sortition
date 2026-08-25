"""Choosing the exchange rate between quality and price from logs.

``cost_weight`` decides how much predicted quality a policy will give up to save
a dollar. It was a number the operator typed. Nothing about a routing problem
makes that number guessable: it depends on the price gap between the arms, on how
often the expensive arm is actually better, and on the traffic mix.

Two things have to hold for a sweep to be worth trusting, and each has a test
here. The frontier must be real -- spending less must actually cost quality, or
the sweep is reading noise. And the point must be chosen somewhere other than
where it is reported, or the winner of a nine-way search is partly whichever
grid point got luckiest on the split it was scored on.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from sortition.sim import epsilon_greedy_policy, make_problem, sample_logs
from sortition.sim.to_frame import to_frame
from sortition.targets import PolicyTarget
from sortition.train import split_three_ways, sweep, train


def _problem_and_logs(
    *, n: int = 24_000, seed: int = 0
) -> tuple[object, object, pl.DataFrame]:
    problem = make_problem(n_contexts=800, n_arms=4, seed=seed)
    weights = np.random.default_rng(seed).standard_normal((6, 4))
    behavior = epsilon_greedy_policy(weights, epsilon=0.5)
    logs = sample_logs(problem, behavior, n, seed=seed + 1)
    return problem, logs, to_frame(logs, problem, fallback_rate=0.0, seed=seed + 2)


class TestSplit:
    def test_the_three_parts_are_disjoint_and_complete(self) -> None:
        _, _, frame = _problem_and_logs(n=2_000)
        fit, tune, held = split_three_ways(frame, tune=0.25, holdout=0.25, seed=0)

        ids = [
            set(part.get_column("request_id").to_list()) for part in (fit, tune, held)
        ]
        assert fit.height + tune.height + held.height == frame.height
        assert not ids[0] & ids[1]
        assert not ids[0] & ids[2]
        assert not ids[1] & ids[2]

    def test_the_shares_are_respected(self) -> None:
        _, _, frame = _problem_and_logs(n=4_000)
        _, tune, held = split_three_ways(frame, tune=0.2, holdout=0.3, seed=0)
        assert tune.height / frame.height == pytest.approx(0.2, abs=0.03)
        assert held.height / frame.height == pytest.approx(0.3, abs=0.03)


class TestFrontier:
    def test_spending_less_is_what_a_higher_weight_buys(self) -> None:
        # If estimated cost did not fall as the weight rises, the cost term is
        # inert and there is no trade-off to choose a point on.
        _, _, frame = _problem_and_logs()
        fit, tune, _ = split_three_ways(frame, seed=0)
        result = sweep(fit, tune, grid=(0.0, 0.25, 0.5, 1.0, 2.0), seed=0)

        costs = [point.cost for point in result.frontier]
        assert costs == sorted(costs, reverse=True)
        assert costs[0] > 1.5 * costs[-1]

    def test_the_batched_scores_match_the_deployed_path(self) -> None:
        # The sweep rescores the log in two batched booster calls; a serving
        # request goes one row at a time. A fast path that disagreed with the
        # deployed one would tune a weight for a policy that never ships.
        _, _, frame = _problem_and_logs(n=6_000)
        policy = train(frame, cost_weight=1.0).policy
        rows = frame.head(50).get_column("features").to_list()

        quality, cost = policy.score_matrix(
            rows, np.ones((len(rows), len(policy.arms)), dtype=bool)
        )
        for i, features in enumerate(rows):
            one_by_one = policy.predict(features, policy.arms)
            assert list(quality[i]) == pytest.approx(
                [one_by_one[a] for a in policy.arms], rel=1e-9
            )
            priced = policy.predict_cost(features, policy.arms)
            assert list(cost[i]) == pytest.approx(
                [priced[a] for a in policy.arms], rel=1e-9
            )

    def test_chunking_does_not_change_a_single_prediction(self) -> None:
        # The batch is (rows x arms) design rows wide. At four arms that is a few
        # megabytes; at the hundred-model rosters routing exists for it is
        # gigabytes, so it is cut into blocks. A block boundary that dropped or
        # misaligned a row would shift a policy's choice with nothing to say so.
        from sortition.decide import tree as tree_module

        _, _, frame = _problem_and_logs(n=6_000)
        policy = train(frame, cost_weight=1.0).policy
        rows = frame.head(500).get_column("features").to_list()
        mask = np.ones((len(rows), len(policy.arms)), dtype=bool)

        whole = policy.score_matrix(rows, mask)
        original = tree_module.MAX_DESIGN_CELLS
        try:
            tree_module.MAX_DESIGN_CELLS = 64
            chunked = policy.score_matrix(rows, mask)
        finally:
            tree_module.MAX_DESIGN_CELLS = original

        assert np.array_equal(whole[0], chunked[0])
        assert np.array_equal(whole[1], chunked[1])

    def test_the_chosen_weight_evaluates_the_same_through_the_target_path(self) -> None:
        # The sweep's own numbers must agree with what `evaluate` reports for the
        # artifact that gets deployed, or the frontier describes a policy the
        # rest of the system does not.
        from sortition.eval import evaluate

        _, _, frame = _problem_and_logs()
        fit, tune, _ = split_three_ways(frame, seed=0)
        result = sweep(fit, tune, grid=(0.0, 0.5, 2.0), seed=0)

        for point in result.frontier:
            target = PolicyTarget(
                policy=result.policy_at(point.cost_weight),
                epsilon=result.epsilon,
                name=f"w{point.cost_weight}",
            )
            direct = evaluate(tune, target, metric="outcome", estimator="dr")
            assert point.quality == pytest.approx(direct.value, rel=1e-6)


class TestSelection:
    def test_it_refuses_to_trade_when_the_dear_arm_really_is_better(self) -> None:
        # The simulator has a genuine quality ladder: every dollar saved buys a
        # measurably worse outcome. Asked for a trade costing at most a point of
        # outcome, the only defensible answer is that there isn't one.
        _, _, frame = _problem_and_logs()
        fit, tune, _ = split_three_ways(frame, seed=0)
        result = sweep(
            fit, tune, grid=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0), tolerance=0.01, seed=0
        )

        assert result.chosen.cost_weight == 0.0
        assert all(
            not point.non_inferior_at(0.01)
            for point in result.frontier
            if point.cost_weight > 0.0
        )

    def test_a_stated_tolerance_is_what_unlocks_a_trade(self) -> None:
        _, _, frame = _problem_and_logs()
        fit, tune, _ = split_three_ways(frame, seed=0)
        result = sweep(
            fit, tune, grid=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0), tolerance=0.08, seed=0
        )

        assert result.chosen.cost_weight > 0.0
        assert result.chosen.cost < result.frontier[0].cost
        # And it stayed inside what was asked for, with the interval to prove it.
        assert result.chosen.quality_difference_interval[0] >= -0.08

    def test_too_little_data_produces_timidity_rather_than_confidence(self) -> None:
        # The rule this replaces -- "cheapest point not *measurably* worse" --
        # gets this exactly backwards: on a log too small to measure anything,
        # every point passes and it returns the most aggressive weight on the
        # grid. A non-inferiority margin fails a wide interval instead.
        _, _, frame = _problem_and_logs(n=24_000)
        fit, tune, _ = split_three_ways(frame, seed=0)
        thin = tune.head(400)

        generous = sweep(fit, thin, grid=(0.0, 1.0, 4.0), tolerance=0.08, seed=0)
        assert generous.chosen.cost_weight == 0.0
        widest = max(
            p.quality_difference_interval[1] - p.quality_difference_interval[0]
            for p in generous.frontier
        )
        assert widest > 0.16, widest

    def test_a_budget_is_honored_when_one_is_given(self) -> None:
        _, _, frame = _problem_and_logs()
        fit, tune, _ = split_three_ways(frame, seed=0)
        free = sweep(fit, tune, grid=(0.0,), seed=0).frontier[0].cost
        budget = free * 0.6

        result = sweep(
            fit, tune, grid=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0), budget=budget, seed=0
        )
        assert result.chosen.cost <= budget
        # And it is the best quality available inside that budget, not merely
        # some point inside it.
        affordable = [p for p in result.frontier if p.cost <= budget]
        assert result.chosen.quality == max(p.quality for p in affordable)

    def test_an_unreachable_budget_is_refused_rather_than_approximated(self) -> None:
        _, _, frame = _problem_and_logs(n=8_000)
        fit, tune, _ = split_three_ways(frame, seed=0)
        with pytest.raises(ValueError, match="budget"):
            sweep(fit, tune, grid=(0.0, 1.0), budget=1e-9, seed=0)


class TestItActuallySavesMoney:
    """The oracle check: the simulator knows the true cost and true quality."""

    def test_a_budget_buys_a_real_saving_and_the_quality_loss_is_honest(
        self,
    ) -> None:
        # The estimates come from the tuning log; the simulator knows the truth.
        # A frontier that looked good on the log and was wrong about what it
        # bought would be the failure this whole project exists to prevent.
        problem, _, frame = _problem_and_logs(n=30_000)
        fit, tune, _ = split_three_ways(frame, seed=0)
        blind_cost = sweep(fit, tune, grid=(0.0,), seed=0).frontier[0].cost

        result = sweep(
            fit,
            tune,
            grid=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
            budget=blind_cost * 0.5,
            seed=0,
        )
        blind = _true(problem, result.policy_at(0.0), result.epsilon)
        tuned = _true(problem, result.policy, result.epsilon)

        assert tuned["cost"] < 0.6 * blind["cost"], (tuned, blind)
        # The estimated cost saving must be within a few percent of the true one.
        assert result.chosen.saving == pytest.approx(
            blind["cost"] - tuned["cost"], rel=0.15
        ), (result.chosen.saving, blind["cost"] - tuned["cost"])
        # And the estimated quality loss must not understate the real one.
        estimated_loss = -result.chosen.quality_difference
        assert estimated_loss > 0.7 * (blind["quality"] - tuned["quality"])


def _true(problem, policy, epsilon: float) -> dict[str, float]:
    """The simulator's exact value and cost for a trained policy."""
    from sortition.decide.tree import discount_matrix
    from sortition.exploration import apply_epsilon_floor

    features = [
        {
            "n_tokens": float(500.0 * np.exp(0.5 * x[0])),
            "code_fraction": float(1.0 / (1.0 + np.exp(-x[1]))),
            "context_tokens": float(abs(x[2]) * 1000.0),
            "tools_required": bool(x[3] > 0.0),
        }
        for x in problem.contexts
    ]
    eligible = problem.eligible
    quality, cost = policy.score_matrix(features, eligible)
    scores = discount_matrix(quality, cost, eligible, policy.cost_weight)
    k = scores.shape[1]
    greedy = np.zeros_like(scores)
    greedy[np.arange(len(scores)), (k - 1) - scores[:, ::-1].argmax(axis=1)] = 1.0
    probs = apply_epsilon_floor(greedy, eligible, epsilon)
    return {
        "quality": float((probs * problem.q).sum(axis=1).mean()),
        "cost": float((probs * problem.cost).sum(axis=1).mean()),
    }
