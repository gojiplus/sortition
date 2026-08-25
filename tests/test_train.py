"""Training a policy from logs.

The test that matters is ``test_beats_the_incumbent_on_held_out_logs``. Fitting a
model is easy; the question is whether the resulting policy is actually better,
measured somewhere it has not seen, and bounded by what is achievable at all.

The simulator makes that checkable: the best possible contextual policy has a
computable value, so an estimate exceeding it is a bug rather than a triumph.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from sortition.eval import evaluate
from sortition.features import infer_spec, matrix, vectorize
from sortition.sim import epsilon_greedy_policy, make_problem, sample_logs
from sortition.sim.to_frame import to_frame
from sortition.targets import PolicyTarget
from sortition.train import train, train_test_split


def _logs(*, n: int = 20_000, epsilon: float = 0.5, seed: int = 0) -> pl.DataFrame:
    problem = make_problem(n_contexts=800, n_arms=4, seed=seed)
    weights = np.random.default_rng(seed).standard_normal((6, 4))
    policy = epsilon_greedy_policy(weights, epsilon=epsilon)
    logs = sample_logs(problem, policy, n, seed=seed + 1)
    return to_frame(logs, problem, fallback_rate=0.0, seed=seed + 2)


class TestFeatureSpec:
    def test_only_keys_numeric_on_every_row_survive(self) -> None:
        # A key present on some rows would have to be imputed, and a silently
        # zero-filled feature is worse than a missing one.
        rows = [{"a": 1, "b": 2.0}, {"a": 3, "c": 4}]
        assert infer_spec(rows) == ("a",)

    def test_bools_count_as_numeric(self) -> None:
        assert "tools" in infer_spec([{"tools": True}, {"tools": False}])

    def test_strings_are_excluded(self) -> None:
        assert infer_spec([{"model": "x", "n": 1}, {"model": "y", "n": 2}]) == ("n",)

    def test_spec_order_is_stable(self) -> None:
        # The vector's meaning is positional; a wobbling order would silently
        # feed the model different features than it was trained on.
        rows = [{"z": 1, "a": 2, "m": 3}] * 3
        assert infer_spec(rows) == ("a", "m", "z")

    def test_vectorize_follows_the_spec_not_the_dict(self) -> None:
        assert vectorize({"b": 2.0, "a": 1.0}, ("a", "b")) == [1.0, 2.0]

    def test_missing_features_become_zero_rather_than_failing(self) -> None:
        # Refusing to route is worse than routing on an incomplete vector.
        assert vectorize({"a": 1.0}, ("a", "b")) == [1.0, 0.0]

    def test_matrix_matches_row_by_row(self) -> None:
        rows = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]
        spec = infer_spec(rows)
        np.testing.assert_allclose(
            matrix(rows, spec), np.array([vectorize(r, spec) for r in rows])
        )


class TestSplit:
    def test_partitions_without_overlap(self) -> None:
        logs = _logs(n=2_000)
        train_rows, held = train_test_split(logs, holdout=0.3, seed=0)
        assert train_rows.height + held.height == logs.height
        assert set(train_rows["request_id"]) & set(held["request_id"]) == set()

    def test_holdout_share_is_approximately_right(self) -> None:
        logs = _logs(n=5_000)
        _, held = train_test_split(logs, holdout=0.4, seed=0)
        assert held.height / logs.height == pytest.approx(0.4, abs=0.03)

    @pytest.mark.parametrize("holdout", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_degenerate_splits(self, holdout: float) -> None:
        with pytest.raises(ValueError, match=r"holdout must be in \(0, 1\)"):
            train_test_split(_logs(n=100), holdout=holdout)


class TestTraining:
    def test_refuses_too_few_rows(self) -> None:
        # Below a few hundred rows a boosted model fits noise, and saying so is
        # more useful than returning a policy nobody should deploy.
        with pytest.raises(ValueError, match="a boosted model fits noise"):
            train(_logs(n=200))

    def test_refuses_a_log_with_no_numeric_features(self) -> None:
        logs = _logs(n=2_000).with_columns(pl.lit("{}").alias("features"))
        with pytest.raises(ValueError, match="nothing to learn"):
            train(logs)

    def test_refuses_a_missing_metric(self) -> None:
        logs = _logs(n=2_000).drop("outcome")
        with pytest.raises(ValueError, match="no 'outcome' column"):
            train(logs)

    def test_produces_a_deployable_artifact(self, tmp_path: Path) -> None:
        # A trained policy has to go through the same artifact format as a rules
        # table, or it cannot be deployed or hot-reloaded like one.
        from sortition.decide import ReloadingEngine, save

        result = train(_logs(n=3_000), name="learned")
        assert result.artifact.kind == "tree"
        path = tmp_path / "learned.json"
        save(result.artifact, path)

        engine = ReloadingEngine(path, poll_interval=3600.0)
        decision = engine.decide(
            features={
                "n_tokens": 400.0,
                "code_fraction": 0.8,
                "context_tokens": 5_000.0,
                "tools_required": False,
            },
            eligible=list(result.arms),
        )
        assert decision.chosen_arm in result.arms
        assert decision.policy_version == result.artifact.policy_version

    def test_the_artifact_round_trips_through_json(self, tmp_path: Path) -> None:
        from sortition.decide.artifact import load, save

        result = train(_logs(n=3_000))
        path = tmp_path / "p.json"
        save(result.artifact, path)
        restored, _, _ = load(path)

        features = {
            "n_tokens": 300.0,
            "code_fraction": 0.2,
            "context_tokens": 1_000.0,
            "tools_required": True,
        }
        assert restored.score(features, result.arms) == pytest.approx(
            result.policy.score(features, result.arms)
        )

    def test_cost_weight_shifts_the_choice_toward_cheaper_arms(self) -> None:
        logs = _logs(n=4_000)
        # A large request. The penalty is the dollar gap over a fixed scale, so a
        # 300-token call is barely penalised however high the weight -- correctly,
        # since moving it saves almost nothing.
        features = {
            "n_tokens": 4_000.0,
            "code_fraction": 0.5,
            "context_tokens": 2_000.0,
            "tools_required": False,
        }

        free = train(logs, cost_weight=0.0).policy
        thrifty = train(logs, cost_weight=5.0).policy
        arms = free.arms

        costs = free.predict_cost(features, arms)
        cheapest = min(arms, key=lambda a: costs[a])
        best_free = max(arms, key=lambda a: free.score(features, arms)[a])
        best_thrifty = max(arms, key=lambda a: thrifty.score(features, arms)[a])
        # A large enough weight has to move the choice, or the cost term is inert.
        assert best_free != best_thrifty or best_thrifty == cheapest

    def test_predicted_cost_rises_with_the_size_of_the_request(self) -> None:
        # The defect this replaces: every request was charged the arm's global
        # mean, so a 20k-token call and a 200-token call were priced the same.
        policy = train(_logs(n=8_000)).policy
        arms = policy.arms
        base = {
            "code_fraction": 0.5,
            "context_tokens": 1_000.0,
            "tools_required": False,
        }
        small = policy.predict_cost({**base, "n_tokens": 250.0}, arms)
        large = policy.predict_cost({**base, "n_tokens": 2_000.0}, arms)
        for arm in arms:
            assert large[arm] > 2.0 * small[arm]

    def test_the_cost_model_beats_charging_every_request_the_arm_mean(self) -> None:
        # Measured against the simulator's noise-free expected cost, on rows the
        # model never saw, so this is out-of-sample accuracy and not a fit.
        problem = make_problem(n_contexts=800, n_arms=4, seed=0)
        weights = np.random.default_rng(0).standard_normal((6, 4))
        sampled = sample_logs(
            problem, epsilon_greedy_policy(weights, epsilon=0.5), 12_000, seed=1
        )
        frame = to_frame(sampled, problem, fallback_rate=0.0, seed=2)

        fit_on, held_out = np.arange(len(frame)) < 8_000, np.arange(len(frame)) >= 8_000
        policy = train(frame.filter(fit_on)).policy

        truth = problem.cost[sampled.context_idx[held_out], sampled.action[held_out]]
        arm_of = [problem.arms[a] for a in sampled.action[held_out]]
        rows = frame.filter(held_out).get_column("features").to_list()

        predicted = np.array(
            [
                policy.predict_cost(f, policy.arms)[a]
                for f, a in zip(rows, arm_of, strict=True)
            ]
        )
        per_arm_mean = (
            frame.filter(fit_on).group_by("chosen_arm").agg(pl.col("cost_usd").mean())
        )
        means = dict(zip(*per_arm_mean.to_dict(as_series=False).values(), strict=True))
        constant = np.array([means[a] for a in arm_of])

        learned_mae = float(np.abs(predicted - truth).mean())
        constant_mae = float(np.abs(constant - truth).mean())
        assert learned_mae < 0.6 * constant_mae, (learned_mae, constant_mae)

    def test_the_cost_gap_between_arms_widens_with_request_size(self) -> None:
        # This is what a per-arm constant cannot express, and it is the whole
        # reason the trade-off is request-conditional: on a short request the
        # premium arm is barely dearer, so it is worth taking.
        policy = train(_logs(n=8_000)).policy
        arms = policy.arms
        base = {
            "code_fraction": 0.5,
            "context_tokens": 1_000.0,
            "tools_required": False,
        }
        small = policy.predict_cost({**base, "n_tokens": 250.0}, arms)
        large = policy.predict_cost({**base, "n_tokens": 2_000.0}, arms)
        assert (max(large.values()) - min(large.values())) > 2.0 * (
            max(small.values()) - min(small.values())
        )

    def test_the_penalty_scales_with_the_request_and_does_not_normalize_away(
        self,
    ) -> None:
        # The whole point of pricing per request. Normalizing cost within each
        # request's eligible set cancels exactly this: with cost = size * price,
        # both the numerator and the spread scale by size, so every request gets
        # the same penalties and the dearest arm is always charged the maximum.
        # The scale has to be fixed at training time, not per request.
        policy = train(_logs(n=8_000), cost_weight=1.0).policy
        arms = policy.arms
        base = {
            "code_fraction": 0.5,
            "context_tokens": 1_000.0,
            "tools_required": False,
        }

        def penalty(tokens: float) -> dict[str, float]:
            features = {**base, "n_tokens": tokens}
            quality = policy.predict(features, arms)
            scored = policy.score(features, arms)
            return {arm: quality[arm] - scored[arm] for arm in arms}

        small, large = penalty(500.0), penalty(5_000.0)
        dearest = max(
            arms,
            key=lambda a: policy.predict_cost({**base, "n_tokens": 500.0}, arms)[a],
        )
        # It grows with the request rather than being pinned. How much depends on
        # the booster, which compresses predicted cost near the edges of what it
        # saw; the assertion below is the one about this policy's arithmetic.
        assert large[dearest] > 1.5 * small[dearest], (small[dearest], large[dearest])

        # The penalty is exactly the dollar gap over the fixed scale, so the two
        # ratios agree. Per-set normalization would make the left side 1.0 for
        # any pair of requests, which is the defect this guards.
        def gap(tokens: float) -> float:
            costs = policy.predict_cost({**base, "n_tokens": tokens}, arms)
            return costs[dearest] - min(costs.values())

        assert large[dearest] / small[dearest] == pytest.approx(
            gap(5_000.0) / gap(500.0), rel=1e-6
        )

    def test_a_cost_weight_with_no_cost_column_is_refused(self) -> None:
        # Silently ignoring the weight would ship a policy that says it prices
        # cost and does not.
        logs = _logs(n=3_000).drop("cost_usd")
        with pytest.raises(ValueError, match="cost_usd"):
            train(logs, cost_weight=1.0)

    def test_propensity_weighting_is_recorded_as_a_choice(self) -> None:
        # Both paths must produce a usable policy; the weighting changes what is
        # learned, not whether anything is.
        logs = _logs(n=3_000)
        for weighted in (True, False):
            result = train(logs, weight_by_propensity=weighted)
            assert result.n_rows > 0


class TestBeatsTheIncumbent:
    """The only question that decides whether training was worth doing."""

    @pytest.fixture(scope="class")
    def trained(self) -> tuple[pl.DataFrame, object, float]:
        problem = make_problem(n_contexts=800, n_arms=4, seed=0)
        weights = np.random.default_rng(0).standard_normal((6, 4))
        # A deliberately mediocre incumbent, so there is headroom to capture.
        logs = sample_logs(
            problem, epsilon_greedy_policy(weights, epsilon=0.5), 40_000, seed=1
        )
        frame = to_frame(logs, problem, fallback_rate=0.0, seed=2)
        train_rows, held_out = train_test_split(frame, holdout=0.4, seed=7)
        result = train(train_rows, metric="outcome", epsilon=0.05, name="learned")
        # The best any policy could achieve: argmax q per context.
        oracle = float(problem.q.max(axis=1).mean())
        return held_out, result.policy, oracle

    def test_beats_the_incumbent_on_held_out_logs(
        self, trained: tuple[pl.DataFrame, object, float]
    ) -> None:
        held_out, policy, _ = trained
        target = PolicyTarget(policy=policy, epsilon=0.05, name="learned")
        estimate = evaluate(held_out, target, metric="outcome", estimator="dr")
        observed = float(held_out["outcome"].to_numpy().mean())

        assert estimate.trustworthy
        assert estimate.interval is not None
        # Better than what actually happened, and the interval says so rather
        # than the point estimate alone.
        assert estimate.interval.low > observed

    def test_does_not_exceed_what_is_achievable(
        self, trained: tuple[pl.DataFrame, object, float]
    ) -> None:
        # An estimate above the oracle is a bug, not a triumph. This is the
        # check a real log could never give you.
        held_out, policy, oracle = trained
        target = PolicyTarget(policy=policy, epsilon=0.05, name="learned")
        estimate = evaluate(held_out, target, metric="outcome", estimator="dr")
        assert estimate.value <= oracle

    def test_beats_every_constant_policy(
        self, trained: tuple[pl.DataFrame, object, float]
    ) -> None:
        # The point of a contextual policy: no single arm should match it.
        held_out, policy, _ = trained
        learned = evaluate(
            held_out, PolicyTarget(policy=policy, epsilon=0.05), metric="outcome"
        ).value
        arms = sorted(set(held_out["chosen_arm"].to_list()))
        best_arm = max(
            evaluate(held_out, f"always:{arm}", metric="outcome").value for arm in arms
        )
        assert learned > best_arm
