"""The decision path, and its agreement with the evaluation path.

The last class here is the important one. Everything else checks that decisions
are well-formed; ``TestRoundTrip`` checks that a policy which was *deployed*
evaluates as itself when it is later used as a *target*. If those two code paths
ever compute exploration probabilities differently, every estimate acquires a
bias that nothing in the output would reveal.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import pytest

from sortition.decide import (
    Beta,
    ConstantPolicy,
    CostAwarePolicy,
    DecisionEngine,
    ExplorationConfig,
    RulesPolicy,
    propensities,
    sample_with_propensity,
)
from sortition.eval import importance_weights
from sortition.exploration import epsilon_greedy_propensity
from sortition.targets import AlwaysArm, EpsilonFloor

ARMS = ("cheap", "standard", "premium")


def _engine(
    epsilon: float = 0.1, seed: int = 0, arm: str = "standard"
) -> DecisionEngine:
    return DecisionEngine(
        policy=ConstantPolicy(arm),
        exploration=ExplorationConfig(epsilon=epsilon),
        rng=random.Random(seed),
    )


class TestDecision:
    def test_returns_a_valid_decision(self) -> None:
        decision = _engine().decide(features={}, eligible=list(ARMS))
        assert decision.chosen_arm in ARMS
        assert decision.chosen_arm in decision.eligible_set
        assert decision.propensity > 0.0
        # Fallbacks are instructions, not draws, so they must not overlap the
        # arm that was actually sampled.
        assert decision.chosen_arm not in decision.fallback_chain

    def test_fallback_chain_covers_the_rest_of_the_eligible_set(self) -> None:
        decision = _engine().decide(features={}, eligible=list(ARMS))
        assert set(decision.fallback_chain) == set(ARMS) - {decision.chosen_arm}

    def test_empty_eligible_set_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no eligible arms"):
            _engine().decide(features={}, eligible=[])

    def test_duplicate_arms_are_collapsed(self) -> None:
        decision = _engine().decide(
            features={}, eligible=["cheap", "cheap", "standard"]
        )
        assert decision.eligible_set == ("cheap", "standard")

    def test_single_arm_is_deterministic(self) -> None:
        # With one option there is no choice to randomize, so the propensity is
        # exactly 1 rather than an epsilon-adjusted fiction.
        decision = _engine().decide(features={}, eligible=["cheap"])
        assert decision.propensity == 1.0
        assert decision.explore is False

    def test_policy_version_records_the_exploration_rate(self) -> None:
        # Two deployments that differ only in epsilon are different policies and
        # their logs must not be pooled.
        assert _engine(epsilon=0.05).policy_version == "always:standard+eps0.05"
        assert _engine(epsilon=0.2).policy_version == "always:standard+eps0.2"

    def test_decision_ids_are_unique(self) -> None:
        engine = _engine()
        ids = {
            engine.decide(features={}, eligible=list(ARMS)).decision_id
            for _ in range(50)
        }
        assert len(ids) == 50


class TestPropensities:
    def test_greedy_propensity_includes_the_exploration_share(self) -> None:
        # The classic epsilon-greedy bug is logging 1-eps for the greedy arm.
        # It can also be drawn by the uniform exploration branch.
        assert epsilon_greedy_propensity(n_eligible=4, epsilon=0.2, is_greedy=True) == (
            pytest.approx(0.8 + 0.2 / 4)
        )
        assert epsilon_greedy_propensity(
            n_eligible=4, epsilon=0.2, is_greedy=False
        ) == (pytest.approx(0.2 / 4))

    def test_propensities_sum_to_one_over_the_eligible_set(self) -> None:
        n, epsilon = 5, 0.3
        total = epsilon_greedy_propensity(
            n_eligible=n, epsilon=epsilon, is_greedy=True
        ) + (n - 1) * epsilon_greedy_propensity(
            n_eligible=n, epsilon=epsilon, is_greedy=False
        )
        assert total == pytest.approx(1.0)

    def test_logged_propensity_matches_realized_frequency(self) -> None:
        # The propensity is a claim about how often this arm gets chosen. Check
        # the claim against what actually happens.
        engine = _engine(epsilon=0.4, seed=7)
        counts: dict[str, int] = dict.fromkeys(ARMS, 0)
        logged: dict[str, float] = {}
        for _ in range(20_000):
            decision = engine.decide(features={}, eligible=list(ARMS))
            counts[decision.chosen_arm] += 1
            logged[decision.chosen_arm] = decision.propensity
        for arm, count in counts.items():
            assert count / 20_000 == pytest.approx(logged[arm], abs=0.02)

    def test_no_exploration_yields_a_degenerate_propensity(self) -> None:
        engine = DecisionEngine(
            policy=ConstantPolicy("premium"),
            exploration=ExplorationConfig(strategy="none"),
        )
        decision = engine.decide(features={}, eligible=list(ARMS))
        assert decision.propensity == 1.0
        assert decision.chosen_arm == "premium"

    def test_unexplored_policy_warns_loudly(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="sortition.decide.engine"):
            DecisionEngine(
                policy=ConstantPolicy("premium"),
                exploration=ExplorationConfig(strategy="none"),
            )
        assert "support no counterfactual claim" in caplog.text


class TestRulesPolicy:
    @pytest.fixture
    def policy(self) -> RulesPolicy:
        return RulesPolicy.from_dict(
            {
                "arms": list(ARMS),
                "default": ["standard", "cheap", "premium"],
                "rules": [
                    {
                        "when": {"tools_required": True},
                        "exclude": ["cheap"],
                        "label": "tools_required",
                    },
                    {
                        "when": {"context_tokens": {"gte": 32000}},
                        "prefer": ["premium"],
                        "label": "long_context",
                    },
                    {
                        "when": {"code_fraction": {"gte": 0.5}},
                        "prefer": ["premium", "standard"],
                    },
                ],
            }
        )

    def test_default_order_applies_when_no_rule_matches(
        self, policy: RulesPolicy
    ) -> None:
        scores = policy.score({}, ARMS)
        assert max(scores, key=lambda arm: scores[arm]) == "standard"

    def test_first_matching_rule_wins(self, policy: RulesPolicy) -> None:
        scores = policy.score({"context_tokens": 64_000, "code_fraction": 0.9}, ARMS)
        assert max(scores, key=lambda arm: scores[arm]) == "premium"

    def test_hard_constraints_shrink_the_eligible_set(
        self, policy: RulesPolicy
    ) -> None:
        survivors, applied = policy.hard_filter({"tools_required": True}, ARMS)
        assert "cheap" not in survivors
        assert applied == ("tools_required",)

    def test_engine_records_applied_constraints(self, policy: RulesPolicy) -> None:
        engine = DecisionEngine(policy=policy, rng=random.Random(1))
        decision = engine.decide(features={"tools_required": True}, eligible=list(ARMS))
        assert "cheap" not in decision.eligible_set
        assert decision.hard_constraints_applied == ("tools_required",)

    def test_constraints_that_would_empty_the_set_are_ignored(self) -> None:
        # Refusing to route is worse than routing to a disfavoured arm.
        policy = RulesPolicy.from_dict(
            {
                "arms": ["a", "b"],
                "rules": [{"when": {}, "exclude": ["a", "b"], "label": "all"}],
            }
        )
        survivors, applied = policy.hard_filter({}, ("a", "b"))
        assert survivors == ("a", "b")
        assert applied == ()

    def test_arms_no_rule_mentions_keep_positive_support(
        self, policy: RulesPolicy
    ) -> None:
        # Dropping them from the score would silently shrink the eligible set
        # and make their counterfactual unanswerable.
        scores = policy.score({"context_tokens": 64_000}, ARMS)
        assert set(scores) == set(ARMS)

    def test_rejects_rules_naming_unknown_arms(self) -> None:
        with pytest.raises(ValueError, match="not in 'arms'"):
            RulesPolicy.from_dict({"arms": ["a"], "rules": [{"prefer": ["b"]}]})

    def test_rejects_an_empty_arm_list(self) -> None:
        with pytest.raises(ValueError, match="non-empty 'arms'"):
            RulesPolicy.from_dict({"arms": []})

    def test_mistyped_feature_is_a_non_match_not_a_crash(
        self, policy: RulesPolicy
    ) -> None:
        # One caller sending a string where a number belongs must not break
        # routing for everyone.
        scores = policy.score({"context_tokens": "lots"}, ARMS)
        assert max(scores, key=lambda arm: scores[arm]) == "standard"

    def test_loads_from_yaml(self, tmp_path: object) -> None:
        from pathlib import Path

        assert isinstance(tmp_path, Path)
        path = tmp_path / "rules.yaml"
        path.write_text(
            "arms: [cheap, premium]\n"
            "default: [cheap, premium]\n"
            "rules:\n"
            "  - when: {tools_required: true}\n"
            "    prefer: [premium]\n",
            encoding="utf-8",
        )
        policy = RulesPolicy.from_yaml(path)
        scores = policy.score({"tools_required": True}, ("cheap", "premium"))
        assert max(scores, key=lambda arm: scores[arm]) == "premium"


class TestCostAwarePolicy:
    def test_cost_weight_zero_picks_the_best_arm(self) -> None:
        policy = CostAwarePolicy(
            quality={"cheap": 0.4, "premium": 0.9},
            cost_usd={"cheap": 0.001, "premium": 0.05},
            cost_weight=0.0,
        )
        scores = policy.score({}, ("cheap", "premium"))
        assert max(scores, key=lambda arm: scores[arm]) == "premium"

    def test_a_high_cost_weight_flips_the_choice(self) -> None:
        policy = CostAwarePolicy(
            quality={"cheap": 0.4, "premium": 0.9},
            cost_usd={"cheap": 0.001, "premium": 0.05},
            cost_weight=1.0,
        )
        scores = policy.score({}, ("cheap", "premium"))
        assert max(scores, key=lambda arm: scores[arm]) == "cheap"


class TestThompson:
    def test_estimated_propensity_matches_selection_frequency(self) -> None:
        # The claim the upstream LiteLLM patch would be making. If this does not
        # hold, logging the number is worse than logging nothing.
        posteriors = {
            "cheap": Beta(6.0, 4.0),
            "standard": Beta(7.0, 3.0),
            "premium": Beta(8.0, 2.0),
        }
        rng = random.Random(3)
        estimated = propensities(posteriors, n_samples=40_000, rng=rng)

        counts: dict[str, int] = dict.fromkeys(posteriors, 0)
        for _ in range(20_000):
            arm, _ = sample_with_propensity(posteriors, n_samples=0, rng=rng)
            counts[arm] += 1
        for arm, count in counts.items():
            assert count / 20_000 == pytest.approx(estimated[arm], abs=0.02)

    def test_propensity_is_never_zero(self) -> None:
        # A zero would be self-contradictory -- the arm was demonstrably chosen
        # -- and an infinite importance weight downstream.
        posteriors = {"a": Beta(1.0, 30.0), "b": Beta(30.0, 1.0)}
        rng = random.Random(5)
        for _ in range(300):
            _, propensity = sample_with_propensity(posteriors, n_samples=16, rng=rng)
            assert propensity >= 1.0 / 17

    def test_propensities_sum_to_one(self) -> None:
        posteriors = {"a": Beta(2.0, 3.0), "b": Beta(3.0, 2.0), "c": Beta(1.0, 1.0)}
        estimated = propensities(posteriors, n_samples=2_000, rng=random.Random(0))
        assert sum(estimated.values()) == pytest.approx(1.0)

    def test_is_reproducible_under_a_seeded_rng(self) -> None:
        posteriors = {"a": Beta(2.0, 3.0), "b": Beta(3.0, 2.0)}
        first = sample_with_propensity(posteriors, n_samples=32, rng=random.Random(11))
        second = sample_with_propensity(posteriors, n_samples=32, rng=random.Random(11))
        assert first == second

    def test_single_arm_is_certain(self) -> None:
        arm, propensity = sample_with_propensity(
            {"only": Beta(1.0, 1.0)}, rng=random.Random(0)
        )
        assert (arm, propensity) == ("only", 1.0)

    def test_cost_scores_shift_the_distribution(self) -> None:
        posteriors = {"cheap": Beta(5.0, 5.0), "premium": Beta(5.0, 5.0)}
        rng = random.Random(2)
        neutral = propensities(posteriors, n_samples=4_000, rng=rng)
        tilted = propensities(
            posteriors, scores={"cheap": 0.5}, n_samples=4_000, rng=rng
        )
        assert neutral["cheap"] == pytest.approx(0.5, abs=0.03)
        assert tilted["cheap"] > 0.85

    def test_rejects_no_arms(self) -> None:
        with pytest.raises(ValueError, match="no arms"):
            sample_with_propensity({})


class TestRoundTrip:
    """A deployed policy must evaluate as itself.

    This is the test that catches drift between the decision path and the
    evaluation path. If the propensity logged at decision time and the
    probability computed at evaluation time disagree, importance weights stop
    being 1 and every estimate is quietly biased.
    """

    def test_importance_weights_are_exactly_one(self) -> None:
        epsilon, arm = 0.15, "standard"
        engine = _engine(epsilon=epsilon, seed=42, arm=arm)

        decisions = [
            engine.decide(features={}, eligible=list(ARMS)) for _ in range(2_000)
        ]
        arms = tuple(sorted(ARMS))
        index = {name: i for i, name in enumerate(arms)}

        action = np.array([index[d.chosen_arm] for d in decisions], dtype=np.int64)
        propensity = np.array([d.propensity for d in decisions], dtype=np.float64)
        eligible = np.ones((len(decisions), len(arms)), dtype=bool)

        # The same policy, expressed on the evaluation side.
        target = EpsilonFloor(AlwaysArm(arm), epsilon)
        target_probs = target.probabilities([{}] * len(decisions), eligible, arms)

        weights = importance_weights(action, propensity, target_probs)
        np.testing.assert_allclose(weights, 1.0, rtol=1e-12)

    def test_the_two_paths_agree_arm_by_arm(self) -> None:
        epsilon, arm = 0.25, "premium"
        arms = tuple(sorted(ARMS))
        eligible = np.ones((1, len(arms)), dtype=bool)
        target_probs = EpsilonFloor(AlwaysArm(arm), epsilon).probabilities(
            [{}], eligible, arms
        )[0]

        for i, name in enumerate(arms):
            deployed = epsilon_greedy_propensity(
                n_eligible=len(arms), epsilon=epsilon, is_greedy=name == arm
            )
            assert deployed == pytest.approx(float(target_probs[i]), rel=1e-12)
