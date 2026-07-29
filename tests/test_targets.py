"""The bridge between deploying a policy and evaluating one.

``TestRulesRoundTrip`` is the reason this file exists. ``test_decide.py`` already
asserts that a deployed ``ConstantPolicy`` evaluates as ``AlwaysArm``, but that
is a hand-written correspondence between two specific classes. This checks the
general claim: take a real rules policy, run traffic through the deployed
decision path, then ask the evaluator what that same policy would have done, and
get importance weights of exactly 1.

If those two ever disagree, every estimate about every policy acquires a bias
that nothing in the output would reveal -- the weights would still look
plausible.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import pytest

from sortition.decide import (
    ConstantPolicy,
    DecisionEngine,
    ExplorationConfig,
    ReloadingEngine,
    RulesPolicy,
    build,
    save,
)
from sortition.eval import importance_weights
from sortition.targets import PolicyTarget, parse_target

ARMS = ("cheap", "premium", "standard")  # sorted, as the arm universe would be

RULES = {
    "arms": list(ARMS),
    "default": ["standard", "cheap", "premium"],
    "rules": [
        {"label": "tools", "when": {"tools_required": True}, "exclude": ["cheap"]},
        {
            "label": "long",
            "when": {"context_tokens": {"gte": 32000}},
            "prefer": ["premium"],
        },
        {
            "label": "code",
            "when": {"code_fraction": {"gte": 0.5}},
            "prefer": ["premium"],
        },
    ],
}


def _features(rng: random.Random) -> dict[str, object]:
    """A request with enough variety to exercise every rule."""
    return {
        "tools_required": rng.random() < 0.3,
        "context_tokens": rng.choice([1_000, 8_000, 64_000]),
        "code_fraction": round(rng.random(), 3),
    }


class TestRulesRoundTrip:
    @pytest.fixture
    def artifact(self, tmp_path: Path) -> Path:
        path = tmp_path / "policy.json"
        save(build(RulesPolicy.from_dict(RULES), ExplorationConfig(epsilon=0.2)), path)
        return path

    def test_a_deployed_rules_policy_evaluates_as_itself(self, artifact: Path) -> None:
        engine = ReloadingEngine(artifact, poll_interval=3600.0)
        rng = random.Random(7)

        features, actions, propensities = [], [], []
        eligible = np.zeros((1_500, len(ARMS)), dtype=bool)
        for i in range(1_500):
            row = _features(rng)
            decision = engine.decide(features=row, eligible=list(ARMS))
            features.append(row)
            actions.append(ARMS.index(decision.chosen_arm))
            propensities.append(decision.propensity)
            # The logged eligible set is post-hard-filter, which is what a
            # gateway records and what the estimator later reads.
            for arm in decision.eligible_set:
                eligible[i, ARMS.index(arm)] = True

        target = parse_target(f"policy:{artifact}")
        probs = target.probabilities(features, eligible, ARMS)

        weights = importance_weights(
            np.array(actions, dtype=np.int64),
            np.array(propensities, dtype=np.float64),
            probs,
        )
        np.testing.assert_allclose(weights, 1.0, rtol=1e-12)

    def test_the_target_is_named_for_the_artifact(self, artifact: Path) -> None:
        # Reports identify policies by version; a target that forgot which
        # artifact it came from could not be attributed.
        from sortition.decide.artifact import load

        _, _, meta = load(artifact)
        assert parse_target(f"policy:{artifact}").name == meta.policy_version

    def test_epsilon_comes_from_the_artifact(self, artifact: Path) -> None:
        assert parse_target(f"policy:{artifact}").epsilon == 0.2

    def test_an_explicit_epsilon_overrides_the_artifact(self, artifact: Path) -> None:
        # Asking "what if we had explored less?" is a real question, and a
        # different sampling distribution than the one deployed.
        assert parse_target(f"policy:{artifact}+eps0.05").epsilon == 0.05

    def test_an_edited_artifact_is_rejected_here_too(self, artifact: Path) -> None:
        # The same guard as the decision path: a file edited without being
        # rebuilt would attribute traffic to a policy that never ran.
        artifact.write_text(
            artifact.read_text(encoding="utf-8").replace(
                '"epsilon": 0.2', '"epsilon": 0.9'
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="edited without being rebuilt"):
            parse_target(f"policy:{artifact}")


class TestSupport:
    def test_probabilities_stay_inside_the_logged_eligible_set(self) -> None:
        # The logged set is already post-hard-filter. Putting mass outside it
        # would claim a counterfactual the log cannot support.
        target = PolicyTarget(policy=ConstantPolicy("premium"), epsilon=0.1)
        eligible = np.array([[True, False, True], [True, True, True]])
        probs = target.probabilities([{}, {}], eligible, ARMS)

        assert probs[0, ARMS.index("premium")] == 0.0
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-12)
        assert (probs[~eligible] == 0).all()

    def test_hard_constraints_intersect_rather_than_replace(self) -> None:
        # The rule excludes `cheap`, so no mass should land there even though
        # the log says it was available.
        target = PolicyTarget(policy=RulesPolicy.from_dict(RULES), epsilon=0.0)
        eligible = np.ones((1, len(ARMS)), dtype=bool)
        probs = target.probabilities([{"tools_required": True}], eligible, ARMS)
        assert probs[0, ARMS.index("cheap")] == 0.0

    def test_a_filter_that_excludes_everything_falls_back(self) -> None:
        # Refusing to answer is worse than answering about what was available.
        blocking = RulesPolicy.from_dict(
            {"arms": list(ARMS), "rules": [{"when": {}, "exclude": list(ARMS)}]}
        )
        target = PolicyTarget(policy=blocking, epsilon=0.0)
        probs = target.probabilities([{}], np.ones((1, len(ARMS)), dtype=bool), ARMS)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-12)

    def test_rows_are_scored_independently(self) -> None:
        # A rules policy is context-dependent; if every row got the same answer
        # the features would not be reaching score().
        target = PolicyTarget(policy=RulesPolicy.from_dict(RULES), epsilon=0.0)
        eligible = np.ones((2, len(ARMS)), dtype=bool)
        probs = target.probabilities(
            [{"context_tokens": 64_000}, {"context_tokens": 100}], eligible, ARMS
        )
        assert probs[0].argmax() != probs[1].argmax()


class TestParsing:
    def test_rejects_an_empty_artifact_path(self) -> None:
        with pytest.raises(ValueError, match="names no artifact"):
            parse_target("policy:")

    def test_error_message_lists_the_policy_form(self) -> None:
        # The README points people at policy:, so the error has to mention it.
        with pytest.raises(ValueError, match=re.escape("policy:<artifact.json>")):
            parse_target("rules-v3")

    def test_missing_artifact_is_a_clear_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_target(f"policy:{tmp_path / 'absent.json'}")


class TestAgreesWithTheDecisionPath:
    @pytest.mark.parametrize("epsilon", [0.0, 0.05, 0.5, 1.0])
    def test_propensities_match_across_exploration_rates(self, epsilon: float) -> None:
        # The shared apply_epsilon_floor is what makes this hold. A second
        # implementation in PolicyTarget would break it silently.
        policy = RulesPolicy.from_dict(RULES)
        engine = DecisionEngine(
            policy=policy,
            exploration=ExplorationConfig(
                strategy="none" if epsilon == 0.0 else "epsilon_greedy", epsilon=epsilon
            ),
            rng=random.Random(3),
        )
        target = PolicyTarget(policy=policy, epsilon=epsilon)
        rng = random.Random(11)

        for _ in range(200):
            row = _features(rng)
            decision = engine.decide(features=row, eligible=list(ARMS))
            mask = np.zeros((1, len(ARMS)), dtype=bool)
            for arm in decision.eligible_set:
                mask[0, ARMS.index(arm)] = True
            probs = target.probabilities([row], mask, ARMS)[0]
            assert probs[ARMS.index(decision.chosen_arm)] == pytest.approx(
                decision.propensity, rel=1e-12
            )
