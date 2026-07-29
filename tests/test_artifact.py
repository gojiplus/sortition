"""Versioned policies, hot reload, and concurrency on the decision path."""

from __future__ import annotations

import random
import threading
from collections import Counter
from pathlib import Path

import pytest

from sortition.decide import (
    ConstantPolicy,
    CostAwarePolicy,
    DecisionEngine,
    ExplorationConfig,
    ReloadingEngine,
    RulesPolicy,
    WeightedPolicy,
    build,
    load,
    save,
)
from sortition.decide.artifact import content_hash

ARMS = ("cheap", "standard", "premium")


def _rules() -> RulesPolicy:
    return RulesPolicy.from_dict(
        {
            "arms": list(ARMS),
            "default": ["standard", "cheap", "premium"],
            "rules": [
                {
                    "when": {"tools_required": True},
                    "exclude": ["cheap"],
                    "label": "tools",
                },
                {"when": {"context_tokens": {"gte": 32000}}, "prefer": ["premium"]},
            ],
        }
    )


class TestVersioning:
    def test_version_is_derived_from_content(self, tmp_path: Path) -> None:
        # Two deployments must not be able to disagree about what a version
        # means, so the same policy always produces the same string.
        first = build(_rules(), ExplorationConfig(epsilon=0.05))
        second = build(_rules(), ExplorationConfig(epsilon=0.05))
        assert first.policy_version == second.policy_version

    def test_changing_a_rule_changes_the_version(self) -> None:
        base = build(_rules(), ExplorationConfig(epsilon=0.05))
        edited = RulesPolicy.from_dict(
            {"arms": list(ARMS), "default": ["premium", "standard", "cheap"]}
        )
        assert build(edited, ExplorationConfig(epsilon=0.05)).policy_version != (
            base.policy_version
        )

    def test_changing_only_epsilon_changes_the_version(self) -> None:
        # Same preferences, different sampling distribution. Pooling their logs
        # into one estimate would be wrong, so they must not share a version.
        low = build(_rules(), ExplorationConfig(epsilon=0.05))
        high = build(_rules(), ExplorationConfig(epsilon=0.30))
        assert low.policy_version != high.policy_version

    @pytest.mark.parametrize(
        "policy",
        [
            ConstantPolicy("premium"),
            WeightedPolicy(weights={"cheap": 1.0, "premium": 2.0}),
            CostAwarePolicy(
                quality={"cheap": 0.4, "premium": 0.9},
                cost_usd={"cheap": 0.001, "premium": 0.05},
                cost_weight=0.5,
            ),
        ],
    )
    def test_every_policy_kind_round_trips(
        self, policy: object, tmp_path: Path
    ) -> None:
        path = tmp_path / "policy.json"
        save(build(policy, ExplorationConfig(epsilon=0.1)), path)  # type: ignore[arg-type]
        restored, exploration, _ = load(path)
        assert restored == policy
        assert exploration.epsilon == 0.1

    def test_rules_round_trip_preserves_behaviour(self, tmp_path: Path) -> None:
        path = tmp_path / "policy.json"
        original = _rules()
        save(build(original, ExplorationConfig(epsilon=0.1)), path)
        restored, _, _ = load(path)
        features = {"tools_required": True, "context_tokens": 64_000}
        assert restored.score(features, ARMS) == original.score(features, ARMS)
        assert restored.hard_filter(features, ARMS) == original.hard_filter(
            features, ARMS
        )

    def test_an_edited_file_is_rejected(self, tmp_path: Path) -> None:
        # Editing the JSON without rebuilding would attribute traffic to a
        # policy that never ran.
        path = tmp_path / "policy.json"
        save(build(_rules()), path)
        text = path.read_text(encoding="utf-8").replace(
            '"epsilon": 0.05', '"epsilon": 0.5'
        )
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="edited without being rebuilt"):
            load(path)

    def test_unknown_policy_types_fail_loudly(self) -> None:
        class Homemade:
            name = "homemade"

            def score(self, features: dict, eligible: tuple) -> dict:
                return {}

        with pytest.raises(TypeError, match="no artifact representation"):
            build(Homemade())  # type: ignore[arg-type]

    def test_hash_ignores_key_order(self) -> None:
        left = content_hash("rules", {"a": 1, "b": 2}, {"strategy": "none"})
        right = content_hash("rules", {"b": 2, "a": 1}, {"strategy": "none"})
        assert left == right


class TestHotReload:
    def test_serves_the_artifact_it_was_given(self, tmp_path: Path) -> None:
        path = tmp_path / "policy.json"
        artifact = build(ConstantPolicy("premium"), ExplorationConfig(epsilon=0.0))
        save(artifact, path)

        engine = ReloadingEngine(path, poll_interval=0.0)
        decision = engine.decide(features={}, eligible=list(ARMS))
        assert decision.chosen_arm == "premium"
        assert decision.policy_version == artifact.policy_version

    def test_picks_up_a_new_policy_without_a_restart(self, tmp_path: Path) -> None:
        path = tmp_path / "policy.json"
        save(build(ConstantPolicy("cheap"), ExplorationConfig(epsilon=0.0)), path)
        engine = ReloadingEngine(path, poll_interval=0.0)
        assert engine.decide(features={}, eligible=list(ARMS)).chosen_arm == "cheap"

        save(build(ConstantPolicy("premium"), ExplorationConfig(epsilon=0.0)), path)
        after = engine.decide(features={}, eligible=list(ARMS))
        assert after.chosen_arm == "premium"
        assert engine.reloads == 1
        # The change is visible in the log, so the two policies' rows can never
        # be pooled into one estimate by accident.
        assert after.policy_version == engine.policy_version

    def test_a_corrupt_artifact_keeps_the_old_policy_serving(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "policy.json"
        save(build(ConstantPolicy("premium"), ExplorationConfig(epsilon=0.0)), path)
        engine = ReloadingEngine(path, poll_interval=0.0)
        good = engine.policy_version

        path.write_text("{ not json", encoding="utf-8")
        decision = engine.decide(features={}, eligible=list(ARMS))

        # Never fall open to a default: that would route real traffic under a
        # policy nobody chose.
        assert decision.chosen_arm == "premium"
        assert engine.policy_version == good
        assert engine.reload_errors == 1

    def test_a_broken_artifact_is_not_retried_every_request(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "policy.json"
        save(build(ConstantPolicy("premium"), ExplorationConfig(epsilon=0.0)), path)
        engine = ReloadingEngine(path, poll_interval=0.0)
        path.write_text("{ not json", encoding="utf-8")
        for _ in range(5):
            engine.decide(features={}, eligible=list(ARMS))
        assert engine.reload_errors == 1

    def test_poll_interval_bounds_the_stat_calls(self, tmp_path: Path) -> None:
        path = tmp_path / "policy.json"
        save(build(ConstantPolicy("cheap"), ExplorationConfig(epsilon=0.0)), path)
        engine = ReloadingEngine(path, poll_interval=3600.0)
        save(build(ConstantPolicy("premium"), ExplorationConfig(epsilon=0.0)), path)
        # Within the interval the change is not picked up, which is the point:
        # the hot path costs one clock comparison, not a stat per request.
        assert engine.decide(features={}, eligible=list(ARMS)).chosen_arm == "cheap"
        assert engine.reloads == 0

    def test_a_missing_artifact_fails_at_startup(self, tmp_path: Path) -> None:
        # Starting with no policy is worse than not starting.
        with pytest.raises((OSError, ValueError)):
            ReloadingEngine(tmp_path / "absent.json")


class TestConcurrency:
    def test_propensities_hold_up_under_threads(self) -> None:
        # random.Random is not thread-safe. Interleaved draws would not corrupt
        # anything visibly, they would just stop matching the logged
        # propensities -- which is the failure this project exists to prevent.
        engine = DecisionEngine(
            policy=ConstantPolicy("standard"),
            exploration=ExplorationConfig(epsilon=0.4),
            rng=random.Random(0),
        )
        counts: Counter[str] = Counter()
        logged: dict[str, float] = {}
        lock = threading.Lock()

        def worker() -> None:
            local: Counter[str] = Counter()
            seen: dict[str, float] = {}
            for _ in range(2_000):
                decision = engine.decide(features={}, eligible=list(ARMS))
                local[decision.chosen_arm] += 1
                seen[decision.chosen_arm] = decision.propensity
            with lock:
                counts.update(local)
                logged.update(seen)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        total = sum(counts.values())
        assert total == 16_000
        for arm, count in counts.items():
            assert count / total == pytest.approx(logged[arm], abs=0.02)

    def test_reload_under_concurrent_decisions(self, tmp_path: Path) -> None:
        path = tmp_path / "policy.json"
        save(build(ConstantPolicy("cheap"), ExplorationConfig(epsilon=0.0)), path)
        engine = ReloadingEngine(path, poll_interval=0.0)
        seen: list[str] = []
        lock = threading.Lock()

        def worker() -> None:
            local = [
                engine.decide(features={}, eligible=list(ARMS)).chosen_arm
                for _ in range(500)
            ]
            with lock:
                seen.extend(local)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        save(build(ConstantPolicy("premium"), ExplorationConfig(epsilon=0.0)), path)
        for thread in threads:
            thread.join()

        # Every request sees one policy or the other, never a mixture of the
        # two, and none of them fail.
        assert set(seen) <= {"cheap", "premium"}
        assert len(seen) == 2_000
