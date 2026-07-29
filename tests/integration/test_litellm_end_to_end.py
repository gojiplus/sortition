"""End to end through a real LiteLLM Router.

Nothing is mocked out of sortition here. A genuine ``litellm.Router`` runs the
plugin pipeline, executes against ``mock_response`` deployments (no network, no
keys, the idiom LiteLLM's own routing tests use), and fires the real callback
machinery. What is asserted is that a decision made at routing time survives all
the way into a log that the estimators can read.

The first test is the load-bearing one. LiteLLM's ``get_standard_logging_metadata``
filters metadata to the keys declared on ``StandardLoggingMetadata``, which
strips a plugin's signals from ``standard_logging_object``. The logger therefore
reads raw kwargs instead. If that path ever closes, this fails and the logger
needs redesigning -- so it is checked directly rather than inferred from the
tests above it.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

litellm = pytest.importorskip("litellm", reason="needs the [litellm] extra")

from sortition.decide import (  # noqa: E402
    ConstantPolicy,
    DecisionEngine,
    ExplorationConfig,
)
from sortition.eval import doctor, evaluate, importance_weights  # noqa: E402
from sortition.frame import to_arrays  # noqa: E402
from sortition.integrations import (  # noqa: E402
    SortitionLogger,
    SortitionPlugin,
    read_decision,
)
from sortition.store import LogStore  # noqa: E402
from sortition.targets import AlwaysArm, EpsilonFloor  # noqa: E402

CHEAP = "openai/gpt-4o-mini"
MID = "openai/gpt-4.1"
PREMIUM = "openai/gpt-5.1"
ARMS = (CHEAP, MID, PREMIUM)

MESSAGES = [{"role": "user", "content": "explain this function"}]


def _model_list(broken: str | None = None) -> list[dict[str, Any]]:
    """Three deployments in one model group, optionally with one that fails."""
    entries = []
    for name in ARMS:
        params: dict[str, Any] = {"model": name, "api_key": "sk-test"}
        if name == broken:
            params["mock_response"] = Exception("simulated provider outage")
        else:
            params["mock_response"] = f"served by {name}"
        entries.append({"model_name": "smart", "litellm_params": params})
    return entries


def _engine(epsilon: float = 0.2, seed: int = 0, arm: str = MID) -> DecisionEngine:
    return DecisionEngine(
        policy=ConstantPolicy(arm),
        exploration=ExplorationConfig(epsilon=epsilon),
        rng=random.Random(seed),
    )


async def _drain() -> None:
    """Wait for LiteLLM's background logging worker to finish.

    Callbacks are enqueued on a worker whose queue is bound to the running event
    loop. Sleeping is a race; ``flush()`` awaits the queue's unfinished-task
    counter, so it also covers callbacks that have been dequeued but not yet
    completed.
    """
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    # Bounded: flush() awaits queue.join(), which never returns if the worker
    # task is not consuming in this loop. A hang here would be a test-harness
    # failure masquerading as a product bug.
    with contextlib.suppress(TimeoutError):  # only on a stalled worker
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10.0)
    # Yield once more so any callback dequeued but still running can finish.
    await asyncio.sleep(0.05)


def _reset_callbacks() -> None:
    """Clear every LiteLLM callback list, not just ``litellm.callbacks``.

    Assigning to ``litellm.callbacks`` leaves the internal callback manager
    holding the previous test's logger instances, which are bound to an event
    loop that no longer exists. The symptom is a callback that fires alone but
    silently stops firing once another test has run before it.
    """
    litellm.logging_callback_manager._reset_all_callbacks()
    litellm.callbacks = []


@pytest.fixture(autouse=True)
def _clean_callbacks() -> Any:
    """Isolate LiteLLM's global callback state between tests."""
    _reset_callbacks()
    yield
    _reset_callbacks()


class TestSignalPath:
    """The assumption the logger is built on, checked directly."""

    def test_plugin_signals_reach_the_callback(self) -> None:
        captured: list[dict[str, Any]] = []

        # Must subclass CustomLogger: litellm dispatches with isinstance, so a
        # duck-typed object is silently never called.
        from litellm.integrations.custom_logger import CustomLogger

        class Probe(CustomLogger):
            async def async_log_success_event(
                self,
                kwargs: dict[str, Any],
                response_obj: Any,
                start_time: Any,
                end_time: Any,
            ) -> None:
                captured.append(kwargs)

        litellm.callbacks = [Probe()]
        router = litellm.Router(
            model_list=_model_list(), plugins=[SortitionPlugin(_engine())]
        )

        async def run() -> None:
            await router.acompletion(model="smart", messages=MESSAGES)
            await _drain()

        asyncio.run(run())
        assert captured, "the success callback never fired"

        decision = read_decision(captured[0])
        assert decision is not None, (
            "sortition's signals did not survive into the callback kwargs; "
            "the logger reads litellm_params.metadata.routing_plugin_signals"
        )
        assert 0.0 < decision["propensity"] <= 1.0
        assert decision["chosen_arm"] in ARMS

        # And confirm the reason the raw-kwargs path is needed at all: the
        # standard payload really does strip unknown metadata keys.
        standard = captured[0].get("standard_logging_object") or {}
        assert "routing_plugin_signals" not in (standard.get("metadata") or {})


class TestRouting:
    def test_plugin_narrows_to_the_sampled_arm(self) -> None:
        plugin = SortitionPlugin(_engine(epsilon=0.0, arm=PREMIUM))
        router = litellm.Router(model_list=_model_list(), plugins=[plugin])
        response = asyncio.run(router.acompletion(model="smart", messages=MESSAGES))
        # The engine has no exploration and always prefers PREMIUM, so the only
        # deployment LiteLLM could have reached is that one.
        assert response.choices[0].message.content == f"served by {PREMIUM}"

    def test_exploration_actually_spreads_traffic(self) -> None:
        plugin = SortitionPlugin(_engine(epsilon=0.9, seed=3))
        router = litellm.Router(model_list=_model_list(), plugins=[plugin])

        async def run() -> set[str]:
            seen = set()
            for _ in range(40):
                response = await router.acompletion(model="smart", messages=MESSAGES)
                seen.add(response.choices[0].message.content)
            return seen

        # Without this, the log can only ever confirm what the policy prefers.
        assert len(asyncio.run(run())) > 1

    def test_plugin_fails_open(self) -> None:
        class Exploding:
            name = "boom"

            def score(self, features: dict[str, Any], eligible: tuple[str, ...]) -> Any:
                raise RuntimeError("policy is broken")

        plugin = SortitionPlugin(DecisionEngine(policy=Exploding()))
        router = litellm.Router(model_list=_model_list(), plugins=[plugin])

        # A bug in sortition must cost one unlogged row, never a failed request.
        response = asyncio.run(router.acompletion(model="smart", messages=MESSAGES))
        assert response.choices[0].message.content.startswith("served by ")


class TestLogging:
    def test_logger_writes_an_evaluable_row(self, tmp_path: Path) -> None:
        store = LogStore(tmp_path / "logs", flush_every=1)
        litellm.callbacks = [SortitionLogger(store)]
        router = litellm.Router(
            model_list=_model_list(), plugins=[SortitionPlugin(_engine())]
        )

        async def run() -> None:
            await router.acompletion(model="smart", messages=MESSAGES)
            await _drain()

        asyncio.run(run())
        rows = store.read()
        assert rows.height == 1

        row = rows.row(0, named=True)
        assert row["chosen_arm"] in ARMS
        assert 0.0 < row["propensity"] <= 1.0
        assert row["served_arm"] == row["chosen_arm"]
        assert row["fallback_depth"] == 0
        assert row["policy_version"] == "always:openai/gpt-4.1+eps0.2"
        assert row["cost_usd"] is not None
        assert row["tokens_in"] is not None
        assert set(row["eligible_set"]) == set(ARMS)

    def test_fallback_is_detected(self, tmp_path: Path) -> None:
        # The premium arm always fails, and the engine always prefers it, so
        # LiteLLM's own failover has to step in and serve something else.
        store = LogStore(tmp_path / "logs", flush_every=1)
        litellm.callbacks = [SortitionLogger(store)]
        router = litellm.Router(
            model_list=_model_list(broken=PREMIUM),
            plugins=[SortitionPlugin(_engine(epsilon=0.0, arm=PREMIUM))],
            fallbacks=[{"smart": ["smart"]}],
            num_retries=2,
        )

        async def run() -> None:
            with pytest.raises(Exception, match="simulated provider outage"):
                await router.acompletion(model="smart", messages=MESSAGES)
            await _drain()

        asyncio.run(run())
        rows = store.read()
        assert rows.height >= 1
        # However the gateway resolved it, the row must not claim the sampled
        # arm served successfully.
        assert all(
            r["status"] == "failure" or r["served_arm"] != r["chosen_arm"]
            for r in rows.iter_rows(named=True)
        )

    def test_late_outcomes_join_onto_decisions(self, tmp_path: Path) -> None:
        store = LogStore(tmp_path / "logs", flush_every=1)
        sink = SortitionLogger(store)
        litellm.callbacks = [sink]
        router = litellm.Router(
            model_list=_model_list(), plugins=[SortitionPlugin(_engine())]
        )

        async def run() -> None:
            await router.acompletion(model="smart", messages=MESSAGES)
            await _drain()

        asyncio.run(run())
        request_id = store.read().row(0, named=True)["request_id"]

        # Hours later, in real deployments.
        sink.record_outcome(request_id, 1.0)
        joined = store.load()
        assert joined.row(0, named=True)["outcome"] == 1.0


class TestEndToEnd:
    @pytest.fixture(scope="class")
    def traffic(self, tmp_path_factory: pytest.TempPathFactory) -> Any:
        """Drive a few hundred requests through the whole stack, once."""
        directory = tmp_path_factory.mktemp("traffic")
        store = LogStore(directory / "logs", flush_every=50)
        sink = SortitionLogger(store)
        saved = list(litellm.callbacks)
        litellm.callbacks = [sink]

        epsilon, preferred = 0.3, MID
        router = litellm.Router(
            model_list=_model_list(),
            plugins=[SortitionPlugin(_engine(epsilon=epsilon, seed=11, arm=preferred))],
        )

        async def run() -> None:
            for i in range(300):
                await router.acompletion(model="smart", messages=MESSAGES)
                if i % 3 == 0:
                    await asyncio.sleep(0)
            await _drain()

        asyncio.run(run())
        litellm.callbacks = saved

        rows = store.read()
        for request_id in rows.get_column("request_id").to_list():
            sink.record_outcome(request_id, float(random.random() < 0.6))
        return store.load(), epsilon, preferred

    def test_the_log_is_evaluable(self, traffic: Any) -> None:
        logs, _, _ = traffic
        assert logs.height >= 250
        report = doctor(logs)
        assert "effectively unexplored" not in report
        assert "ESS" in report

    def test_estimates_have_intervals(self, traffic: Any) -> None:
        logs, _, _ = traffic
        estimate = evaluate(logs, f"always:{PREMIUM}", metric="outcome")
        assert estimate.interval is not None
        assert estimate.trustworthy
        assert 0.0 <= estimate.value <= 1.0

    def test_cost_is_estimable_too(self, traffic: Any) -> None:
        logs, _, _ = traffic
        estimate = evaluate(logs, f"always:{CHEAP}", metric="cost_usd")
        assert estimate.interval is not None
        assert estimate.value > 0.0

    def test_the_deployed_policy_evaluates_as_itself(self, traffic: Any) -> None:
        """The round trip, through the gateway rather than in isolation.

        Real traffic, a real Router, a real callback, a real parquet file -- and
        the policy that produced it still has importance weights of exactly 1
        when evaluated against its own logs. Any drift between how the plugin
        computes a propensity and how the evaluator computes a probability would
        show up here as weights that are merely close to 1.
        """
        logs, epsilon, preferred = traffic
        arrays = to_arrays(logs)
        target = EpsilonFloor(AlwaysArm(preferred), epsilon)
        target_probs = target.probabilities(
            arrays.features, arrays.eligible, arrays.arms
        )
        weights = importance_weights(arrays.action, arrays.propensity, target_probs)
        np.testing.assert_allclose(weights, 1.0, rtol=1e-12)
