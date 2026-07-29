"""The log schema is the stable contract; these tests are what freeze it."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from sortition.schema import SCHEMA_VERSION, Decision, DecisionRow, ExecutionRow, OutcomeRow


def _decision(**overrides: object) -> Decision:
    base: dict[str, object] = {
        "decision_id": "d-1",
        "policy_version": "rules-v1+eps05",
        "chosen_arm": "premium",
        "propensity": 0.95,
        "eligible_set": ("premium", "standard"),
    }
    return Decision(**(base | overrides))  # type: ignore[arg-type]


def test_decision_roundtrips() -> None:
    d = _decision()
    assert Decision.model_validate_json(d.model_dump_json()) == d
    assert d.schema_version == SCHEMA_VERSION


def test_chosen_arm_must_be_eligible() -> None:
    # The propensity describes a draw from the eligible set. An arm outside it
    # has no such probability, so the row would be uninterpretable downstream.
    with pytest.raises(ValidationError, match="not in eligible_set"):
        _decision(chosen_arm="experimental")


def test_zero_propensity_rejected() -> None:
    # A zero-probability action cannot have been logged, and would put an
    # infinite importance weight into every estimate built on the row.
    with pytest.raises(ValidationError):
        _decision(propensity=0.0)


def test_propensity_must_be_a_probability() -> None:
    with pytest.raises(ValidationError):
        _decision(propensity=1.4)


def test_decision_is_immutable() -> None:
    # Decisions are historical facts; a mutated one makes its estimate
    # irreproducible.
    with pytest.raises(ValidationError):
        _decision().chosen_arm = "standard"  # type: ignore[misc]


def test_fallback_chain_carries_no_propensities() -> None:
    # Fallback arms are instructions to the gateway, not draws. If they ever
    # gain probabilities, this contract has been broken.
    d = _decision(fallback_chain=("standard",))
    assert d.fallback_chain == ("standard",)
    assert isinstance(d.propensity, float)


def test_unknown_fields_rejected() -> None:
    # Additive-only means new fields arrive deliberately, with defaults.
    with pytest.raises(ValidationError):
        _decision(score=0.9)


def test_execution_row_defaults_to_no_fallback() -> None:
    row = ExecutionRow(request_id="r-1", served_arm="premium")
    assert row.fallback_depth == 0
    assert row.status == "success"


def test_outcomes_are_separate_from_decisions() -> None:
    # Outcomes arrive late. Keeping them in their own append-only table is what
    # lets a decision row stay immutable.
    assert "outcome" not in DecisionRow.model_fields
    assert "value" not in DecisionRow.model_fields
    o = OutcomeRow(request_id="r-1", ts=datetime.now(UTC), value=1.0)
    assert o.bounded is True


def test_decision_row_reserves_session_id() -> None:
    # Multi-turn affinity is deferred, but the column exists now so the data is
    # there when it is addressed.
    row = DecisionRow(request_id="r-1", ts=datetime.now(UTC), decision=_decision())
    assert row.session_id is None
