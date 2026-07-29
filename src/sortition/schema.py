"""The stable contract.

Everything else in this project may change. These types are versioned and
additive-only: fields may be added with defaults, never removed or retyped.

Three tables, deliberately separate:

    Decision   what the policy chose, and with what probability
    Execution  what the gateway actually did (arrives at call resolution)
    Outcome    how it turned out (may arrive hours later)

They are joined by ``request_id`` at evaluation time. The decision row is never
mutated, which is what keeps a historical estimate reproducible.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1

Arm = str
"""A logical model group (``"premium-reasoning"``), never a deployment or endpoint.

Sortition names the group; the gateway resolves it to somewhere to send bytes.
"""

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class Decision(BaseModel):
    """What the policy chose for one request.

    A single sampled arm, not a ranked list with a probability on each entry.
    The ranked-list form cannot be made unbiased: if the gateway falls back, the
    arm that served was drawn from (sampler o provider-failure), not from the
    sampler, and its logged probability no longer describes how it was selected.

    ``fallback_chain`` therefore carries no propensities. Those arms are
    instructions to the gateway, not draws from a distribution.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = SCHEMA_VERSION
    decision_id: str
    policy_version: str
    """Identifies the artifact that decided. Makes any decision reproducible."""

    chosen_arm: Arm
    propensity: Probability = Field(gt=0.0)
    """P(chosen_arm | features, policy), over ``eligible_set``.

    Strictly positive: an arm drawn with probability zero is a contradiction, and
    a zero here would put an infinite importance weight into every downstream
    estimate.
    """

    eligible_set: tuple[Arm, ...]
    """Arms surviving the hard filter. The support the sampler drew from.

    Logged because a target policy that assigns mass outside this set has no
    observable counterfactual on this row, and the estimator must say so rather
    than silently extrapolate.
    """

    explore: bool = False
    fallback_chain: tuple[Arm, ...] = ()
    hard_constraints_applied: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _chosen_arm_is_eligible(self) -> Decision:
        if self.chosen_arm not in self.eligible_set:
            raise ValueError(
                f"chosen_arm {self.chosen_arm!r} is not in eligible_set "
                f"{self.eligible_set!r}; the propensity would describe a draw "
                "from a distribution the arm was not part of"
            )
        return self


class DecisionRow(BaseModel):
    """A decision as persisted: the choice, its context, and its features."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    request_id: str
    ts: datetime
    tenant: str | None = None
    session_id: str | None = None
    """Reserved. Multi-turn affinity correlates arms across rows and breaks the
    independence the estimators assume; recorded now so the data exists when
    that is addressed."""

    decision: Decision
    features: dict[str, Any] = Field(default_factory=dict)
    """The exact feature vector the policy saw. Not a reconstruction."""


class ExecutionRow(BaseModel):
    """What the gateway actually did, recorded when the call resolves."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    request_id: str
    served_arm: Arm | None
    """The arm that actually produced the response. ``None`` if the call failed
    outright."""

    fallback_depth: int = Field(default=0, ge=0)
    """0 if the chosen arm served. Greater than 0 means the gateway's own retry
    or failover machinery overrode the decision, so the served arm was not drawn
    from the sampler. Such rows are excluded from evaluation by default and
    reported as a logging leakage rate."""

    status: Literal["success", "failure"] = "success"
    cost_usd: float | None = None
    latency_ms: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None


class OutcomeRow(BaseModel):
    """How a request turned out.

    Append-only and separate from the decision, because outcomes are late: a
    thumbs-up arrives in seconds, a resolution flag or CSAT score in hours. An
    upsert into the decision row would make historical estimates irreproducible,
    so outcomes accumulate here and are joined at evaluation time.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    request_id: str
    ts: datetime
    name: str = "outcome"
    """Which outcome this is. Multiple named outcomes per request are expected
    (``"thumbs"``, ``"resolved"``, ``"eval_score"``)."""

    value: float
    bounded: bool = True
    """Whether ``value`` lies in [0, 1]. Bounded outcomes get anytime-valid
    betting intervals; unbounded ones fall back to bootstrap with winsorization."""


class PolicyArtifact(BaseModel):
    """A versioned, self-describing policy.

    Frozen now, before there is a trained policy to put in it, so that the
    trainer drops in later without touching the decision API.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    policy_version: str
    kind: Literal["rules", "tree", "constant", "weighted", "cost_aware"]
    """Widening this set is additive: every previously valid value still
    validates, so old artifacts keep loading."""

    created_at: datetime
    arms: tuple[Arm, ...]
    feature_spec: tuple[str, ...] = ()
    cost_weight: float = 0.0
    exploration: dict[str, Any] = Field(default_factory=dict)
    """``{"strategy": "epsilon_greedy", "epsilon": 0.05}`` or
    ``{"strategy": "thompson", "n_samples": 128}``."""

    git_sha: str | None = None
    training_window: tuple[datetime, datetime] | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    """Kind-specific body: rule table, serialized booster, constant arm."""
