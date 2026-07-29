"""Fitting a policy from logs, and the discipline that makes the result honest.

The model is the easy part. Two things around it decide whether the number you
end up quoting means anything.

**Train and evaluate on different rows.** A policy chosen to look good on a set
of logs will look good on those logs. :func:`train_test_split` exists so the
comparison against the incumbent is made somewhere the candidate has not seen,
and :func:`train` refuses to quietly do otherwise.

**Weight the fit by inverse propensity.** The logs were not collected uniformly:
the incumbent policy sent most traffic to the arms it already liked, so an
unweighted fit learns most about those arms and least about the ones a candidate
policy would need to be confident about. Weighting by 1/propensity recovers what
the outcome model would have seen under uniform assignment, which is the same
correction the estimators make and for the same reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from sortition.decide.engine import ExplorationConfig
from sortition.decide.tree import TreePolicy
from sortition.features import infer_spec, matrix
from sortition.frame import to_arrays
from sortition.schema import PolicyArtifact

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)

# Below this, a boosted model is fitting noise and a rules table is the better
# starting point.
MIN_ROWS_TO_TRAIN = 500


@dataclass(frozen=True)
class TrainingResult:
    """A fitted policy and what it was fitted on."""

    artifact: PolicyArtifact
    policy: TreePolicy
    n_rows: int
    feature_spec: tuple[str, ...]
    arms: tuple[str, ...]


def train_test_split(
    logs: pl.DataFrame, *, holdout: float = 0.3, seed: int = 0
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split logs into a training set and a held-out set.

    Split at random rather than by time. A time split would confound the
    candidate's quality with whatever else changed between the two periods --
    traffic mix, provider latency, a prompt change -- and there is no way to tell
    those apart afterwards.

    Args:
        logs: The log table.
        holdout: Share of rows reserved for evaluation.
        seed: Seed for the split.

    Returns:
        The training rows and the held-out rows.

    Raises:
        ValueError: If ``holdout`` is not strictly between 0 and 1.
    """
    if not 0.0 < holdout < 1.0:
        raise ValueError(f"holdout must be in (0, 1), got {holdout}")
    rng = np.random.default_rng(seed)
    mask = rng.random(logs.height) >= holdout
    return logs.filter(mask), logs.filter(~mask)


def train(
    logs: pl.DataFrame,
    *,
    metric: str = "outcome",
    cost_weight: float = 0.0,
    epsilon: float = 0.05,
    name: str | None = None,
    weight_by_propensity: bool = True,
    seed: int = 0,
    **booster_kwargs: Any,
) -> TrainingResult:
    """Fit a tree policy from logged routing decisions.

    Args:
        logs: Training rows. Use :func:`train_test_split` and keep the rest back,
            or the comparison against the incumbent will flatter the result.
        metric: Outcome column to predict.
        cost_weight: How much predicted quality to trade for price. Zero picks
            the best arm regardless of cost.
        epsilon: Exploration floor for the resulting policy. Keeping it above
            zero is what lets the *next* policy be trained from these logs too.
        name: Label prefixed to the artifact's content hash.
        weight_by_propensity: Correct for the incumbent's non-uniform assignment.
        seed: Seed for the booster.
        **booster_kwargs: Passed to ``LGBMRegressor``.

    Returns:
        The fitted policy and its artifact.

    Raises:
        ValueError: If the log lacks the metric, has no usable features, or has
            too few rows to fit anything meaningful.
    """
    from lightgbm import LGBMRegressor

    from sortition.decide.artifact import build

    data = to_arrays(logs, metrics=(metric,))
    if metric not in data.metrics:
        raise ValueError(f"log has no {metric!r} column to learn from")

    values = data.metrics[metric]
    observed = ~np.isnan(values)
    if int(observed.sum()) < MIN_ROWS_TO_TRAIN:
        raise ValueError(
            f"only {int(observed.sum())} rows have a {metric!r} outcome; below "
            f"{MIN_ROWS_TO_TRAIN} a boosted model fits noise and a rules table "
            "is the better starting point"
        )

    spec = infer_spec(data.features)
    if not spec:
        raise ValueError(
            "no feature is numeric on every row, so there is nothing to learn "
            "from; check what the gateway is recording in `features`"
        )

    features = matrix(data.features, spec)[observed]
    action = data.action[observed]
    target = values[observed]

    # Arm identity as a one-hot block: one model over all arms shares what makes
    # a request hard, which per-arm models discard where data is thinnest.
    onehot = np.zeros((len(action), len(data.arms)), dtype=np.float64)
    onehot[np.arange(len(action)), action] = 1.0
    design = np.hstack([features, onehot])

    sample_weight = None
    if weight_by_propensity:
        # The same correction the estimators apply: undo the incumbent's
        # preference so the model learns about arms it rarely chose.
        sample_weight = 1.0 / data.propensity[observed]
        sample_weight = sample_weight / sample_weight.mean()

    booster = LGBMRegressor(
        n_estimators=booster_kwargs.pop("n_estimators", 300),
        learning_rate=booster_kwargs.pop("learning_rate", 0.05),
        num_leaves=booster_kwargs.pop("num_leaves", 31),
        min_child_samples=booster_kwargs.pop("min_child_samples", 20),
        random_state=seed,
        verbose=-1,
        **booster_kwargs,
    )
    booster.fit(design, target, sample_weight=sample_weight)

    costs = _mean_cost_per_arm(logs, data)
    policy = TreePolicy(
        booster_text=booster.booster_.model_to_string(),
        feature_spec=spec,
        arms=data.arms,
        cost_usd=costs,
        cost_weight=cost_weight,
        name=name or "tree",
    )

    artifact = build(policy, ExplorationConfig(epsilon=epsilon), name=name)
    logger.info(
        "trained %s on %d rows, %d features, %d arms",
        artifact.policy_version,
        int(observed.sum()),
        len(spec),
        len(data.arms),
    )
    return TrainingResult(
        artifact=artifact,
        policy=policy,
        n_rows=int(observed.sum()),
        feature_spec=spec,
        arms=data.arms,
    )


def _mean_cost_per_arm(logs: pl.DataFrame, data: Any) -> dict[str, float]:
    """Average observed cost per arm, for the policy's cost term.

    Args:
        logs: The log table.
        data: The reduced arrays, for the arm universe.

    Returns:
        Mean cost per arm, empty when the log has no cost column.
    """
    if "cost_usd" not in logs.columns:
        return {}
    costs: dict[str, float] = {}
    for arm in data.arms:
        rows = (
            logs.filter(logs["chosen_arm"] == arm).get_column("cost_usd").drop_nulls()
        )
        if len(rows):
            costs[arm] = float(rows.to_numpy().mean())
    return costs
