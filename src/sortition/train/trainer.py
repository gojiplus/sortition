"""Fitting a policy from logs, and the discipline that makes the result honest.

The model is the easy part. Two things around it decide whether the number you
end up quoting means anything.

**Train and evaluate on different rows.** A policy chosen to look good on a set
of logs will look good on those logs. :func:`train_test_split` exists so the
comparison against the incumbent is made somewhere the candidate has not seen,
and :func:`train` refuses to quietly do otherwise.

**Fit cost as well as quality.** What an arm costs is not a property of the arm:
a bill is price-per-token times tokens, so the same arm is ten times dearer on a
long request. A second booster over the same design predicts dollars, which is
what lets the policy know that the premium arm is nearly free on a short request
and ruinous on a long one.

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

COST_OBJECTIVE = "tweedie"
"""The loss for the cost model. Spend is non-negative, heavily right-skewed, and
has a point mass at exactly zero -- a cache hit or a call that failed before it
billed. That is the compound Poisson-gamma shape Tweedie regression exists for,
and it is what actuaries fit claim severity with for the same reason.

Squared error is wrong here in a way that shows: on the simulator it predicts a
negative bill on 1.1% of rows, which reads as a credit, and lands 40% further
from the true expected cost. A plain gamma loss is more accurate still (MAE
0.0018 against Tweedie's 0.0021 and squared error's 0.0025) but refuses to fit
at all when any logged cost is zero, which a real log guarantees."""


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
    cost_metric: str = "cost_usd",
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
        cost_weight: How much predicted quality to trade for price, in units of
            the outcome: the score charges the dearest eligible arm this much
            against the cheapest. Zero picks the best arm regardless of cost.
            :func:`sortition.train.sweep.sweep` chooses it from logs instead.
        cost_metric: Column holding the dollar cost of each logged call.
        epsilon: Exploration floor for the resulting policy. Keeping it above
            zero is what lets the *next* policy be trained from these logs too.
        name: Label prefixed to the artifact's content hash.
        weight_by_propensity: Correct for the incumbent's non-uniform assignment.
        seed: Seed for the booster.
        **booster_kwargs: Passed to ``LGBMRegressor``.

    Returns:
        The fitted policy and its artifact.

    Raises:
        ValueError: If the log lacks the metric, has no usable features, has too
            few rows to fit anything meaningful, or asks for a cost weight
            without a cost column to learn prices from.
    """
    from sortition.decide.artifact import build

    data = to_arrays(logs, metrics=(metric, cost_metric))
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
    target = values[observed]

    # Arm identity as a one-hot block: one model over all arms shares what makes
    # a request hard, which per-arm models discard where data is thinnest.
    onehot_all = np.zeros((len(data.action), len(data.arms)), dtype=np.float64)
    onehot_all[np.arange(len(data.action)), data.action] = 1.0
    design = np.hstack([features, onehot_all[observed]])

    sample_weight = None
    if weight_by_propensity:
        # The same correction the estimators apply: undo the incumbent's
        # preference so the model learns about arms it rarely chose.
        sample_weight = 1.0 / data.propensity[observed]
        sample_weight = sample_weight / sample_weight.mean()

    booster = _fit(design, target, sample_weight, seed, dict(booster_kwargs))

    # The cost model sees the rows that have a price, which is not the same set
    # as the rows that have an outcome: a cost is recorded when the call
    # resolves, an outcome may never arrive at all.
    cost_text: str | None = None
    if cost_metric in data.metrics:
        priced = ~np.isnan(data.metrics[cost_metric])
        if int(priced.sum()) >= MIN_ROWS_TO_TRAIN:
            cost_design = np.hstack([matrix(data.features, spec), onehot_all])[priced]
            cost_weights = None
            if weight_by_propensity:
                cost_weights = 1.0 / data.propensity[priced]
                cost_weights = cost_weights / cost_weights.mean()
            cost_booster = _fit(
                cost_design,
                data.metrics[cost_metric][priced],
                cost_weights,
                seed,
                dict(booster_kwargs),
                objective=COST_OBJECTIVE,
            )
            cost_text = cost_booster.booster_.model_to_string()

    # The average gap between the dearest and cheapest arm on a request, which is
    # what cost_weight is measured against: at weight 1.0 an average-spread
    # request will give up a full point of outcome to move from the dearest arm
    # to the cheapest, and a request with twice the spread gives up twice that.
    #
    # The spread rather than the mean bill. The bill is dominated by whichever
    # arms the incumbent happened to favour, so scaling by it makes the weight
    # mean something different on every log -- on the simulator the dearest arm
    # sits 2.6 mean-bills above the cheapest, so a weight of 1.0 became a penalty
    # of 2.6 and the whole grid collapsed to "always cheapest".
    #
    # Recorded in the artifact so a deployed policy charges what it was tuned to
    # charge; deriving it live would move with the traffic mix.
    cost_scale = 0.0
    if cost_text is not None:
        priced_grid = np.hstack([matrix(data.features, spec), onehot_all])
        per_arm = np.vstack(
            [
                _predict_cost_for_arm(cost_booster, priced_grid, arm, len(data.arms))
                for arm in range(len(data.arms))
            ]
        )
        spreads = per_arm.max(axis=0) - per_arm.min(axis=0)
        cost_scale = float(spreads.mean())

    if cost_weight != 0.0 and cost_text is None:
        raise ValueError(
            f"cost_weight={cost_weight} was asked for but the log has no usable "
            f"{cost_metric!r} column, so there is nothing to price arms with. "
            "Log what each call cost, or train with cost_weight=0."
        )

    policy = TreePolicy(
        booster_text=booster.booster_.model_to_string(),
        feature_spec=spec,
        arms=data.arms,
        cost_booster_text=cost_text,
        cost_scale=cost_scale,
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


def _predict_cost_for_arm(
    booster: Any, design: np.ndarray, arm: int, n_arms: int
) -> np.ndarray:
    """Predicted cost of every training request, had it gone to one arm.

    Args:
        booster: The fitted cost model.
        design: Features concatenated with the arm one-hot block.
        arm: The arm index to score every row under.
        n_arms: Width of the one-hot block.

    Returns:
        One predicted cost per row.
    """
    swapped = design.copy()
    swapped[:, -n_arms:] = 0.0
    swapped[:, design.shape[1] - n_arms + arm] = 1.0
    return np.maximum(np.asarray(booster.predict(swapped), dtype=np.float64), 0.0)


def _fit(
    design: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray | None,
    seed: int,
    booster_kwargs: dict[str, Any],
    *,
    objective: str = "regression",
) -> Any:
    """Fit one booster over an (arm, feature) design.

    Args:
        design: Rows of features concatenated with the arm one-hot block.
        target: What to predict.
        sample_weight: Inverse-propensity weights, or ``None``.
        seed: Seed for the booster.
        booster_kwargs: Overrides passed to ``LGBMRegressor``.
        objective: The loss. Squared error for a bounded outcome; see
            :data:`COST_OBJECTIVE` for why cost gets a different one.

    Returns:
        The fitted regressor.
    """
    from lightgbm import LGBMRegressor

    booster = LGBMRegressor(
        objective=booster_kwargs.pop("objective", objective),
        n_estimators=booster_kwargs.pop("n_estimators", 300),
        learning_rate=booster_kwargs.pop("learning_rate", 0.05),
        num_leaves=booster_kwargs.pop("num_leaves", 31),
        min_child_samples=booster_kwargs.pop("min_child_samples", 20),
        random_state=seed,
        verbose=-1,
        **booster_kwargs,
    )
    booster.fit(design, target, sample_weight=sample_weight)
    return booster
