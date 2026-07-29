"""A learned policy: gradient-boosted trees over the logged features.

Ranks arms by predicted outcome minus a weighted, normalized cost. The cost
weight is the operator's exchange rate between quality and price, and it is the
one number worth tuning from logs rather than guessing.

This satisfies the same ``Policy`` protocol as a rules table, so a trained policy
deploys through the same engine, serializes into the same artifact format, and is
evaluated by the same estimators. Replacing rules with a model changes no other
part of the system, which is what the frozen artifact schema was for.

Feature vectors are built through ``sortition.features`` against the spec
recorded in the artifact, so the policy scores live requests exactly as it was
trained.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sortition.features import vectorize

logger = logging.getLogger(__name__)


@dataclass
class TreePolicy:
    """Scores arms with a boosted-tree outcome model.

    One model over all arms, with the arm as a one-hot block, rather than a
    model per arm: arms share most of what makes a request easy or hard, and
    per-arm models discard that exactly where data is thinnest.
    """

    booster_text: str
    """The serialized LightGBM model. Text so an artifact stays diffable and
    needs no pickle, which would make deploying a policy a code-execution
    decision."""

    feature_spec: tuple[str, ...]
    arms: tuple[str, ...]
    cost_usd: dict[str, float] = field(default_factory=dict)
    cost_weight: float = 0.0
    name: str = "tree"
    _booster: Any = field(default=None, init=False, repr=False, compare=False)

    @property
    def booster(self) -> Any:
        """The deserialized model, loaded on first use.

        Returns:
            A LightGBM ``Booster``.
        """
        if self._booster is None:
            import lightgbm as lgb

            self._booster = lgb.Booster(model_str=self.booster_text)
        return self._booster

    def predict(
        self, features: dict[str, Any], eligible: tuple[str, ...]
    ) -> dict[str, float]:
        """Predicted outcome for each eligible arm.

        Args:
            features: The request's feature vector.
            eligible: Arms surviving the hard filter.

        Returns:
            A predicted outcome per arm.
        """
        base = vectorize(features, self.feature_spec)
        index = {arm: i for i, arm in enumerate(self.arms)}
        rows = []
        for arm in eligible:
            onehot = [0.0] * len(self.arms)
            if arm in index:
                onehot[index[arm]] = 1.0
            rows.append(base + onehot)
        predictions = self.booster.predict(np.array(rows, dtype=np.float64))
        return dict(zip(eligible, (float(p) for p in predictions), strict=True))

    def score(
        self, features: dict[str, Any], eligible: tuple[str, ...]
    ) -> dict[str, float]:
        """Rank arms by predicted outcome, discounted by cost.

        Args:
            features: The request's feature vector.
            eligible: Arms surviving the hard filter.

        Returns:
            A score per eligible arm; higher is preferred.
        """
        if not eligible:
            return {}
        quality = self.predict(features, eligible)
        if self.cost_weight == 0.0 or not self.cost_usd:
            return quality

        costs = [self.cost_usd.get(arm, 0.0) for arm in eligible]
        spread = max(costs) - min(costs)
        if spread <= 0.0:
            return quality
        cheapest = min(costs)
        # Cost is normalized within the eligible set rather than globally, so
        # the weight means the same thing whatever mix of arms a request sees.
        return {
            arm: quality[arm]
            - self.cost_weight * ((self.cost_usd.get(arm, 0.0) - cheapest) / spread)
            for arm in eligible
        }
