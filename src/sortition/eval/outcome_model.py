"""Cross-fitted outcome models for the doubly robust estimator.

DR needs ``q_hat(x, a)`` for *every* arm, including the ones not taken. That is
what rules out ``cross_val_predict``: it returns out-of-fold predictions at the
observed rows only, and the counterfactual arms are the whole point. So the
folds are walked explicitly -- fit on K-1 folds, predict all arms on the held-out
one.

The cross-fitting is not a refinement. An outcome model that has already seen
the row it is predicting produces residuals that are too small, which quiets the
IPS correction term and biases DR toward whatever the model believes. Nothing in
the output looks wrong when this happens, which is why the test suite asserts
the in-sample variant *is* biased.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class Regressor(Protocol):
    """The sklearn-style fit/predict pair this module needs."""

    def fit(self, X: FloatArray, y: FloatArray) -> Any: ...
    def predict(self, X: FloatArray) -> Any: ...


def default_regressor(seed: int = 0) -> Regressor:
    """LightGBM when installed, else sklearn's histogram booster.

    Both are gradient-boosted trees over the same features; the fallback keeps
    the eval path installable without the training extra.
    """
    try:
        from lightgbm import LGBMRegressor

        model: Regressor = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            random_state=seed,
            verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=seed,
        )
    return model


def _design(contexts: FloatArray, arm: IntArray | int, n_arms: int) -> FloatArray:
    """Context features plus a one-hot arm indicator.

    One model over all arms rather than one model per arm: arms share most of
    what makes a request easy or hard, and per-arm models throw that away
    exactly where data is thinnest.
    """
    n = contexts.shape[0]
    onehot = np.zeros((n, n_arms), dtype=np.float64)
    if isinstance(arm, int | np.integer):
        onehot[:, int(arm)] = 1.0
    else:
        onehot[np.arange(n), arm] = 1.0
    return np.hstack([contexts, onehot])


def fit_outcome_model(
    contexts: FloatArray,
    action: IntArray,
    reward: FloatArray,
    n_arms: int,
    *,
    n_folds: int = 5,
    seed: int = 0,
    cross_fit: bool = True,
    make_regressor: Any = None,
) -> FloatArray:
    """Out-of-fold ``q_hat`` of shape ``(n, n_arms)``.

    Set ``cross_fit=False`` only to demonstrate the bias it causes; it is never
    the right choice for a real estimate.
    """
    from sklearn.model_selection import KFold

    contexts = np.asarray(contexts, dtype=np.float64)
    action = np.asarray(action, dtype=np.int64)
    reward = np.asarray(reward, dtype=np.float64)
    n = contexts.shape[0]
    factory = make_regressor or default_regressor
    q_hat = np.zeros((n, n_arms), dtype=np.float64)

    if not cross_fit:
        model = factory(seed)
        model.fit(_design(contexts, action, n_arms), reward)
        for a in range(n_arms):
            q_hat[:, a] = model.predict(_design(contexts, a, n_arms))
        return q_hat

    folds = KFold(n_splits=min(n_folds, n), shuffle=True, random_state=seed)
    for train_idx, test_idx in folds.split(contexts):
        model = factory(seed)
        model.fit(_design(contexts[train_idx], action[train_idx], n_arms), reward[train_idx])
        for a in range(n_arms):
            q_hat[test_idx, a] = model.predict(_design(contexts[test_idx], a, n_arms))
    return q_hat
