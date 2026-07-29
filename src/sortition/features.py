"""Turning a logged feature dict into a vector, defined once.

The same reason ``sortition.exploration`` exists. A trained policy is fitted on
vectors built from logs and then scores vectors built from live requests; if
those two are assembled differently -- a different key order, a different
handling of a missing value -- the deployed policy is not the policy that was
trained, and nothing in its output would say so.

So the feature spec is recorded in the policy artifact, and both sides build
their vectors through this module against that spec.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

# bool is deliberately numeric: a tools-required flag is a real feature, and
# treating it as categorical would drop it.
_NUMERIC = (bool, int, float)


def infer_spec(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    """Choose the feature names a policy can be trained on.

    Only keys that are numeric on *every* row are kept. A key present on some
    rows and absent on others would have to be imputed, and a silently
    zero-filled feature is worse than a missing one because the model treats it
    as a measurement.

    Args:
        rows: Per-row feature dicts, as logged.

    Returns:
        Feature names in a stable order.
    """
    shared: set[str] | None = None
    for row in rows:
        numeric = {k for k, v in (row or {}).items() if isinstance(v, _NUMERIC)}
        shared = numeric if shared is None else (shared & numeric)
    return tuple(sorted(shared or ()))


def vectorize(row: dict[str, Any], spec: tuple[str, ...]) -> list[float]:
    """Build one feature vector against a fixed spec.

    A key the spec expects but the request lacks becomes 0.0, because refusing
    to route is worse than routing on an incomplete vector. It is logged at
    debug: a spec drifting away from what the gateway actually sends is a real
    problem, just not one worth failing a request over.

    Args:
        row: The request's features.
        spec: Feature names, in order.

    Returns:
        One value per name in ``spec``.
    """
    values: list[float] = []
    for name in spec:
        value = (row or {}).get(name)
        if isinstance(value, _NUMERIC):
            values.append(float(value))
        else:
            if value is not None:
                logger.debug("feature %r is not numeric (%r); using 0.0", name, value)
            values.append(0.0)
    return values


def matrix(rows: list[dict[str, Any]], spec: tuple[str, ...]) -> FloatArray:
    """Build a feature matrix against a fixed spec.

    Args:
        rows: Per-row feature dicts.
        spec: Feature names, in order.

    Returns:
        An ``(len(rows), len(spec))`` array.
    """
    if not spec:
        return np.zeros((len(rows), 0), dtype=np.float64)
    return np.array([vectorize(row, spec) for row in rows], dtype=np.float64)
