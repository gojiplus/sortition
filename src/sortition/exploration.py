"""The exploration math, defined once for both the deploy and evaluate paths.

This module exists to prevent a specific, silent failure. The propensity logged
when a request is routed and the probability computed when that same policy is
later evaluated must come from the same formula. If they drift apart by even a
rounding rule, evaluating a policy against its own logs stops returning
importance weights of exactly 1, every estimate acquires a bias nothing in the
output advertises, and the diagnostics cannot see it because the weights still
look reasonable.

So `sortition.decide` and `sortition.targets` both call these functions, and a
test asserts the round trip holds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def uniform_over_eligible(eligible: BoolArray) -> FloatArray:
    """Spread probability evenly across the arms that survived the hard filter.

    Args:
        eligible: An ``(n, K)`` boolean mask of arms available per row.

    Returns:
        An ``(n, K)`` array whose rows sum to 1, zero off-support.

    Raises:
        ValueError: If any row has an empty eligible set.
    """
    counts = eligible.sum(axis=1, keepdims=True)
    if (counts == 0).any():
        raise ValueError(
            "some rows have an empty eligible set; nothing could have served them"
        )
    return np.where(eligible, 1.0 / counts, 0.0)


def apply_epsilon_floor(
    greedy_probs: FloatArray, eligible: BoolArray, epsilon: float
) -> FloatArray:
    """Mix a policy with uniform exploration.

    With a deterministic ``greedy_probs`` this is exactly epsilon-greedy: the
    preferred arm gets ``1 - eps + eps/|E|`` and every other eligible arm gets
    ``eps/|E|``. Those probabilities are analytic, which is the whole reason to
    prefer epsilon-greedy for a first deployment -- nothing downstream inherits
    sampling error from the sampler itself.

    Args:
        greedy_probs: An ``(n, K)`` array of the unexplored policy's probabilities.
        eligible: An ``(n, K)`` boolean mask of arms available per row.
        epsilon: Share of traffic to spread uniformly over the eligible set.

    Returns:
        An ``(n, K)`` array whose rows sum to 1.

    Raises:
        ValueError: If ``epsilon`` is outside [0, 1].
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
    if epsilon == 0.0:
        return greedy_probs
    explore = uniform_over_eligible(eligible)
    return np.asarray(
        (1.0 - epsilon) * greedy_probs + epsilon * explore, dtype=np.float64
    )


def epsilon_greedy_propensity(
    *, n_eligible: int, epsilon: float, is_greedy: bool
) -> float:
    """The scalar propensity of one arm under epsilon-greedy.

    The per-request counterpart of :func:`apply_epsilon_floor`, for the decision
    path where only one row exists and numpy would be overhead.

    Args:
        n_eligible: Size of the eligible set the sampler drew from.
        epsilon: Share of traffic spread uniformly over that set.
        is_greedy: Whether this arm was the policy's preferred one.

    Returns:
        P(arm | features, policy), strictly positive whenever epsilon > 0.

    Raises:
        ValueError: If the eligible set is empty or epsilon is outside [0, 1].
    """
    if n_eligible <= 0:
        raise ValueError("eligible set is empty; nothing could have served the request")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
    share = epsilon / n_eligible
    return share + (1.0 - epsilon) if is_greedy else share
