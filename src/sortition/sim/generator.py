"""A contextual bandit whose policy values are exact, not estimated.

The context pool is finite. That single choice is what makes the oracle exact:
with contexts drawn uniformly from a fixed pool of ``n_contexts``, the value of
any policy is a closed-form average over that pool,

    V(pi) = (1/M) sum_i sum_a pi(a | x_i) q(x_i, a)

with no Monte Carlo error of its own. A coverage test whose "truth" carried its
own sampling noise would be measuring two things at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


class Policy(Protocol):
    """Maps contexts and an eligibility mask to action probabilities.

    Returns an ``(n, K)`` array whose rows sum to 1. Well-behaved policies place
    no mass on ineligible arms; the estimators detect those that do rather than
    assuming it, which is how support violations get caught.
    """

    def __call__(self, contexts: FloatArray, eligible: BoolArray) -> FloatArray: ...


@dataclass(frozen=True)
class BanditProblem:
    """A bandit with a finite context pool and known expected rewards."""

    contexts: FloatArray
    """``(M, d)`` the full context pool."""

    q: FloatArray
    """``(M, K)`` expected reward per context-arm pair, in [0, 1]."""

    cost: FloatArray
    """``(K,)`` expected cost per arm in USD. Unbounded and right-skewed when
    realized, which exercises a different interval path than bounded rewards."""

    eligible: BoolArray
    """``(M, K)`` which arms survive the hard filter for each context."""

    arms: tuple[str, ...]

    @property
    def n_contexts(self) -> int:
        return int(self.contexts.shape[0])

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    def value(self, policy: Policy) -> float:
        """The exact expected reward of ``policy``. No sampling involved."""
        probs = policy(self.contexts, self.eligible)
        return float((probs * self.q).sum(axis=1).mean())

    def cost_value(self, policy: Policy) -> float:
        """The exact expected cost of ``policy``."""
        probs = policy(self.contexts, self.eligible)
        return float((probs * self.cost[None, :]).sum(axis=1).mean())


@dataclass(frozen=True)
class LoggedData:
    """One row per request, as an estimator sees it."""

    context_idx: IntArray
    """``(n,)`` index into the problem's context pool."""

    contexts: FloatArray
    """``(n, d)`` the contexts themselves, for fitting outcome models."""

    action: IntArray
    """``(n,)`` arm index actually chosen by the behavior policy."""

    propensity: FloatArray
    """``(n,)`` P(action | context, behavior policy). Strictly positive."""

    reward: FloatArray
    """``(n,)`` realized bounded outcome in [0, 1]."""

    cost: FloatArray
    """``(n,)`` realized cost in USD. Unbounded, right-skewed."""

    eligible: BoolArray
    """``(n, K)`` the support the behavior policy drew from."""

    @property
    def n(self) -> int:
        return int(self.action.shape[0])


def _masked_softmax(scores: FloatArray, eligible: BoolArray, temperature: float) -> FloatArray:
    """Softmax restricted to eligible arms, renormalized over the survivors."""
    masked = np.where(eligible, scores / temperature, -np.inf)
    masked = masked - masked.max(axis=1, keepdims=True)
    exp = np.where(eligible, np.exp(masked), 0.0)
    return exp / exp.sum(axis=1, keepdims=True)


def make_problem(
    *,
    n_contexts: int = 500,
    n_arms: int = 4,
    n_features: int = 6,
    seed: int = 0,
    ineligible_rate: float = 0.0,
) -> BanditProblem:
    """Build a problem with heterogeneous arm quality.

    Expected rewards vary with context, so a context-aware policy genuinely beats
    a constant one and the estimators have something to distinguish. Arm costs
    increase with index, mimicking the cheap-to-premium ladder that makes routing
    worth doing at all.

    ``ineligible_rate`` drops arms from the eligible set at random, which is how
    a hard constraint (tool support, context window, region) shows up in a log.
    Arm 0 is always kept eligible so that no context has an empty support.
    """
    rng = np.random.default_rng(seed)
    contexts = rng.standard_normal((n_contexts, n_features))

    # Each arm scores contexts differently; the logistic link keeps q in (0, 1)
    # so rewards are Bernoulli and the bounded-outcome interval path applies.
    weights = rng.standard_normal((n_features, n_arms)) * 0.8
    intercepts = np.linspace(-0.4, 0.4, n_arms)
    logits = contexts @ weights + intercepts[None, :]
    q = 1.0 / (1.0 + np.exp(-logits))

    cost = np.geomspace(0.001, 0.05, n_arms)

    eligible = np.ones((n_contexts, n_arms), dtype=bool)
    if ineligible_rate > 0.0:
        eligible = rng.random((n_contexts, n_arms)) >= ineligible_rate
        eligible[:, 0] = True

    arms = tuple(f"arm-{i}" for i in range(n_arms))
    return BanditProblem(contexts=contexts, q=q, cost=cost, eligible=eligible, arms=arms)


def sample_logs(
    problem: BanditProblem,
    behavior: Policy,
    n: int,
    *,
    seed: int = 0,
) -> LoggedData:
    """Draw ``n`` logged requests under ``behavior``.

    Contexts are drawn with replacement from the pool, which is what makes the
    pool average the population value the estimators are targeting.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, problem.n_contexts, size=n)
    contexts = problem.contexts[idx]
    eligible = problem.eligible[idx]

    probs = behavior(contexts, eligible)
    # Vectorized categorical draw: one uniform per row against the row's CDF.
    cdf = probs.cumsum(axis=1)
    draws = rng.random((n, 1))
    action = (draws > cdf).sum(axis=1).astype(np.int64)
    action = np.clip(action, 0, problem.n_arms - 1)

    rows = np.arange(n)
    propensity = probs[rows, action]
    q_taken = problem.q[idx, action]
    reward = (rng.random(n) < q_taken).astype(np.float64)

    # Realized cost is right-skewed around the arm's expected cost: token counts
    # vary a lot per request. Calibrated so the mean equals problem.cost[action].
    sigma = 0.6
    noise = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=n)
    cost = problem.cost[action] * noise

    return LoggedData(
        context_idx=idx.astype(np.int64),
        contexts=contexts,
        action=action,
        propensity=propensity,
        reward=reward,
        cost=cost,
        eligible=eligible,
    )


def uniform_policy() -> Policy:
    """Uniform over eligible arms. Maximal exploration, worst greedy value."""

    def policy(contexts: FloatArray, eligible: BoolArray) -> FloatArray:
        counts = eligible.sum(axis=1, keepdims=True)
        return np.where(eligible, 1.0 / counts, 0.0)

    return policy


def constant_policy(arm: int) -> Policy:
    """Always ``arm`` where eligible; falls back to uniform where it is not.

    The fallback matters: a constant policy that assigns mass to an ineligible
    arm has no observable counterfactual there, and silently pretending otherwise
    is exactly the bias this project exists to catch.
    """

    def policy(contexts: FloatArray, eligible: BoolArray) -> FloatArray:
        n, k = eligible.shape
        probs = np.zeros((n, k))
        can = eligible[:, arm]
        probs[can, arm] = 1.0
        if not can.all():
            fallback = ~can
            counts = eligible[fallback].sum(axis=1, keepdims=True)
            probs[fallback] = np.where(eligible[fallback], 1.0 / counts, 0.0)
        return probs

    return policy


def softmax_policy(weights: FloatArray, *, temperature: float = 1.0) -> Policy:
    """Softmax over linear scores. A smooth, everywhere-positive policy."""

    def policy(contexts: FloatArray, eligible: BoolArray) -> FloatArray:
        return _masked_softmax(contexts @ weights, eligible, temperature)

    return policy


def epsilon_greedy_policy(weights: FloatArray, *, epsilon: float = 0.05) -> Policy:
    """Greedy on linear scores, with ``epsilon`` spread over eligible arms.

    Propensities are analytic: ``1 - eps + eps/|E|`` for the greedy arm and
    ``eps/|E|`` elsewhere. This is the whole reason to prefer epsilon-greedy for
    a first deployment -- every logged probability is exact, so nothing
    downstream inherits Monte Carlo error from the sampler.
    """

    def policy(contexts: FloatArray, eligible: BoolArray) -> FloatArray:
        scores = np.where(eligible, contexts @ weights, -np.inf)
        greedy = scores.argmax(axis=1)
        counts = eligible.sum(axis=1, keepdims=True)
        probs = np.where(eligible, epsilon / counts, 0.0)
        probs[np.arange(len(greedy)), greedy] += 1.0 - epsilon
        return probs

    return policy
