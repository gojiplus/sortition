"""Thompson sampling that reports the probability with which it chose.

Deliberately written with nothing but ``random`` and ``dataclasses``. This code
is meant to be portable into LiteLLM's ``adaptive_router/bandit.py``, whose
Thompson sampler already draws one Beta variate per arm and takes an argmax but
never records ``P(chosen)``. numpy is not a LiteLLM dependency, so an
implementation that reaches for it could not be contributed upstream.

Why the probability cannot be recovered afterwards: an arm wins when its scored
Beta draw beats every other arm's, which for more than two arms has no closed
form. It has to be estimated at decision time, while the posteriors are in hand,
or not at all.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

DEFAULT_PROPENSITY_SAMPLES = 128


@dataclass(frozen=True)
class Beta:
    """A Beta posterior over one arm's quality."""

    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        """Posterior mean, or 0.5 when the arm has no mass at all."""
        total = self.alpha + self.beta
        return self.alpha / total if total > 0 else 0.5


def _argmax_draw(
    posteriors: dict[str, Beta],
    scores: dict[str, float],
    rng: random.Random,
) -> str:
    """Draw one joint sample and return the winning arm.

    Args:
        posteriors: Beta posterior per arm.
        scores: Deterministic additive score per arm, e.g. a cost preference.
        rng: Source of randomness.

    Returns:
        The arm whose sampled quality plus deterministic score is largest.
    """
    best_arm = ""
    best = float("-inf")
    for arm, posterior in posteriors.items():
        value = rng.betavariate(posterior.alpha, posterior.beta) + scores.get(arm, 0.0)
        if value > best:
            best = value
            best_arm = arm
    return best_arm


def sample_with_propensity(
    posteriors: dict[str, Beta],
    *,
    scores: dict[str, float] | None = None,
    n_samples: int = DEFAULT_PROPENSITY_SAMPLES,
    rng: random.Random | None = None,
) -> tuple[str, float]:
    """Thompson-sample an arm and estimate the probability it was chosen with.

    The first replicate decides, so the distribution over chosen arms is
    identical to plain Thompson sampling. ``n_samples`` further replicates
    estimate how often that arm wins.

    The deciding replicate is counted in the estimate. That guarantees
    ``propensity >= 1 / (n_samples + 1)`` rather than allowing zero, which would
    be self-contradictory -- the arm demonstrably was chosen -- and would put an
    infinite importance weight into every downstream estimate. The cost is an
    upward bias of order ``1 / n_samples``, far smaller than the Monte Carlo
    noise it displaces.

    Args:
        posteriors: Beta posterior per arm. Must be non-empty.
        scores: Optional deterministic additive score per arm.
        n_samples: Replicates used to estimate the propensity. Zero returns a
            propensity of 1.0, which is only honest for a single-arm choice.
        rng: Source of randomness; a seeded ``random.Random`` makes this
            reproducible.

    Returns:
        The chosen arm and its estimated selection probability.

    Raises:
        ValueError: If ``posteriors`` is empty or ``n_samples`` is negative.
    """
    if not posteriors:
        raise ValueError("sample_with_propensity called with no arms")
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")

    generator = rng if rng is not None else random.Random()  # noqa: S311
    weights = scores or {}

    chosen = _argmax_draw(posteriors, weights, generator)
    if len(posteriors) == 1 or n_samples == 0:
        return chosen, 1.0

    wins = 1  # the deciding replicate
    for _ in range(n_samples):
        if _argmax_draw(posteriors, weights, generator) == chosen:
            wins += 1
    return chosen, wins / (n_samples + 1)


def propensities(
    posteriors: dict[str, Beta],
    *,
    scores: dict[str, float] | None = None,
    n_samples: int = DEFAULT_PROPENSITY_SAMPLES,
    rng: random.Random | None = None,
) -> dict[str, float]:
    """Estimate the full selection distribution over arms.

    Used for evaluating a Thompson policy as a *target*, where every arm's
    probability is needed rather than just the one that was chosen.

    Args:
        posteriors: Beta posterior per arm.
        scores: Optional deterministic additive score per arm.
        n_samples: Replicates to draw.
        rng: Source of randomness.

    Returns:
        A probability per arm, summing to 1.

    Raises:
        ValueError: If ``posteriors`` is empty or ``n_samples`` is not positive.
    """
    if not posteriors:
        raise ValueError("propensities called with no arms")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    generator = rng if rng is not None else random.Random()  # noqa: S311
    weights = scores or {}
    counts = dict.fromkeys(posteriors, 0)
    for _ in range(n_samples):
        counts[_argmax_draw(posteriors, weights, generator)] += 1
    return {arm: count / n_samples for arm, count in counts.items()}
