"""Whether a log can answer the question at all.

An off-policy estimate is only as good as the overlap between the policy that
logged the data and the policy being asked about. When the target policy wants
to do something the logging policy almost never did, the estimator still returns
a number -- one dominated by a handful of enormous importance weights, with a
variance the point estimate does not advertise.

These diagnostics run before any estimate is reported, and they can refuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

# Below this fraction of nominal sample size the estimate is driven by so few
# rows that a point estimate is more misleading than no answer.
MIN_ESS_FRACTION = 0.05
# Fallbacks mean the gateway overrode the sampler; a few percent is normal
# operational noise, more than this and the log no longer describes the policy.
MAX_LEAKAGE_RATE = 0.05


@dataclass(frozen=True)
class Diagnostics:
    """Whether the logged data supports the question being asked of it."""

    n: int
    ess: float
    """Kish effective sample size: ``(sum w)^2 / sum w^2``. The number of
    equally-weighted observations carrying the same information."""

    ess_fraction: float
    max_weight: float
    weight_p99: float
    support_violations: int
    """Rows where the target policy puts mass on an arm outside the logged
    eligible set. There is no counterfactual to borrow on those rows."""

    support_violation_rate: float
    leakage_rate: float
    """Fraction of rows dropped because the gateway's own fallback machinery
    served an arm the sampler did not draw."""

    n_excluded_leakage: int = 0
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def overlap_ok(self) -> bool:
        """Whether a point estimate should be reported at all."""
        return (
            self.ess_fraction >= MIN_ESS_FRACTION
            and self.support_violation_rate == 0.0
            and self.leakage_rate <= MAX_LEAKAGE_RATE
        )

    def explain(self) -> str:
        """Render the diagnostics as human-readable lines.

        Returns:
            A multi-line summary, one concern per line.
        """
        lines = [
            f"n={self.n}  ESS={self.ess:.1f} ({self.ess_fraction:.1%} of n)",
            f"max weight={self.max_weight:.1f}  p99={self.weight_p99:.1f}",
        ]
        if self.support_violations:
            lines.append(
                f"support violations: {self.support_violations} "
                f"({self.support_violation_rate:.1%})"
            )
        if self.n_excluded_leakage:
            lines.append(
                f"leakage: {self.n_excluded_leakage} rows excluded "
                f"({self.leakage_rate:.1%}) -- gateway fallback overrode the sampler"
            )
        lines.extend(f"WARNING: {w}" for w in self.warnings)
        return "\n".join(lines)


def effective_sample_size(weights: FloatArray) -> float:
    """Kish ESS. Equals ``n`` when all weights are equal, 1 when one dominates."""
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    return total**2 / float((weights**2).sum())


def compute_diagnostics(
    weights: FloatArray,
    *,
    target_probs: FloatArray,
    eligible: BoolArray | None = None,
    n_excluded_leakage: int = 0,
) -> Diagnostics:
    """Assess whether these weights can support an estimate."""
    n = int(weights.shape[0])
    ess = effective_sample_size(weights)
    ess_fraction = ess / n if n else 0.0

    violations = 0
    if eligible is not None:
        # Mass assigned to an arm the logging policy could not have chosen.
        off_support = np.where(eligible, 0.0, target_probs).sum(axis=1)
        violations = int((off_support > 1e-9).sum())
    violation_rate = violations / n if n else 0.0

    total_rows = n + n_excluded_leakage
    leakage_rate = n_excluded_leakage / total_rows if total_rows else 0.0

    warnings: list[str] = []
    if ess_fraction < MIN_ESS_FRACTION:
        warnings.append(
            f"effective sample size is {ess_fraction:.1%} of n -- the target "
            "policy is too far from the logging policy for this data to answer"
        )
    if violations:
        warnings.append(
            f"{violations} rows ({violation_rate:.1%}) place target mass outside "
            "the logged eligible set; those counterfactuals are unobservable"
        )
    if leakage_rate > MAX_LEAKAGE_RATE:
        warnings.append(
            f"{leakage_rate:.1%} of rows were dropped to gateway fallback, above "
            f"the {MAX_LEAKAGE_RATE:.0%} threshold -- the log no longer describes "
            "the policy that was configured"
        )
    if float(weights.max(initial=0.0)) > n / 10:
        warnings.append(
            "a single row carries more than 10% of the total weight; the estimate "
            "is effectively an average over a handful of observations"
        )

    return Diagnostics(
        n=n,
        ess=ess,
        ess_fraction=ess_fraction,
        max_weight=float(weights.max(initial=0.0)),
        weight_p99=float(np.quantile(weights, 0.99)) if n else 0.0,
        support_violations=violations,
        support_violation_rate=violation_rate,
        leakage_rate=leakage_rate,
        n_excluded_leakage=n_excluded_leakage,
        warnings=tuple(warnings),
    )
