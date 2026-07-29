"""Correctness of the estimators, checked against a known ground truth.

There is no way to validate an off-policy estimator on production traffic: the
counterfactual is unobserved by construction, so there is nothing to compare an
estimate to. Every test here runs against ``sortition.sim``, where the true
policy value is computable in closed form.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.tree import DecisionTreeRegressor

from sortition.eval import (
    estimate,
    fit_outcome_model,
    importance_weights,
    ips_scores,
    snips_value,
)
from sortition.sim import (
    BanditProblem,
    constant_policy,
    epsilon_greedy_policy,
    make_problem,
    sample_logs,
    uniform_policy,
)


def _memorizer(seed: int) -> DecisionTreeRegressor:
    """An outcome model with enough capacity to memorize its training rows."""
    return DecisionTreeRegressor(random_state=seed)


class TestUnbiasedness:
    """IPS is unbiased for the target policy value; that is its whole claim."""

    def test_ips_recovers_truth(
        self, problem: BanditProblem, weights: tuple[np.ndarray, np.ndarray]
    ) -> None:
        wb, wt = weights
        behavior = epsilon_greedy_policy(wb, epsilon=0.3)
        target = epsilon_greedy_policy(wt, epsilon=0.1)
        truth = problem.value(target)

        logs = sample_logs(problem, behavior, 200_000, seed=11)
        est = estimate(
            action=logs.action,
            reward=logs.reward,
            propensity=logs.propensity,
            target_probs=target(logs.contexts, logs.eligible),
            estimator="ips",
            ci_method="none",
        )
        # At this sample size the standard error is tiny; three of them is a
        # tight band around the analytic value.
        assert abs(est.value - truth) < 3 * est.se

    def test_ips_recovers_a_constant_policy(self, problem: BanditProblem) -> None:
        # "What if we always sent everything to the premium arm?" is the
        # question teams actually ask, and it is the hardest for overlap.
        target = constant_policy(2)
        truth = problem.value(target)
        logs = sample_logs(problem, uniform_policy(), 200_000, seed=12)

        est = estimate(
            action=logs.action,
            reward=logs.reward,
            propensity=logs.propensity,
            target_probs=target(logs.contexts, logs.eligible),
            estimator="ips",
            ci_method="none",
        )
        assert abs(est.value - truth) < 3 * est.se

    def test_estimates_cost_as_well_as_outcome(self, problem: BanditProblem) -> None:
        # Cost is the other half of the routing question and is unbounded.
        target = constant_policy(0)
        truth = problem.cost_value(target)
        logs = sample_logs(problem, uniform_policy(), 100_000, seed=13)

        est = estimate(
            action=logs.action,
            reward=logs.cost,
            propensity=logs.propensity,
            target_probs=target(logs.contexts, logs.eligible),
            estimator="ips",
            metric="cost_usd",
            bounded=False,
            ci_method="none",
        )
        assert abs(est.value - truth) < 3 * est.se


class TestDegeneracy:
    """When the target policy is the logging policy, the answer is the data."""

    @settings(deadline=None, max_examples=25)
    @given(
        n_arms=st.integers(min_value=2, max_value=6),
        epsilon=st.floats(min_value=0.05, max_value=1.0),
        seed=st.integers(min_value=0, max_value=500),
    )
    def test_ips_equals_sample_mean_on_policy(self, n_arms: int, epsilon: float, seed: int) -> None:
        problem = make_problem(n_contexts=80, n_arms=n_arms, seed=seed)
        rng = np.random.default_rng(seed)
        policy = epsilon_greedy_policy(rng.standard_normal((6, n_arms)), epsilon=epsilon)
        logs = sample_logs(problem, policy, 400, seed=seed)

        target_probs = policy(logs.contexts, logs.eligible)
        weights = importance_weights(logs.action, logs.propensity, target_probs)
        scores = ips_scores(logs.action, logs.reward, logs.propensity, target_probs)

        # Every weight is exactly 1, so IPS collapses to the plain mean. Any
        # drift here means propensities and target probabilities disagree about
        # the same policy.
        np.testing.assert_allclose(weights, 1.0, rtol=1e-12)
        assert scores.mean() == pytest.approx(logs.reward.mean(), rel=1e-12)

    def test_snips_equals_sample_mean_on_policy(self, problem: BanditProblem) -> None:
        policy = uniform_policy()
        logs = sample_logs(problem, policy, 5_000, seed=14)
        value = snips_value(
            logs.action,
            logs.reward,
            logs.propensity,
            policy(logs.contexts, logs.eligible),
        )
        assert value == pytest.approx(float(logs.reward.mean()), rel=1e-12)


class TestVarianceOrdering:
    """DR exists to beat IPS on variance. Assert that it does."""

    def test_dr_is_tighter_than_ips(
        self, problem: BanditProblem, weights: tuple[np.ndarray, np.ndarray]
    ) -> None:
        wb, wt = weights
        behavior = epsilon_greedy_policy(wb, epsilon=0.3)
        target = epsilon_greedy_policy(wt, epsilon=0.1)
        logs = sample_logs(problem, behavior, 20_000, seed=15)
        target_probs = target(logs.contexts, logs.eligible)
        q_hat = fit_outcome_model(logs.contexts, logs.action, logs.reward, problem.n_arms, seed=0)

        common = {
            "action": logs.action,
            "reward": logs.reward,
            "propensity": logs.propensity,
            "target_probs": target_probs,
            "ci_method": "none",
        }
        ips = estimate(**common, estimator="ips")  # type: ignore[arg-type]
        dr = estimate(**common, estimator="dr", q_hat=q_hat)  # type: ignore[arg-type]
        assert dr.se < ips.se


class TestSupportViolations:
    """A counterfactual outside the logged support is not an estimate."""

    def test_violation_is_detected_and_blocks_trust(self) -> None:
        problem = make_problem(n_contexts=300, n_arms=4, seed=2, ineligible_rate=0.5)
        logs = sample_logs(problem, uniform_policy(), 4_000, seed=16)

        # A target that ignores eligibility entirely: uniform over all arms,
        # including ones the hard filter excluded.
        n, k = logs.eligible.shape
        target_probs = np.full((n, k), 1.0 / k)

        est = estimate(
            action=logs.action,
            reward=logs.reward,
            propensity=logs.propensity,
            target_probs=target_probs,
            estimator="ips",
            eligible=logs.eligible,
            ci_method="none",
        )
        assert est.diagnostics.support_violations > 0
        assert not est.trustworthy
        assert any("outside the logged eligible set" in w for w in est.diagnostics.warnings)

    def test_respecting_eligibility_reports_no_violations(self, problem: BanditProblem) -> None:
        logs = sample_logs(problem, uniform_policy(), 2_000, seed=17)
        target = constant_policy(1)
        est = estimate(
            action=logs.action,
            reward=logs.reward,
            propensity=logs.propensity,
            target_probs=target(logs.contexts, logs.eligible),
            estimator="ips",
            eligible=logs.eligible,
            ci_method="none",
        )
        assert est.diagnostics.support_violations == 0


class TestLeakage:
    """Rows where the gateway overrode the sampler are reported, not hidden."""

    def test_leakage_rate_is_surfaced(self, problem: BanditProblem) -> None:
        logs = sample_logs(problem, uniform_policy(), 1_000, seed=18)
        est = estimate(
            action=logs.action,
            reward=logs.reward,
            propensity=logs.propensity,
            target_probs=uniform_policy()(logs.contexts, logs.eligible),
            estimator="ips",
            n_excluded_leakage=200,
            ci_method="none",
        )
        assert est.diagnostics.leakage_rate == pytest.approx(200 / 1200)
        assert not est.trustworthy
        assert any("fallback" in w for w in est.diagnostics.warnings)


class TestCrossFitting:
    """Why the outcome model must be fit out-of-fold.

    The danger is not what one might expect. With exact logged propensities --
    which is the regime sortition is built for, since the sampler recorded them
    -- doubly robust estimation keeps the *point estimate* near truth even when
    the outcome model has memorized its training rows. What collapses is the
    *interval*: in-sample residuals are artificially small, so the correction
    term is too quiet, the standard error understates the real sampling
    variability, and a nominal 95% interval covers far less often than that.

    An overconfident interval is worse than a wide one, because nothing about it
    looks wrong.
    """

    def test_in_sample_fitting_shrinks_residuals(self, problem: BanditProblem) -> None:
        logs = sample_logs(problem, uniform_policy(), 1_500, seed=19)
        rows = np.arange(logs.n)

        def mean_abs_residual(cross_fit: bool) -> float:
            q = fit_outcome_model(
                logs.contexts,
                logs.action,
                logs.reward,
                problem.n_arms,
                seed=0,
                cross_fit=cross_fit,
                make_regressor=_memorizer,
            )
            return float(np.abs(logs.reward - q[rows, logs.action]).mean())

        # The mechanism itself: a memorizing model fits its own training rows
        # far better than held-out ones.
        assert mean_abs_residual(cross_fit=False) < 0.85 * mean_abs_residual(cross_fit=True)

    @pytest.mark.slow
    def test_in_sample_fitting_destroys_interval_coverage(
        self, n_replications: int, weights: tuple[np.ndarray, np.ndarray]
    ) -> None:
        problem = make_problem(n_contexts=200, n_arms=4, seed=0)
        wb, wt = weights
        behavior = epsilon_greedy_policy(wb, epsilon=0.3)
        target = epsilon_greedy_policy(wt, epsilon=0.1)
        truth = problem.value(target)
        reps = min(n_replications, 150)

        def coverage(cross_fit: bool) -> float:
            covered = 0
            for s in range(reps):
                logs = sample_logs(problem, behavior, 1_500, seed=500 + s)
                q = fit_outcome_model(
                    logs.contexts,
                    logs.action,
                    logs.reward,
                    problem.n_arms,
                    seed=0,
                    cross_fit=cross_fit,
                    make_regressor=_memorizer,
                )
                est = estimate(
                    action=logs.action,
                    reward=logs.reward,
                    propensity=logs.propensity,
                    target_probs=target(logs.contexts, logs.eligible),
                    q_hat=q,
                    estimator="dr",
                    ci_method="normal",
                )
                assert est.interval is not None
                covered += est.interval.covers(truth)
            return covered / reps

        assert coverage(cross_fit=True) >= 0.90
        # The regression guard: if someone drops cross-fitting, this fails.
        assert coverage(cross_fit=False) < 0.90
