"""The simulator is the oracle every other test trusts, so it is tested first.

If ``BanditProblem.value`` does not agree with what sampling from the problem
actually produces, every estimator test downstream is measuring the wrong thing.
"""

from __future__ import annotations

import numpy as np
import pytest

from sortition.sim import (
    BanditProblem,
    constant_policy,
    epsilon_greedy_policy,
    make_problem,
    sample_logs,
    softmax_policy,
    uniform_policy,
)


class TestExactValue:
    def test_analytic_value_matches_realized_rewards(
        self, problem: BanditProblem
    ) -> None:
        policy = uniform_policy()
        logs = sample_logs(problem, policy, 400_000, seed=21)
        # On-policy, the plain mean of realized rewards estimates the same
        # quantity that value() computes in closed form.
        assert float(logs.reward.mean()) == pytest.approx(
            problem.value(policy), abs=0.004
        )

    def test_analytic_cost_matches_realized_cost(self, problem: BanditProblem) -> None:
        policy = uniform_policy()
        logs = sample_logs(problem, policy, 400_000, seed=22)
        assert float(logs.cost.mean()) == pytest.approx(
            problem.cost_value(policy), rel=0.03
        )

    def test_value_has_no_sampling_noise(self, problem: BanditProblem) -> None:
        # The finite context pool is what makes the oracle exact; calling it
        # twice must give bit-identical answers.
        policy = softmax_policy(np.random.default_rng(0).standard_normal((6, 4)))
        assert problem.value(policy) == problem.value(policy)

    def test_cost_varies_with_request_size(self, problem: BanditProblem) -> None:
        # A bill is price-per-token times tokens. A cost that is constant per arm
        # makes the cheap-to-premium ladder the only thing a router can price,
        # and no per-request cost model has anything to learn from it.
        assert problem.cost.shape == (problem.n_contexts, problem.n_arms)
        premium = problem.cost[:, -1]
        assert premium.max() / premium.min() > 2.0

    def test_the_cost_ladder_survives_within_a_request(
        self, problem: BanditProblem
    ) -> None:
        # Size scales every arm together, so the premium arm is dearer than the
        # budget arm on every single request, not merely on average.
        assert bool((np.diff(problem.cost, axis=1) > 0).all())

    def test_policies_are_distinguishable(self, problem: BanditProblem) -> None:
        # If every policy had the same value there would be nothing to estimate.
        values = [problem.value(constant_policy(a)) for a in range(problem.n_arms)]
        assert max(values) - min(values) > 0.02


class TestPolicyContract:
    @pytest.mark.parametrize("ineligible_rate", [0.0, 0.4])
    def test_probabilities_are_distributions(self, ineligible_rate: float) -> None:
        problem = make_problem(
            n_contexts=200, n_arms=5, seed=3, ineligible_rate=ineligible_rate
        )
        rng = np.random.default_rng(0)
        policies = [
            uniform_policy(),
            constant_policy(1),
            softmax_policy(rng.standard_normal((6, 5))),
            epsilon_greedy_policy(rng.standard_normal((6, 5)), epsilon=0.1),
        ]
        for policy in policies:
            probs = policy(problem.contexts, problem.eligible)
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-12)
            assert (probs >= 0).all()
            # No built-in policy may put mass on an arm the hard filter removed.
            assert (probs[~problem.eligible] == 0).all()

    def test_epsilon_greedy_propensities_are_analytic(self) -> None:
        # The reason to prefer epsilon-greedy for a first deployment: the logged
        # probability is exact, so nothing downstream inherits sampler noise.
        problem = make_problem(n_contexts=100, n_arms=4, seed=4)
        weights = np.random.default_rng(0).standard_normal((6, 4))
        epsilon = 0.2
        probs = epsilon_greedy_policy(weights, epsilon=epsilon)(
            problem.contexts, problem.eligible
        )

        greedy = (problem.contexts @ weights).argmax(axis=1)
        expected_greedy = 1.0 - epsilon + epsilon / 4
        np.testing.assert_allclose(
            probs[np.arange(len(greedy)), greedy], expected_greedy, rtol=1e-12
        )

    def test_constant_policy_falls_back_where_ineligible(self) -> None:
        problem = make_problem(n_contexts=200, n_arms=4, seed=5, ineligible_rate=0.6)
        probs = constant_policy(3)(problem.contexts, problem.eligible)
        blocked = ~problem.eligible[:, 3]
        assert blocked.any()
        assert (probs[blocked, 3] == 0).all()
        np.testing.assert_allclose(probs[blocked].sum(axis=1), 1.0, rtol=1e-12)


class TestSampling:
    def test_logged_propensity_matches_the_policy(self, problem: BanditProblem) -> None:
        policy = epsilon_greedy_policy(
            np.random.default_rng(0).standard_normal((6, 4)), epsilon=0.25
        )
        logs = sample_logs(problem, policy, 5_000, seed=23)
        probs = policy(logs.contexts, logs.eligible)
        np.testing.assert_allclose(
            logs.propensity, probs[np.arange(logs.n), logs.action], rtol=1e-12
        )

    def test_action_frequencies_match_propensities(
        self, problem: BanditProblem
    ) -> None:
        policy = uniform_policy()
        logs = sample_logs(problem, policy, 200_000, seed=24)
        empirical = np.bincount(logs.action, minlength=problem.n_arms) / logs.n
        expected = policy(logs.contexts, logs.eligible).mean(axis=0)
        np.testing.assert_allclose(empirical, expected, atol=0.01)

    def test_never_samples_an_ineligible_arm(self) -> None:
        problem = make_problem(n_contexts=200, n_arms=5, seed=6, ineligible_rate=0.5)
        logs = sample_logs(problem, uniform_policy(), 20_000, seed=25)
        assert logs.eligible[np.arange(logs.n), logs.action].all()

    def test_propensities_are_strictly_positive(self, problem: BanditProblem) -> None:
        logs = sample_logs(problem, constant_policy(0), 2_000, seed=26)
        assert (logs.propensity > 0).all()

    def test_rewards_are_bounded_and_costs_are_not(
        self, problem: BanditProblem
    ) -> None:
        logs = sample_logs(problem, uniform_policy(), 10_000, seed=27)
        assert set(np.unique(logs.reward)) <= {0.0, 1.0}
        assert logs.cost.min() > 0.0
        # Right-skewed: the mean sits above the median, which is what forces the
        # bootstrap path rather than a bounded-mean interval.
        assert float(logs.cost.mean()) > float(np.median(logs.cost))
