"""Cases with exactly one right answer.

Everything else in this suite measures a policy against a simulator whose best
achievable value is a number nobody can hit exactly, so the assertions are
inequalities and a regression has to be large to show up. These problems are
built so the optimum is obvious and stated in advance: if quality is identical
and one arm costs ten times more, there is one correct policy and the sweep
either finds it or is broken.

They are deliberately easy. That is the point -- a failure here is a defect, not
a research finding.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from sortition.sim import BanditProblem, epsilon_greedy_policy, sample_logs
from sortition.sim.to_frame import to_frame
from sortition.train import split_three_ways, sweep, train

N_CONTEXTS = 400
N_ROWS = 12_000


def _problem(q: np.ndarray, price: np.ndarray, *, seed: int = 0) -> BanditProblem:
    """A problem with the quality and price ladder stated outright.

    Args:
        q: ``(n_arms,)`` expected reward, identical for every context.
        price: ``(n_arms,)`` dollars per unit of request size.
        seed: Seed for the context pool.

    Returns:
        The problem, with cost scaling in the same size dimension `to_frame`
        renders as ``n_tokens``.
    """
    rng = np.random.default_rng(seed)
    contexts = rng.standard_normal((N_CONTEXTS, 6))
    size = np.exp(0.5 * contexts[:, 0])
    return BanditProblem(
        contexts=contexts,
        q=np.tile(q, (N_CONTEXTS, 1)),
        cost=price[None, :] * size[:, None],
        eligible=np.ones((N_CONTEXTS, len(q)), dtype=bool),
        arms=tuple(f"arm-{i}" for i in range(len(q))),
    )


def _logs(problem: BanditProblem, *, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    behavior = epsilon_greedy_policy(
        rng.standard_normal((6, problem.n_arms)), epsilon=0.5
    )
    return to_frame(
        sample_logs(problem, behavior, N_ROWS, seed=seed),
        problem,
        fallback_rate=0.0,
        seed=seed + 1,
    )


def _chosen_arms(policy, problem: BanditProblem) -> np.ndarray:
    """The arm this policy would pick for every context in the pool."""
    from sortition.decide.tree import discount_matrix

    features = [
        {
            "n_tokens": float(500.0 * np.exp(0.5 * x[0])),
            "code_fraction": float(1.0 / (1.0 + np.exp(-x[1]))),
            "context_tokens": float(abs(x[2]) * 1000.0),
            "tools_required": bool(x[3] > 0.0),
        }
        for x in problem.contexts
    ]
    quality, cost = policy.score_matrix(features, problem.eligible)
    scores = discount_matrix(quality, cost, problem.eligible, policy.cost_weight)
    k = scores.shape[1]
    return (k - 1) - scores[:, ::-1].argmax(axis=1)


class TestEqualQualityDifferentPrice:
    """Same outcome either way, one arm ten times dearer.

    The optimum is obvious: take the cheap arm, save ninety percent, lose
    nothing. What the sweep can *prove* is a different question, and these tests
    are about the difference between the two.

    A non-inferiority margin fires only when it exceeds the half-width of the
    interval on the paired difference. Here the true difference is exactly zero,
    so the interval sits on zero and its lower end is minus the half-width --
    which is 0.034 at 12,000 logged rows, 0.018 at 40,000 and 0.0098 at 120,000.
    Asking for a margin of 0.01 on a small log is asking the log to prove
    something it does not contain, and the sweep is right to refuse.

    Refusing silently would be the defect, because "the dear arm is worth it" and
    "this log cannot tell" look identical from the outside. So the contract under
    test is that the sweep names the margin that would clear and what it buys.
    """

    def test_it_names_the_margin_that_would_unlock_the_saving(self) -> None:
        problem = _problem(np.array([0.5, 0.5]), np.array([0.001, 0.01]))
        fit, tune, _ = split_three_ways(_logs(problem), seed=0)
        result = sweep(fit, tune, seed=0)

        assert result.chosen.cost_weight == 0.0
        offer = result.next_best
        assert offer is not None
        assert offer.tolerance_required > 0.0
        assert offer.saving > 0.0

    def test_taking_that_margin_takes_the_cheaper_arm(self) -> None:
        # The end of the argument: the number the sweep reported is one that
        # actually works, and using it routes to the cheap arm.
        problem = _problem(np.array([0.5, 0.5]), np.array([0.001, 0.01]))
        fit, tune, _ = split_three_ways(_logs(problem), seed=0)
        offer = sweep(fit, tune, seed=0).next_best
        assert offer is not None

        result = sweep(fit, tune, tolerance=offer.tolerance_required, seed=0)

        assert result.chosen.cost_weight > 0.0
        assert result.chosen.saving > 0.0
        picked = _chosen_arms(result.policy, problem)
        assert (picked == 0).mean() > 0.95, (picked == 1).mean()

    def test_the_bill_falls_to_the_cheap_arm_s_bill(self) -> None:
        problem = _problem(np.array([0.5, 0.5]), np.array([0.001, 0.01]))
        fit, tune, _ = split_three_ways(_logs(problem), seed=0)
        offer = sweep(fit, tune, seed=0).next_best
        assert offer is not None
        result = sweep(fit, tune, tolerance=offer.tolerance_required, seed=0)

        picked = _chosen_arms(result.policy, problem)
        greedy = float(problem.cost[np.arange(len(picked)), picked].mean())
        floor = float(problem.cost[:, 0].mean())
        assert greedy < 1.5 * floor, (greedy, floor)


class TestDominatedArm:
    """An arm that is both worse and dearer is never the right answer."""

    @pytest.mark.parametrize("cost_weight", [0.0, 0.5, 4.0])
    def test_it_is_almost_never_chosen(self, cost_weight: float) -> None:
        # "Almost" is doing real work. The true gap is 0.4 and the four logged
        # features carry no signal at all here, so the right answer is arm-0 on
        # every request -- but the model estimates quality from 12,000 Bernoulli
        # draws and its predictions scatter, which flips the ranking on about
        # 0.2% of contexts at cost_weight=0.
        #
        # Early stopping on a validation split removes those (0.2% -> 0.0%, and
        # it stops at 38 trees rather than 300) but costs accuracy where the
        # features do carry signal: on the contextual simulator it takes the
        # mean absolute error against the true q from 0.182 to 0.194 and runs to
        # 1060 trees. Trading real accuracy for this is the wrong bargain, so
        # the threshold records what the model actually guarantees.
        problem = _problem(np.array([0.7, 0.3]), np.array([0.001, 0.02]))
        logs = _logs(problem)
        policy = train(logs, cost_weight=cost_weight, seed=0).policy

        picked = _chosen_arms(policy, problem)
        assert (picked == 0).mean() > 0.99, (picked == 1).mean()


class TestNoiselessCost:
    """When the bill is exactly price times size, the model should say so."""

    def test_predicted_cost_recovers_the_price_ladder(self) -> None:
        price = np.array([0.002, 0.02])
        problem = _problem(np.array([0.5, 0.5]), price)
        logs = _logs(problem)
        # Replace the realized cost with its noiseless expectation, so any error
        # left is the model's and not the draw's.
        arm_index = {a: i for i, a in enumerate(problem.arms)}
        idx = np.array([arm_index[a] for a in logs.get_column("chosen_arm").to_list()])
        sizes = np.array(
            [f["n_tokens"] / 500.0 for f in logs.get_column("features").to_list()]
        )
        exact = price[idx] * sizes
        policy = train(logs.with_columns(cost_usd=pl.Series(exact)), seed=0).policy

        for tokens in (400.0, 1_500.0):
            predicted = policy.predict_cost({"n_tokens": tokens}, policy.arms)
            for i, arm in enumerate(policy.arms):
                assert predicted[arm] == pytest.approx(
                    price[i] * tokens / 500.0, rel=0.2
                ), (tokens, arm, predicted[arm])
