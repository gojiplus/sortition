"""Synthetic contextual bandits with exactly known policy values.

Estimators cannot be validated against production traffic: the counterfactual is
by construction unobserved, so there is nothing to check an estimate against.
This module is the oracle instead. It builds problems whose true policy value is
computable in closed form, which turns "is the estimator correct?" into an
assertion.
"""

from sortition.sim.generator import (
    BanditProblem,
    LoggedData,
    Policy,
    constant_policy,
    epsilon_greedy_policy,
    make_problem,
    sample_logs,
    softmax_policy,
    uniform_policy,
)

__all__ = [
    "BanditProblem",
    "LoggedData",
    "Policy",
    "constant_policy",
    "epsilon_greedy_policy",
    "make_problem",
    "sample_logs",
    "softmax_policy",
    "uniform_policy",
]
