"""Counterfactual evaluation of routing policies from logged bandit feedback."""

from sortition.eval.ci import Interval, betting_ci, bootstrap_ci, normal_ci
from sortition.eval.diagnostics import Diagnostics, compute_diagnostics, effective_sample_size
from sortition.eval.estimators import (
    Estimate,
    EstimatorName,
    dm_scores,
    dr_os_scores,
    dr_scores,
    estimate,
    importance_weights,
    ips_scores,
    snips_value,
    switch_dr_scores,
)
from sortition.eval.outcome_model import fit_outcome_model
from sortition.eval.report import Comparison, compare, doctor, evaluate, to_dicts

__all__ = [
    "Comparison",
    "Diagnostics",
    "Estimate",
    "EstimatorName",
    "Interval",
    "betting_ci",
    "bootstrap_ci",
    "compare",
    "compute_diagnostics",
    "dm_scores",
    "doctor",
    "dr_os_scores",
    "dr_scores",
    "effective_sample_size",
    "estimate",
    "evaluate",
    "fit_outcome_model",
    "importance_weights",
    "ips_scores",
    "normal_ci",
    "snips_value",
    "switch_dr_scores",
    "to_dicts",
]
