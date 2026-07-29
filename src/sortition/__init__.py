"""Counterfactual evaluation for LLM routing policies."""

import logging
from importlib.metadata import PackageNotFoundError, version

from sortition.schema import (
    SCHEMA_VERSION,
    Decision,
    DecisionRow,
    ExecutionRow,
    OutcomeRow,
    PolicyArtifact,
)

# A library must not configure logging for the application importing it; the
# NullHandler keeps "no handlers could be found" warnings away while leaving
# every decision about output to the caller.
logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    __version__ = version("sortition")
except PackageNotFoundError:  # pragma: no cover - only when running from source
    __version__ = "0.0.0"

__all__ = [
    "SCHEMA_VERSION",
    "Decision",
    "DecisionRow",
    "ExecutionRow",
    "OutcomeRow",
    "PolicyArtifact",
    "__version__",
]
