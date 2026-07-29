"""Gateway integrations. Sortition holds no credentials and executes no calls.

``SortitionLogger`` is imported lazily because it subclasses LiteLLM's
``CustomLogger``, which LiteLLM requires by ``isinstance`` and which therefore
cannot be faked. Everything else here works without a gateway installed, so that
converting exported logs never needs one.
"""

from typing import TYPE_CHECKING, Any

from sortition.integrations.adapters import read_decision, to_log_row
from sortition.integrations.features import extract
from sortition.integrations.litellm_plugin import SIGNAL_KEY, SortitionPlugin

if TYPE_CHECKING:
    from sortition.integrations.litellm_logger import SortitionLogger

__all__ = [
    "SIGNAL_KEY",
    "SortitionLogger",
    "SortitionPlugin",
    "extract",
    "read_decision",
    "to_log_row",
]


def __getattr__(name: str) -> Any:
    """Resolve ``SortitionLogger`` only when it is actually used.

    Args:
        name: Attribute being looked up.

    Returns:
        The requested attribute.

    Raises:
        AttributeError: If the name is not exported by this package.
    """
    if name == "SortitionLogger":
        from sortition.integrations.litellm_logger import SortitionLogger

        return SortitionLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
