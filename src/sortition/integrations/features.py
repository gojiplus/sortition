"""Cheap lexical features, computed from the messages the gateway already has.

Deliberately lexical. Embeddings are out of v1 for two reasons: a bundled
encoder would need the raw prompt text, which contradicts the property that
makes this deployable next to sensitive traffic, and it would cost far more than
the routing decision it informs. Caller-supplied embeddings remain a reserved,
optional field on the log schema.

Token counts here are estimates, not tokenizer output. The decision path runs
per request inside someone's proxy, and a real tokenizer costs more than the
accuracy is worth for a routing threshold. Anything that needs exact counts can
read them from the execution row afterwards.
"""

from __future__ import annotations

import re
from typing import Any

# Roughly four characters per token for English prose and code alike. Good to
# about 15%, which is far inside the tolerance of a "is this a long request"
# routing rule.
_CHARS_PER_TOKEN = 4.0

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")


def _text_of(message: dict[str, Any]) -> str:
    """Flatten a chat message's content to plain text.

    Args:
        message: One OpenAI-format chat message.

    Returns:
        The message's text, with multimodal parts reduced to their text spans.
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Multimodal content: keep the text parts, ignore images and audio.
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def extract(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the feature vector a policy sees, and the log records.

    Args:
        messages: The request's chat messages, in OpenAI format.
        tools: Tool definitions attached to the request, if any.
        extra: Caller-supplied features merged in last, so a gateway can add
            tenant priors or a task-type hint without changing this code.

    Returns:
        A flat mapping of feature name to scalar value.
    """
    texts = [_text_of(m) for m in messages]
    joined = "\n".join(texts)
    n_chars = len(joined)

    code_chars = sum(len(m.group()) for m in _CODE_FENCE.finditer(joined))
    code_chars += sum(len(m.group()) for m in _INLINE_CODE.finditer(joined))

    last_user = next(
        (
            t
            for m, t in zip(reversed(messages), reversed(texts), strict=True)
            if m.get("role") == "user"
        ),
        "",
    )

    features: dict[str, Any] = {
        "n_messages": len(messages),
        "n_chars": n_chars,
        "estimated_tokens": round(n_chars / _CHARS_PER_TOKEN, 1),
        "last_user_tokens": round(len(last_user) / _CHARS_PER_TOKEN, 1),
        "code_fraction": round(code_chars / n_chars, 4) if n_chars else 0.0,
        "tools_required": bool(tools),
        "n_tools": len(tools or ()),
        "has_system": any(m.get("role") == "system" for m in messages),
        "is_multi_turn": len(messages) > 2,
    }
    if extra:
        features.update(extra)
    return features
