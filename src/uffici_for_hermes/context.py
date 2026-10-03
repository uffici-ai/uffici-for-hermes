"""The context a brief carries, rendered for Hermes's system side.

The platform composes it — who the agent is, then what the organization and the
places this conversation lives in tell every agent there — and the companion
forwards it unchanged. This module only renders it: ONE fenced block, the same
bytes for the same context, because the block becomes part of Hermes's ephemeral
system prompt and therefore of its agent-cache signature. Anything that varied
between two renders of one context (a timestamp, a dict's key order) would cost
the provider's prompt cache on every turn.

🔴 Nothing is dropped silently. A level whose scope this plugin does not know is
rendered under its name and scope; a part the platform could not include arrives
as an instruction and is rendered as one; and a context this plugin cannot read
at all becomes an instruction too, so the agent tells the person instead of
working as if nothing were missing.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

#: The fence around the whole block. Stripped from every text inside it, so no
#: statement can close the block early.
CONTEXT_OPEN = "<<<uffici-context>>>"
CONTEXT_CLOSE = "<<</uffici-context>>>"

#: The scopes this plugin names itself; any other scope is still rendered.
KNOWN_SCOPES: frozenset[str] = frozenset({"identity", "organization", "folder", "conversation"})

#: Said when the brief carried a context this plugin could not read.
UNREADABLE_INSTRUCTION = (
    "The organization's and this place's context could not be read because of a system"
    " problem; tell the person you are working without them."
)

_SCOPE = re.compile(r"^[a-z][a-z_]{0,31}$")
_FENCES = (CONTEXT_OPEN, CONTEXT_CLOSE)


def _unfenced(text: str) -> str:
    """The text with every fence marker removed — repeatedly, so none can re-form."""
    while any(fence in text for fence in _FENCES):
        for fence in _FENCES:
            text = text.replace(fence, "")
    return text


def _flat(text: str) -> str:
    return " ".join(text.split())


def _blocks(context: dict[str, Any]) -> list[tuple[str, str, str]] | None:
    """``(name, scope, text)`` per block, or ``None`` when the shape is not the wire's."""
    raw = context.get("blocks")
    if not isinstance(raw, list):
        return None
    blocks: list[tuple[str, str, str]] = []
    for block in raw:
        if not isinstance(block, dict):
            return None
        name, scope, text, tokens = (
            block.get("name"),
            block.get("scope"),
            block.get("text"),
            block.get("tokens"),
        )
        if not (isinstance(name, str) and name.strip()):
            return None
        if not (isinstance(scope, str) and _SCOPE.match(scope)):
            return None
        if not (isinstance(text, str) and text):
            return None
        if not (isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0):
            return None
        blocks.append((_flat(name), scope, text))
    return blocks


def _instructions(context: dict[str, Any]) -> list[str] | None:
    raw = context.get("missing", [])
    if not isinstance(raw, list):
        return None
    out: list[str] = []
    for missing in raw:
        if not isinstance(missing, dict):
            return None
        part, cls, instruction = (
            missing.get("part"),
            missing.get("class"),
            missing.get("instruction"),
        )
        if not (isinstance(part, str) and part and isinstance(cls, str) and cls):
            return None
        if not (isinstance(instruction, str) and instruction.strip()):
            return None
        out.append(_flat(instruction))
    return out


def render_context(context: Any) -> str | None:
    """The brief's context as one fenced block, or ``None`` when the brief carried none.

    ``None`` means an older companion that sends no context: the caller places
    nothing and the static platform hint stands alone, exactly as before.
    """
    if context is None:
        return None
    blocks = _blocks(context) if isinstance(context, dict) else None
    instructions = _instructions(context) if isinstance(context, dict) else None
    lines = [CONTEXT_OPEN]
    if blocks is None or instructions is None:
        # The shape only — never a byte of the context, which is tenant text.
        logger.warning("uffici: a brief carried a context this plugin could not read")
        lines += ["## Missing", f"- {UNREADABLE_INSTRUCTION}"]
    else:
        for name, scope, text in blocks:
            heading = f"## {_unfenced(name)}"
            if scope not in KNOWN_SCOPES:
                heading += f" ({scope})"
            lines += [heading, _unfenced(text).strip("\n"), ""]
        if instructions:
            lines.append("## Missing")
            lines += [f"- {_unfenced(i)}" for i in instructions]
    while lines[-1] == "":
        lines.pop()
    lines.append(CONTEXT_CLOSE)
    return "\n".join(lines)
