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

A closed send window (``brief.channelSend``, #3179) is placed in the same block,
under its own heading: the platform's sentence that the agent's reply will be held,
so it knows before it answers and tells the person. It rides the system side like
the missing parts because it changes only when the window closes or reopens.

A request for a person (``brief.personRequested``, #3190) is placed the same way, under
its own heading: the correspondent sent the binding's exact handoff phrase and the
organization's members were alerted, so the agent tells them a member will take over.
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

#: The scopes this plugin names itself; any other scope is still rendered. A card
#: conversation's brief carries the ``card`` block — the work the card asks for, on the
#: system side so no lane instruction ever enters the session's history.
KNOWN_SCOPES: frozenset[str] = frozenset(
    {"identity", "organization", "folder", "conversation", "card"}
)

#: Said when the brief carried a context this plugin could not read.
UNREADABLE_INSTRUCTION = (
    "The organization's and this place's context could not be read because of a system"
    " problem; tell the person you are working without them."
)

#: Said when the brief named a closed send window this plugin could not read.
CHANNEL_SEND_UNREADABLE_INSTRUCTION = (
    "This conversation's channel may not deliver your reply right now, and the details"
    " could not be read because of a system problem; tell the person your reply may be"
    " delayed."
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


#: What the agent is told when a brief named a request for a person this plugin could not
#: read (#3190) — never a silent gap.
PERSON_REQUESTED_UNREADABLE_INSTRUCTION = (
    "The person may have asked to talk to a person, and the details could not be read"
    " because of a system problem; tell them a member of the team can take over."
)


def _person_requested(person_requested: Any) -> str | None:
    """The instruction a request for a person carries, or ``None`` when the brief named none.

    🔴 Never dropped, like a send window: one this plugin cannot read still reaches the
    agent. Its class is OPEN.
    """
    if person_requested is None:
        return None
    if isinstance(person_requested, dict):
        cls, instruction = person_requested.get("class"), person_requested.get("instruction")
        if isinstance(cls, str) and cls and isinstance(instruction, str) and instruction.strip():
            return _flat(instruction)
    # The shape only — never the instruction's text.
    logger.warning("uffici: a brief named a person request this plugin could not read")
    return PERSON_REQUESTED_UNREADABLE_INSTRUCTION


def _channel_send(channel_send: Any) -> str | None:
    """The instruction a closed send window carries, or ``None`` when the brief named none.

    🔴 Never dropped: a window this plugin cannot read still reaches the agent, as an
    instruction to tell the person. Its class is OPEN — a class this plugin does not know
    still has its instruction placed.
    """
    if channel_send is None:
        return None
    if isinstance(channel_send, dict):
        cls, instruction = channel_send.get("class"), channel_send.get("instruction")
        if isinstance(cls, str) and cls and isinstance(instruction, str) and instruction.strip():
            return _flat(instruction)
    # The shape only — never the instruction's text.
    logger.warning("uffici: a brief named a channel send window this plugin could not read")
    return CHANNEL_SEND_UNREADABLE_INSTRUCTION


def render_context(
    context: Any, channel_send: Any = None, person_requested: Any = None
) -> str | None:
    """The brief's context as one fenced block, or ``None`` when the brief carried none.

    ``None`` means an older companion that sends no context, no send window and no
    request for a person: the caller places nothing and the static platform hint stands
    alone, exactly as before. A closed send window or a request for a person alone still
    renders the block, holding only them.
    """
    send = _channel_send(channel_send)
    person = _person_requested(person_requested)
    if context is None and send is None and person is None:
        return None
    lines = [CONTEXT_OPEN]
    if context is None:
        blocks: list[tuple[str, str, str]] | None = []
        instructions: list[str] | None = []
    else:
        blocks = _blocks(context) if isinstance(context, dict) else None
        instructions = _instructions(context) if isinstance(context, dict) else None
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
    if send is not None:
        if len(lines) > 1:
            lines.append("")
        lines += ["## Channel", f"- {_unfenced(send)}"]
    if person is not None:
        if len(lines) > 1:
            lines.append("")
        lines += ["## Person requested", f"- {_unfenced(person)}"]
    lines.append(CONTEXT_CLOSE)
    return "\n".join(lines)
