"""Turning a brief's entries into the text one harness turn reads.

🔴 ONE rule, used for both the delta and the opener. The two differ in what the
platform puts in them — a delta omits the answering agent's own entries, an
opener includes them — never in how a line is written. A second rule would let
the same entry read differently depending on which frame carried it, and the
model would have no way to tell that they were the same message.

Nothing here composes: every line is the platform's own text, prefixed with the
author the platform resolved. The plugin is a format translator, and an opinion
inserted here would be an opinion nobody could audit.
"""

from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from typing import Any

#: A gap this long between entries earns a timestamp line. Below it the reading
#: order is obvious and a timestamp per entry is noise the model pays for.
TIMESTAMP_GAP = timedelta(minutes=10)

#: What a tombstone renders as. The platform sends the entry so the model can
#: see that something WAS said and withdrawn — a silent omission would leave a
#: reply that answers a message the transcript no longer shows.
TOMBSTONE_LINE = "(message deleted)"


def _label_for(author: dict[str, Any]) -> str:
    """``<label> (@handle)`` for an agent, ``<label>`` for anyone else.

    🔴 The handle is the whole point for an agent: the platform's wake rule is
    `@slug` mentions, so a model that cannot see an agent's handle cannot
    address it and cannot hand work over.
    """
    label = str(author.get("label") or "").strip()
    handle = str(author.get("handle") or "").strip()
    if not label:
        # The platform always resolves a label; an empty one is still rendered
        # as something rather than as an unattributed line.
        label = str(author.get("type") or "someone")
    return f"{label} ({handle})" if handle else label


def _parse_at(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        # A timestamp we cannot read costs a gap line, never the entry.
        return None


def _attachment_lines(entry: dict[str, Any]) -> list[str]:
    """One reference line per attachment.

    By REFERENCE, never by content: an attachment is a path the agent's own
    tools open, and inlining it here would put a file's bytes into every turn
    that merely mentions it.
    """
    out: list[str] = []
    for attachment in entry.get("attachments") or []:
        if not isinstance(attachment, dict):
            continue
        name = str(attachment.get("name") or "").strip()
        ref = str(attachment.get("ref") or "").strip()
        if name or ref:
            out.append(f"  [attachment] {name or ref}" + (f" — {ref}" if name and ref else ""))
    return out


def render_entry(entry: dict[str, Any]) -> list[str]:
    """One entry as the lines it contributes."""
    author = entry.get("author") or {}
    kind = str(entry.get("kind") or "").strip()
    text = str(entry.get("text") or "")

    if kind == "tool_line":
        # 🔴 Verbatim. A tool line is already the platform's own rendering of
        # what an agent did; re-labelling it with an author would attribute the
        # tool's output to the person who triggered it.
        lines = [text] if text else []
    elif kind == "tombstone":
        lines = [f"{_label_for(author)}: {TOMBSTONE_LINE}"]
    elif kind == "message" or not kind:
        lines = [f"{_label_for(author)}: {text}"]
    else:
        # 🔴 An unknown kind is LABELLED, never dropped and never fatal
        # (decision 44). Wave 4's summary kind must arrive with no version bump,
        # and a reader that refused it would make that arrival a breaking change.
        lines = [f"{_label_for(author)} [{kind}]: {text}"]

    return [*lines, *_attachment_lines(entry)]


def render_entries(entries: Iterable[dict[str, Any]]) -> str:
    """A run of entries as one block of text."""
    out: list[str] = []
    previous: datetime | None = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        at = _parse_at(entry.get("at"))
        if at is not None and previous is not None and at - previous >= TIMESTAMP_GAP:
            out.append(f"[{at.isoformat()}]")
        if at is not None:
            previous = at
        out.extend(render_entry(entry))
    return "\n".join(out)


def render_reason(
    reason: str, cause: dict[str, Any] | None, entries: Sequence[dict[str, Any]]
) -> str:
    """The one line saying why this turn is happening.

    It names the reason the platform gave and, when the causing entry is in the
    delta, who wrote it — so the model knows which message it is answering when
    several arrived together.
    """
    reason = str(reason or "").strip() or "unknown"
    cause_id = str((cause or {}).get("entryId") or "")
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("id") or "") == cause_id:
            who = _label_for(entry.get("author") or {})
            return f"[Uffici] You were woken because: {reason} (by {who})."
    return f"[Uffici] You were woken because: {reason}."


def cause_author(cause: dict[str, Any] | None, entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The author of the entry that caused this turn.

    🔴 The CAUSE's author, never "the last human in the transcript". A brief can
    be caused by another agent's message or by a correspondent outside the
    organization, and Hermes keys authorization and prompt-building off the
    sender — so naming the wrong one misattributes the turn.
    """
    cause = cause or {}
    cause_id = str(cause.get("entryId") or "")
    author_id = str(cause.get("authorId") or "")
    for entry in entries:
        if isinstance(entry, dict) and str(entry.get("id") or "") == cause_id:
            return dict(entry.get("author") or {})
    # The cause may be outside the delta (an entry the agent already answered
    # for, or one the platform bounded away). The id it gave is still the truth.
    return {"id": author_id, "label": "", "type": "member", "self": False}
