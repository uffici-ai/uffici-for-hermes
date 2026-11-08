"""Hermes's durable input queue (Hermes 0.22+), read and acted on as the platform requires.

From 0.22 Hermes admits every inbound message into a durable FIFO per session (its
``session_admissions`` table in ``state.db``) before anything runs it, and keeps that FIFO
across a restart: a row still ``queued`` is run again by the new process, and a row whose
turn had ``started`` becomes ``unknown`` and PAUSES the conversation until someone sends
``/discard`` (``hermes_state_runtime.recover_session_inputs``, ``gateway/session_ingress.py``).

🔴 On Uffici the platform's record of a request is the one answered (#2273): the companion
redelivers every request a restart interrupted. So, before a brief is handed to Hermes:

- a turn Hermes LOST at a restart is discarded through Hermes's own ``/discard`` path
  (``discard_lost_turns``: never re-run) — the companion's redelivery answers it, and a
  paused conversation would otherwise answer nothing but a pause notice;
- a brief Hermes still holds QUEUED from before the restart is not handed over again —
  Hermes runs it, and handing it over too would answer the person twice.

Each delivery is named ``<requestId>~<token>`` (``message_id``), a fresh token per brief, so a
redelivery is never refused as a changed retry of an id Hermes already recorded
(``admission_conflict``), and the request is always recoverable from the name.

Every read here is of a Hermes internal and NEVER fatal: an older Hermes without the
queue, or a store that raises, reads as "nothing held", which is today's behaviour. The
names this module relies on are pinned against the real Hermes by ``dev/hermes_contract.py``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DELIVERY_SEPARATOR",
    "HeldQueue",
    "delivery_id",
    "new_delivery_token",
    "read_queue",
    "request_of",
]

#: Joins a request id and its delivery's token in a brief's ``message_id``.
DELIVERY_SEPARATOR = "~"

#: A delivery's name: only an exact ``~<8 hex>`` suffix is a token, so a request id that
#: itself contains the separator still maps back to itself.
_DELIVERY = re.compile(r"^(?P<request>.+)~[0-9a-f]{8}$")

#: The rows still owed a turn. ``started`` is a turn running in this process.
_PENDING = frozenset({"queued", "started"})


def new_delivery_token() -> str:
    """A fresh token for one delivery of a brief."""
    return secrets.token_hex(4)


def delivery_id(request_id: str, token: str) -> str:
    """The ``message_id`` a brief carries into Hermes: its request and this delivery."""
    return f"{request_id}{DELIVERY_SEPARATOR}{token}"


def request_of(message_id: Any) -> str:
    """The request a brief's ``message_id`` names (the whole id when it carries no token)."""
    text = str(message_id or "")
    named = _DELIVERY.match(text)
    return named.group("request") if named else text


@dataclass
class HeldQueue:
    """What Hermes's queue holds for one conversation's session."""

    authority: Any
    session_id: str
    rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def lost(self) -> list[dict[str, Any]]:
        """Turns a restart cut: ``unknown``, pausing the conversation until discarded."""
        return [row for row in self.rows if row.get("status") == "unknown"]

    def held_deliveries(self, request_id: str) -> set[str]:
        """The deliveries of this request Hermes still owes a turn (queued, or running here)."""
        return {
            str(row.get("request_id"))
            for row in self.rows
            if row.get("status") in _PENDING and request_of(row.get("request_id")) == request_id
        }

    def owes(self, deliveries: set[str]) -> bool:
        """Whether Hermes still owes a turn for any of these deliveries."""
        return any(
            row.get("status") in _PENDING and str(row.get("request_id")) in deliveries
            for row in self.rows
        )

    def holds(self, request_id: str) -> bool:
        """Whether Hermes still owes a turn for this request (queued, or running here)."""
        return any(
            row.get("status") in _PENDING and request_of(row.get("request_id")) == request_id
            for row in self.rows
        )

    async def discard_lost(self) -> bool:
        """Discard every lost turn through Hermes's own ``/discard`` path. Never fatal."""
        try:
            from gateway.slash_commands_session import (  # type: ignore[import-not-found]
                discard_lost_turns,
            )

            await discard_lost_turns(self.authority, self.session_id)
        except Exception as exc:  # noqa: BLE001 — a harness internal; the brief still goes
            logger.warning(
                "uffici: Hermes's lost turn could not be discarded (%s); its conversation "
                "may stay paused",
                type(exc).__name__,
            )
            return False
        return True


async def read_queue(
    adapter: Any, session_key: str, *, pending_only: bool = True
) -> HeldQueue | None:
    """Hermes's queue for the session ``session_key`` names, or None when there is none.

    None means "nothing to reconcile": an older Hermes, no session yet, or a store that
    could not be read (logged). The table is read off the event loop, as Hermes reads it.
    """
    if not session_key:
        return None
    try:
        from gateway.session_authorities import active_authority  # type: ignore[import-not-found]
        from hermes_state_runtime import list_session_admissions  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        runner = getattr(adapter, "gateway_runner", None)
        authority = active_authority(runner) if runner is not None else None
        store = getattr(adapter, "_session_store", None) or getattr(runner, "session_store", None)
        if authority is None or store is None:
            return None
        peeked = store.peek_session_id(session_key)
        session_id = authority.logical_owner(peeked) if peeked else None
        if not session_id:
            return None
        rows = await asyncio.to_thread(
            list_session_admissions, authority.db, session_id=session_id, pending_only=pending_only
        )
        return HeldQueue(authority=authority, session_id=str(session_id), rows=list(rows))
    except Exception as exc:  # noqa: BLE001 — a harness internal; reads as nothing held
        logger.warning("uffici: Hermes's input queue could not be read (%s)", type(exc).__name__)
        return None
