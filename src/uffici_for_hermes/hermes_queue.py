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
  Hermes runs it, and handing it over too would answer the person twice;
- and once the companion says it redelivers, every brief still QUEUED from a previous
  process is cancelled before Hermes's startup recovery replays it (#3125,
  ``withdraw_replays`` in the adapter): the companion's record wins, so the model never runs
  twice and a replay never answers a request nobody has open;
- a brief the companion withdraws is cancelled in Hermes's queue too, not only stopped if
  it is running (#3125): a queued row would otherwise run later and answer whatever request
  was open then.

A cancel goes through Hermes's own ``SessionAuthority.cancel_queued`` with the actor
Hermes's messaging ``/stop`` builds for exactly this (``_hm_stop_owner_turn``: the route's
still-queued ``messaging:*`` rows, cancelled with ``session:control``), so it settles every
observer the way Hermes's own cancel does.

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
    "CONTROL_CAPABILITIES",
    "CONTROL_SUBJECT",
    "CONTROL_TRANSPORT",
    "DELIVERY_SEPARATOR",
    "HeldQueue",
    "delivery_id",
    "is_delivery",
    "new_delivery_token",
    "read_platform_queues",
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

#: The actor Hermes's own messaging ``/stop`` cancels a route's queued rows with
#: (``gateway/run_inbound.py``, ``_hm_stop_owner_turn``); pinned by ``dev/hermes_contract.py``.
CONTROL_SUBJECT = "messaging-control"
CONTROL_CAPABILITIES = frozenset({"session:submit", "session:control"})
CONTROL_TRANSPORT = "busy-command"

#: A Hermes refusal's reason is a closed vocabulary (``RuntimeStoreError.reason``); anything
#: else is logged by its class name only.
_REASON = re.compile(r"^[a-z_]{1,40}$")


def new_delivery_token() -> str:
    """A fresh token for one delivery of a brief."""
    return secrets.token_hex(4)


def delivery_id(request_id: str, token: str) -> str:
    """The ``message_id`` a brief carries into Hermes: its request and this delivery."""
    return f"{request_id}{DELIVERY_SEPARATOR}{token}"


def is_delivery(message_id: Any) -> bool:
    """Whether a ``message_id`` is one of this plugin's deliveries (``<requestId>~<token>``)."""
    return bool(_DELIVERY.match(str(message_id or "")))


def _reason(exc: BaseException) -> str:
    reason = getattr(exc, "reason", None)
    return reason if isinstance(reason, str) and _REASON.match(reason) else type(exc).__name__


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

    def running(self) -> set[str]:
        """The deliveries whose turn Hermes is running on this session now (``started``)."""
        return {str(row.get("request_id")) for row in self.rows if row.get("status") == "started"}

    def replayed(self) -> set[str]:
        """This plugin's deliveries still ``queued`` from a PREVIOUS process.

        A row records the epoch of the process that admitted it (``owner_epoch``); one that is
        not the authority's own is what Hermes's startup recovery would replay. Without an
        epoch to compare there is no evidence, and nothing is called a replay.
        """
        epoch = getattr(self.authority, "epoch", None)
        if epoch is None:
            return set()
        return {
            str(row.get("request_id"))
            for row in self.rows
            if row.get("status") == "queued"
            and is_delivery(row.get("request_id"))
            and row.get("owner_epoch") is not None
            and row.get("owner_epoch") != epoch
        }

    async def cancel(self, deliveries: set[str]) -> set[str]:
        """Cancel these deliveries' QUEUED rows through Hermes's own ``cancel_queued``.

        Returns the deliveries cancelled. A row Hermes claimed meanwhile (``stale_generation``)
        runs where it is; a refusal of any kind is logged by its reason and never fatal.
        """
        targets = [
            row
            for row in self.rows
            if row.get("status") == "queued" and str(row.get("request_id")) in deliveries
        ]
        if not targets:
            return set()
        try:
            from gateway.session_contract import (  # type: ignore[import-not-found]
                SessionActor,
                SessionRef,
            )

            profile = self.authority.profile_id
            ref = SessionRef(profile, self.session_id)
            actor = SessionActor(CONTROL_SUBJECT, profile, CONTROL_CAPABILITIES, CONTROL_TRANSPORT)
        except Exception as exc:  # noqa: BLE001 — a harness internal; nothing is cancelled
            logger.warning(
                "uffici: Hermes's queue cannot be cancelled on (%s); a withdrawn brief still "
                "queued there may run, and its replies are dropped",
                type(exc).__name__,
            )
            return set()
        cancelled: set[str] = set()
        for row in targets:
            try:
                await self.authority.cancel_queued(actor, ref, row["admission_id"])
            except Exception as exc:  # noqa: BLE001 — claimed meanwhile, or refused
                logger.info("uffici: Hermes kept a queued brief (%s)", _reason(exc))
                continue
            cancelled.add(str(row.get("request_id")))
        return cancelled

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


def _authority_and_store(adapter: Any, active_authority: Any) -> tuple[Any, Any] | None:
    runner = getattr(adapter, "gateway_runner", None)
    authority = active_authority(runner) if runner is not None else None
    store = getattr(adapter, "_session_store", None) or getattr(runner, "session_store", None)
    if authority is None or store is None:
        return None
    return authority, store


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
        found = _authority_and_store(adapter, active_authority)
        if found is None:
            return None
        authority, store = found
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


def _platform_of(origin: Any) -> str:
    platform = getattr(origin, "platform", None)
    return str(getattr(platform, "value", platform) or "")


async def read_platform_queues(adapter: Any, platform: str) -> list[HeldQueue] | None:
    """Hermes's queue for every session of ``platform`` its session store lists.

    The sessions Hermes's own startup recovery binds (``recover_profile_native_sessions``
    lists the store's entries by their origin). Empty on a Hermes with no durable queue —
    nothing can be queued there. None when the queue exists but cannot be read: no authority
    or session store wired yet, or a store that raised (logged), so a caller can tell
    "nothing queued" from "not known".

    🔴 The whole enumeration runs in ONE worker thread (#3125 review): the session store's
    listing takes its lock and may load from disk, and this runs inside the gateway's
    ``connect``, on its loop.
    """
    try:
        from gateway.session_authorities import active_authority  # type: ignore[import-not-found]
        from hermes_state_runtime import list_session_admissions  # type: ignore[import-not-found]
    except ImportError:
        return []
    try:
        found = _authority_and_store(adapter, active_authority)
        if found is None:
            return None
        authority, store = found

        def enumerate_queues() -> list[tuple[str, list[dict[str, Any]]]]:
            owners: list[str] = []
            for entry in store.list_sessions():
                if _platform_of(getattr(entry, "origin", None)) != platform:
                    continue
                owner = authority.logical_owner(entry.session_id)
                if owner and str(owner) not in owners:
                    owners.append(str(owner))
            return [
                (
                    owner,
                    list(
                        list_session_admissions(authority.db, session_id=owner, pending_only=True)
                    ),
                )
                for owner in owners
            ]

        listed = await asyncio.to_thread(enumerate_queues)
        return [
            HeldQueue(authority=authority, session_id=owner, rows=rows) for owner, rows in listed
        ]
    except Exception as exc:  # noqa: BLE001 — a harness internal; nothing is withdrawn
        logger.warning("uffici: Hermes's input queues could not be read (%s)", type(exc).__name__)
        return None
