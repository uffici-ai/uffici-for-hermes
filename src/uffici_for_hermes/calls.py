"""The plugin's half of the call tunnel (#2028).

A ``call`` asks the companion for one platform control-plane operation; the one
``result`` carrying the same ``callId`` answers it. Calls are concurrent — any
number may be in flight, and their results arrive in whatever order they finish
— and every one is bounded by its own ``deadlineMs``.

The tunnel ships before its first operation: nothing in this plugin calls it
yet. Conversation-scoped memory is its first consumer (stage 4), which is why
the frame pair is generic rather than a ``memory`` frame.

🔴 A payload never reaches a log line — not the call's, not the result's. They
carry transcript content and memory rows. What may be logged is the ``op``, the
``callId`` and the error class.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .call_contract import (
    CALL_ERROR_CLASSES,
    INSTRUCTIONS,
    MAX_CALL_ID_CHARS,
    MAX_CALL_PAYLOAD_BYTES,
    MAX_DEADLINE_MS,
    OP_PATTERN,
    encoded_size,
)

__all__ = [
    "CALL_ERROR_CLASSES",
    "MAX_CALL_PAYLOAD_BYTES",
    "CallClient",
    "CallResult",
]

logger = logging.getLogger(__name__)

#: How long past the deadline the plugin still waits for the companion's own
#: ``timed_out``. The companion answers BY the deadline; this covers the trip.
DEFAULT_GRACE_SECONDS = 1.0


@dataclass(frozen=True)
class CallResult:
    """The one answer to a call: a payload, or an error class and its sentence."""

    call_id: str
    payload: dict[str, Any] | None = None
    error_class: str | None = None
    #: The sentence the agent relays to the person (decision 32).
    instruction: str | None = None
    #: The control plane's own typed error, when there is one.
    upstream: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.error_class is None


def _failed(call_id: str, error_class: str) -> CallResult:
    return CallResult(
        call_id=call_id, error_class=error_class, instruction=INSTRUCTIONS[error_class]
    )


def _read_result(call_id: str, frame: dict[str, Any]) -> CallResult:
    """A result frame as a CallResult — or ``malformed_payload`` when its shape is wrong.

    The shape rules are the schema's: exactly one of ``payload`` or ``error``, an
    object either way, and an error carrying a class from the closed five and a
    sentence.
    """
    has_payload, has_error = "payload" in frame, "error" in frame
    if has_payload == has_error:
        return _failed(call_id, "malformed_payload")
    if has_payload:
        payload = frame["payload"]
        if not isinstance(payload, dict):
            return _failed(call_id, "malformed_payload")
        return CallResult(call_id=call_id, payload=payload)
    error = frame["error"]
    if not isinstance(error, dict):
        return _failed(call_id, "malformed_payload")
    error_class, instruction = error.get("class"), error.get("instruction")
    upstream = error.get("upstream")
    if (
        error_class not in CALL_ERROR_CLASSES
        or not isinstance(instruction, str)
        or not instruction
        or (upstream is not None and not isinstance(upstream, dict))
    ):
        return _failed(call_id, "malformed_payload")
    return CallResult(
        call_id=call_id, error_class=error_class, instruction=instruction, upstream=upstream
    )


class CallClient:
    """Sends calls and matches their results by ``callId``.

    ``send`` is the wire client's send: it reports whether the frame went, and
    never raises. It encodes with ``call_contract.encode_frame``, the same
    encoder this client measures a call with.
    """

    def __init__(
        self,
        send: Callable[[dict[str, Any]], Awaitable[bool]],
        *,
        new_id: Callable[[], str] | None = None,
        grace_seconds: float = DEFAULT_GRACE_SECONDS,
    ) -> None:
        self._send = send
        self._new_id = new_id or (lambda: str(uuid.uuid4()))
        self._grace = grace_seconds
        self._pending: dict[str, asyncio.Future[CallResult]] = {}

    @property
    def in_flight(self) -> int:
        return len(self._pending)

    async def call(
        self, op: str, payload: dict[str, Any] | None = None, *, deadline_ms: int
    ) -> CallResult:
        """Make one call and wait for its result, never past its own deadline.

        A call outside the contract (an ``op`` that is not a name, a deadline
        out of range, a callId over its bound) is a programming error and raises
        ``ValueError``; every other failure is a ``CallResult`` with its class.
        """
        if not isinstance(op, str) or not OP_PATTERN.match(op):
            raise ValueError("op must be a name: lower-case, 1–128 characters of [a-z0-9_.-]")
        if (
            isinstance(deadline_ms, bool)
            or not isinstance(deadline_ms, int)
            or not 1 <= deadline_ms <= MAX_DEADLINE_MS
        ):
            raise ValueError(f"deadline_ms must be an integer from 1 to {MAX_DEADLINE_MS}")

        call_id = self._new_id()
        if not isinstance(call_id, str) or not 1 <= len(call_id) <= MAX_CALL_ID_CHARS:
            raise ValueError(f"a callId must be 1–{MAX_CALL_ID_CHARS} characters")
        frame: dict[str, Any] = {
            "v": 1, "type": "call", "callId": call_id, "op": op, "deadlineMs": deadline_ms,
        }
        if payload is not None:
            frame["payload"] = payload
        size = encoded_size(frame)
        if size > MAX_CALL_PAYLOAD_BYTES:
            # 🔴 Refused here rather than sent to be refused there: the bound
            # holds in both directions, and a truncation is never the answer.
            logger.warning(
                "uffici: a call over the wire's bound was not sent",
                extra={"op": op, "callId": call_id, "requestBytes": size},
            )
            return _failed(call_id, "malformed_payload")

        future: asyncio.Future[CallResult] = asyncio.get_running_loop().create_future()
        self._pending[call_id] = future
        try:
            # 🔴 The SEND is inside the deadline too: a send awaits the socket's
            # drain, and a companion that stopped reading must not hold a call
            # past its own deadline any more than a slow answer may.
            async with asyncio.timeout(deadline_ms / 1000 + self._grace):
                if not await self._send(frame):
                    return _failed(call_id, "upstream_failed")
                answered = await future
        except TimeoutError:
            # The companion answers BY the deadline; silence past it means the
            # answer is not coming, and the turn must not hang on it.
            answered = _failed(call_id, "timed_out")
        finally:
            self._pending.pop(call_id, None)
        logger.info(
            "uffici: call answered",
            extra={"op": op, "callId": call_id, "class": answered.error_class or "ok",
                   "requestBytes": size},
        )
        return answered

    def on_result(self, frame: dict[str, Any]) -> bool:
        """Resolve the call a result names. False when it names none in flight.

        🔴 A result for no call in flight is DISCARDED, never raised: it is late
        (the call already timed out here) or it was never asked for. The
        connection and every other call on it carry on.
        """
        call_id = frame.get("callId")
        future = self._pending.get(call_id) if isinstance(call_id, str) else None
        if future is None or future.done():
            logger.debug(
                "uffici: discarding a result for no call in flight",
                # A scalar, clamped: an untrusted-shape frame's id.
                extra={"callId": str(call_id)[:MAX_CALL_ID_CHARS]},
            )
            return False
        future.set_result(_read_result(call_id, frame))
        return True

    def fail_all(self) -> None:
        """Fail every call in flight — the connection they rode is gone.

        The companion cancels a detached plugin's calls and answers none of
        them, so waiting out each deadline would only hang the turns that made
        them.
        """
        for call_id, future in list(self._pending.items()):
            if not future.done():
                future.set_result(_failed(call_id, "upstream_failed"))
        self._pending.clear()
