"""The local wire, from the plugin's side.

One WebSocket to a companion on the same machine: a Unix socket by default, TCP
for the pod shape. The plugin presents a token, receives briefs, and sends acks,
posts and dones back.

🔴 The token is SECRET-CLASS. It is read from a file the operator controls, sent
once inside ``hello``, and never logged, never put in an error message and never
included in a frame this module prints — not at any level, not in a whole-frame
dump ([logging.md]).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .call_contract import INSTRUCTIONS, MAX_CALL_ID_CHARS, MAX_CALL_PAYLOAD_BYTES, encode_frame

logger = logging.getLogger(__name__)

#: The wire path the companion serves.
WIRE_PATH = "/v1/ring"

#: The protocol version this plugin speaks.
WIRE_VERSION = 1

#: Where the wire is, and who may reach it.
SOCKET_ENV = "UFFICI_WIRE_SOCKET"
URL_ENV = "UFFICI_COMPANION_URL"
ALLOW_REMOTE_ENV = "UFFICI_COMPANION_ALLOW_REMOTE"
TOKEN_FILE_ENV = "UFFICI_WIRE_TOKEN_FILE"
TOKEN_ENV = "UFFICI_WIRE_TOKEN"

#: The companion's own default, which the plugin must match to find it.
DEFAULT_URL = "http://127.0.0.1:7411"

#: Reconnect backoff. Bounded at both ends: fast enough that a companion
#: restart is invisible to a conversation, slow enough that a companion that is
#: gone for an hour is not a busy loop against a dead socket.
BACKOFF_FIRST_SECONDS = 0.5
BACKOFF_MAX_SECONDS = 30.0


class WireConfigError(RuntimeError):
    """The wire's address or token could not be worked out from the environment."""


@dataclass(frozen=True)
class WireAddress:
    """Where to dial, and how."""

    #: "socket" or "tcp".
    kind: str
    #: The Unix socket path, for ``kind == "socket"``.
    socket_path: str = ""
    #: The ws:// URL, for either kind — a socket connection still needs a URL
    #: for the HTTP upgrade's Host header, which the companion ignores.
    url: str = ""

    def describe(self) -> str:
        """A log-safe description: a scheme and a kind, never a path.

        🔴 The socket lives inside ``$UFFICI_HOME``, the directory that also
        holds the wire token. Its path is a map to the token's location.
        """
        return f"<{self.kind}>"


def _is_loopback(host: str) -> bool:
    return host in {"127.0.0.1", "::1", "localhost", ""}


def resolve_address(get_env: Callable[[str], str | None]) -> WireAddress:
    """Where the companion is.

    A Unix socket wins when one is named: it is the default shape, and it is
    the one the operating system's own permissions protect.
    """
    socket_path = (get_env(SOCKET_ENV) or "").strip()
    if socket_path:
        return WireAddress(kind="socket", socket_path=socket_path, url=f"ws://companion{WIRE_PATH}")

    raw = (get_env(URL_ENV) or "").strip() or DEFAULT_URL
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https", "ws", "wss"}:
        raise WireConfigError(
            f"{URL_ENV} has scheme {parsed.scheme!r}; the local wire is http/ws"
        )
    # 🔴 Loopback unless the operator says otherwise, and the refusal names the
    # switch. The wire's only authentication is a local token: pointing it at
    # another machine sends that token across a network it was never meant for.
    if not _is_loopback(parsed.hostname or "") and (get_env(ALLOW_REMOTE_ENV) or "").strip() != "1":
        raise WireConfigError(
            f"{URL_ENV} points at a host that is not loopback. The wire's only authentication "
            f"is a local token, so this is refused unless {ALLOW_REMOTE_ENV}=1 says the "
            f"connection is trusted (the pod shape, where both containers share a namespace)"
        )
    scheme = "wss" if parsed.scheme in {"https", "wss"} else "ws"
    netloc = parsed.netloc or f"{parsed.hostname or '127.0.0.1'}:{parsed.port or 7411}"
    return WireAddress(kind="tcp", url=f"{scheme}://{netloc}{WIRE_PATH}")


def resolve_token(get_env: Callable[[str], str | None]) -> str:
    """The wire token, from its file or straight from the environment.

    The FILE is the normal path — it is what the companion writes at 0600, and
    a value in a file does not appear in another process's environment listing.
    """
    path = (get_env(TOKEN_FILE_ENV) or "").strip()
    if path:
        try:
            token = Path(path).read_text(encoding="utf-8").strip()
        except OSError as exc:
            # 🔴 ``exc.strerror`` and not ``exc``: the latter renders the path,
            # and the path is inside the directory holding the token.
            raise WireConfigError(
                f"the wire token file named by {TOKEN_FILE_ENV} could not be read: {exc.strerror}"
            ) from None
        except ValueError:
            # 🔴 A file that is not UTF-8 raises UnicodeDecodeError, whose message
            # RENDERS THE OFFENDING BYTES — a fragment of the token — and it is
            # neither an OSError nor anything the caller catches, so it would
            # escape into whatever the harness logs at its platform boundary.
            # Nothing of the payload is repeated here.
            raise WireConfigError(
                f"the wire token file named by {TOKEN_FILE_ENV} is not valid UTF-8 text"
            ) from None
        if not token:
            raise WireConfigError(f"the wire token file named by {TOKEN_FILE_ENV} is empty")
        return token

    token = (get_env(TOKEN_ENV) or "").strip()
    if not token:
        raise WireConfigError(
            f"no wire token: set {TOKEN_FILE_ENV} to the companion's token file "
            f"(normally $UFFICI_HOME/wire-token), or {TOKEN_ENV} for a container"
        )
    return token


def backoff_seconds(attempt: int) -> float:
    """Exponential, capped. ``attempt`` counts from 1."""
    if attempt < 1:
        attempt = 1
    return min(BACKOFF_FIRST_SECONDS * (2 ** (attempt - 1)), BACKOFF_MAX_SECONDS)


#: This harness PROCESS, minted once at import and sent on every hello (#2234).
#:
#: 🔴 The one thing that tells the companion a restarted Hermes from a reconnect. A
#: reconnect keeps working on what it acknowledged; a restart holds nothing of the
#: old turns, so the companion hands them to the new process. Not a secret and not an
#: identity: a fresh one each time the process starts, and nothing is derived from it.
PROCESS_ID = uuid.uuid4().hex


def plugin_hello(token: str, harness_version: str, plugin_version: str, budget_tokens: int) -> dict:
    """The first frame, carrying the token."""
    return {
        "v": WIRE_VERSION,
        "type": "hello",
        "harness": "hermes",
        "harnessVersion": harness_version,
        "pluginVersion": plugin_version,
        "token": token,
        "contextBudgetTokens": budget_tokens,
        "processId": PROCESS_ID,
    }


def describe_frame(frame: dict[str, Any]) -> dict[str, Any]:
    """What may be said about a frame in a log line.

    🔴 The allow-list is the point, and it is an ALLOW-list rather than a
    deny-list on purpose: a frame gains fields over time, and a deny-list would
    publish each new one until somebody noticed. Ids, a type, an enum and
    counts — never ``text``, never ``delta``, never ``entries``, never ``token``.
    """
    out: dict[str, Any] = {}
    allowed = (
        "type", "requestId", "conversationId", "reason", "outcome", "session", "errorClass",
        "callId", "op",
    )
    for key in allowed:
        value = frame.get(key)
        if isinstance(value, str) and value:
            out[key] = value
    if isinstance(frame.get("delta"), list):
        out["deltaEntryCount"] = len(frame["delta"])
    if isinstance(frame.get("entries"), list):
        out["openerEntryCount"] = len(frame["entries"])
    if isinstance(frame.get("text"), str):
        out["textChars"] = len(frame["text"])
    # A result's class is vocabulary; its payload and its upstream error are not.
    error = frame.get("error")
    error_class = error.get("class") if isinstance(error, dict) else None
    if frame.get("type") == "result" and isinstance(error_class, str):
        out["callErrorClass"] = error_class[:32]
    return out


class WireClient:
    """One connection to the companion, reconnecting until told to stop.

    The WebSocket library is injected rather than imported at module scope so
    the whole client is testable against an in-process fake — and so a harness
    whose ``websockets`` differs in shape fails at connect with a clear error
    instead of at import with a stack trace.
    """

    def __init__(
        self,
        *,
        address: WireAddress,
        token: str,
        harness_version: str,
        plugin_version: str,
        budget_tokens: int,
        connect_fn: Callable[[WireAddress], Awaitable[Any]],
        on_frame: Callable[[dict[str, Any]], Awaitable[None]],
        on_connected: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        on_disconnected: Callable[[], None] | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._address = address
        self._token = token
        self._harness_version = harness_version
        self._plugin_version = plugin_version
        self._budget_tokens = budget_tokens
        self._connect_fn = connect_fn
        self._on_frame = on_frame
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected
        self._sleep = sleep_fn

        self._connection: Any = None
        self._running = False
        self._attempt = 0

    @property
    def connected(self) -> bool:
        return self._connection is not None

    async def send(self, frame: dict[str, Any]) -> bool:
        """Send one frame, reporting whether it went.

        False rather than an exception: every caller is a lifecycle hook that
        must keep going — a post that cannot be delivered is a dropped post,
        not a failed turn for the harness.
        """
        connection = self._connection
        if connection is None:
            logger.warning(
                "uffici: no wire connection; dropping frame", extra=describe_frame(frame)
            )
            return False
        try:
            await connection.send(encode_frame(frame))
        except Exception as exc:  # noqa: BLE001 — classified below, never re-raised at a hook
            logger.warning(
                "uffici: sending a frame failed (%s)",
                type(exc).__name__,
                extra=describe_frame(frame),
            )
            return False
        return True

    async def run(self) -> None:
        """Connect, consume frames, reconnect, until ``stop()``."""
        self._running = True
        while self._running:
            try:
                await self._one_connection()
            except Exception as exc:  # noqa: BLE001 — the reconnect loop's whole job
                logger.warning("uffici: wire connection ended (%s)", type(exc).__name__)
            if not self._running:
                break
            self._attempt += 1
            await self._sleep(backoff_seconds(self._attempt))

    async def stop(self) -> None:
        self._running = False
        connection = self._connection
        self._connection = None
        if connection is not None:
            # Shutting down: a close that fails has nobody left to tell.
            with contextlib.suppress(Exception):
                await connection.close()

    async def _one_connection(self) -> None:
        connection = await self._connect_fn(self._address)
        self._connection = connection
        try:
            hello = plugin_hello(
                self._token,
                self._harness_version,
                self._plugin_version,
                self._budget_tokens,
            )
            await connection.send(encode_frame(hello))
            async for raw in connection:
                frame = self._decode(raw)
                if frame is None:
                    continue
                if frame.get("type") == "hello":
                    # The companion's answer. Only now is the connection real:
                    # the surface can still refuse us with a close frame.
                    self._attempt = 0
                    logger.info(
                        "uffici: attached to the companion",
                        extra={"transport": self._address.describe(), **describe_frame(frame)},
                    )
                    if self._on_connected is not None:
                        await self._on_connected(frame)
                    continue
                await self._on_frame(frame)
        finally:
            self._connection = None
            if self._on_disconnected is not None:
                # The calls this connection carried will never be answered: the
                # companion cancels a detached plugin's calls (#2028).
                self._on_disconnected()

    @staticmethod
    def _decode(raw: Any) -> dict[str, Any] | None:
        if isinstance(raw, bytes | bytearray):
            raw = raw.decode("utf-8", errors="replace")
        if not isinstance(raw, str):
            return None
        oversized = len(raw.encode("utf-8")) > MAX_CALL_PAYLOAD_BYTES
        try:
            frame = json.loads(raw)
        except ValueError:
            # 🔴 The frame's BYTES are never logged: a malformed frame is still
            # a frame, and its text is still transcript.
            logger.warning("uffici: a frame that is not JSON was discarded")
            return None
        if not isinstance(frame, dict):
            logger.warning("uffici: a frame that is not an object was discarded")
            return None
        if frame.get("v") != WIRE_VERSION:
            logger.warning(
                "uffici: a frame of another protocol version was discarded",
                # A scalar, clamped: `v` comes off an untrusted-shape frame and a
                # large object would stringify into the log line whole.
                extra={"frameVersion": str(frame.get("v"))[:16]},
            )
            return None
        if oversized and frame.get("type") == "result" and isinstance(frame.get("callId"), str):
            # 🔴 The bound holds in both directions (#2028): an oversized result
            # is answered as malformed_payload for the call it names — never
            # truncated, and its payload never delivered. (It was parsed to find
            # the callId; the socket's own frame limit bounds that.)
            return {
                "v": WIRE_VERSION,
                "type": "result",
                "callId": frame["callId"][:MAX_CALL_ID_CHARS],
                "error": {
                    "class": "malformed_payload",
                    "instruction": INSTRUCTIONS["malformed_payload"],
                },
            }
        return frame


def env_reader() -> Callable[[str], str | None]:
    """The process environment, as a function the resolvers can be tested with."""
    return os.environ.get
