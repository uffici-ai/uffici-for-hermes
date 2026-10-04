"""Hermes's log lines and the plugin's own, handed to the companion.

One ``logging.Handler`` on the ROOT logger sees every record Hermes and this
plugin emit through Python ``logging``. It keeps the ones at or above two floors —
the harness's (Warning by default) and the plugin's (Info) — in a bounded queue,
and a flusher hands them to the companion with the ``logs.push`` call. The
companion redacts them, and ships them to Uffici at the lowest priority it has.

🔴 Nothing here may slow Hermes. ``emit`` only appends to a deque under the
handler's own lock: it never does I/O, never blocks on the wire, and never
raises. The root logger's level is never touched — a level an operator set stays
theirs, which also means an admin's Info or Debug in Uffici cannot bring back a
line Hermes itself is configured not to emit.

What is NOT captured: output printed without ``logging`` (a ``print``, a
subprocess's stdout), and the wire library's own frame dumps (``websockets.*``).
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
import time
from collections import deque
from datetime import UTC, datetime
from typing import Any

#: The call op the companion serves for log lines.
OP_LOGS_PUSH = "logs.push"

#: The queue's bounds: whichever is reached first drops the oldest line.
MAX_ENTRIES = 5000
MAX_BYTES = 2 * 1024 * 1024

#: One push's bounds: the op's 500 lines, under the wire's call bound.
BATCH_ENTRIES = 500
BATCH_BYTES = 200 * 1024

#: The flusher's cadence, and how long a busy harness can hold lines back.
FLUSH_EVERY_SECONDS = 10.0
STALENESS_ESCAPE_SECONDS = 60.0
FAILURE_BACKOFF_SECONDS = 60.0
DISABLED_WAIT_SECONDS = 15 * 60.0
PUSH_DEADLINE_MS = 10_000

#: How long a disconnect may spend handing over what is queued (#2313). 🔴 A restart
#: is never held longer than this: what has not gone by then stays queued.
FINAL_FLUSH_SECONDS = 2.0

#: A line longer than this is clipped here; the companion clips to the stored bound.
MAX_MESSAGE_CHARS = 16_000

#: One attribute value's bound, and all of a line's attributes' — the companion's own
#: attrs bound, in encoded bytes.
MAX_ATTR_VALUE_CHARS = 512
MAX_ATTRS_BYTES = 4096

#: What a clip leaves at its end.
CLIP_MARK = "…[clipped]"

#: The trailing run of non-space characters a clip removes.
_TRAILING_TOKEN = re.compile(r"\S*$")

#: The four levels, from Python's names.
_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,
    "error": logging.ERROR,
}

#: Attributes every LogRecord has; anything else came from ``extra=`` and rides in attrs.
_STANDARD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {
    "message",
    "asctime",
    "taskName",
    "session_tag",
}

#: This package's name AS LOADED. 🔴 Hermes imports a plugin under its own namespace
#: (``hermes_plugins.uffici_for_hermes``), so the plugin's loggers are named from THAT —
#: a literal ``uffici_for_hermes`` prefix read every plugin line as the harness's and held
#: it to the harness's floor, which the first-conversation e2e caught (#2267).
PLUGIN_PACKAGE = __name__.rpartition(".")[0] or __name__


def is_plugin_logger(name: str) -> bool:
    """Whether a logger belongs to this plugin, whatever namespace loaded it."""
    return name == PLUGIN_PACKAGE or name.startswith(f"{PLUGIN_PACKAGE}.")


#: Loggers never captured: the wire library's frame dumps.
_EXCLUDED_PREFIXES = ("websockets",)

#: Set while the flusher runs, so a record logged on its path — the call it makes,
#: an error it reports — is never captured and shipped in turn.
_IN_FLUSHER: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "uffici_in_flusher", default=False
)


def level_name(levelno: int) -> str:
    """A Python level as the platform names it."""
    if levelno >= logging.ERROR:
        return "error"
    if levelno >= logging.WARNING:
        return "warn"
    if levelno >= logging.INFO:
        return "info"
    return "debug"


def clip(text: str, limit: int) -> str:
    """At most ``limit`` characters, never ending inside a token.

    🔴 The companion redacts, and it runs AFTER this: a clip that cut a key or a token
    in half would leave a fragment no pattern matches (#2259 review). So the cut drops
    the trailing run of non-space characters too — a secret is either whole, for the
    redactor, or gone.
    """
    if len(text) <= limit:
        return text
    kept = text[: max(0, limit - len(CLIP_MARK))]
    return _TRAILING_TOKEN.sub("", kept) + CLIP_MARK


def _json_safe(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    return clip(value if isinstance(value, str) else str(value), MAX_ATTR_VALUE_CHARS)


def _encoded_size(entry: dict[str, Any]) -> int:
    """The entry's size on the wire: UTF-8 bytes of its JSON, as the frame bound counts."""
    return len(json.dumps(entry, ensure_ascii=False, separators=(",", ":")).encode()) + 1


class UfficiLogHandler(logging.Handler):
    """The root handler: two floors, a bounded queue, never a blocking call."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self._queue: deque[dict[str, Any]] = deque()
        self._bytes = 0
        self._dropped = 0
        self._harness_floor = logging.WARNING
        self._plugin_floor = logging.INFO

    # ------------------------------------------------------------ policy

    def set_levels(self, harness: str | None, plugin: str | None) -> None:
        """Take the companion's effective floors from a ``logs.push`` answer."""
        with self.lock:
            self._harness_floor = _LEVELS.get(harness or "", self._harness_floor)
            self._plugin_floor = _LEVELS.get(plugin or "", self._plugin_floor)

    @property
    def floors(self) -> tuple[int, int]:
        """The harness's floor and the plugin's, as Python levels."""
        with self.lock:
            return self._harness_floor, self._plugin_floor

    # ------------------------------------------------------------ capture

    def emit(self, record: logging.LogRecord) -> None:
        """Keep one record. 🔴 No I/O, no wait, no raise — see the module docstring."""
        try:
            if _IN_FLUSHER.get() or not self._wanted(record):
                return
            entry = self._entry(record)
            size = _encoded_size(entry)
            with self.lock:
                self._queue.append(entry)
                self._bytes += size
                entry["_size"] = size
                while len(self._queue) > MAX_ENTRIES or self._bytes > MAX_BYTES:
                    oldest = self._queue.popleft()
                    self._bytes -= oldest["_size"]
                    self._dropped += 1
        except Exception:  # noqa: BLE001 — a log handler must never raise into its caller
            self.handleError(record)

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802 — logging's name
        """Swallowed: a line that could not be kept is simply not kept."""

    def _wanted(self, record: logging.LogRecord) -> bool:
        name = record.name or ""
        if name.startswith(_EXCLUDED_PREFIXES):
            return False
        # 🔴 The call client's line about a logs.push is the push talking about itself.
        if name == f"{PLUGIN_PACKAGE}.calls" and getattr(record, "op", None) == OP_LOGS_PUSH:
            return False
        harness_floor, plugin_floor = self._harness_floor, self._plugin_floor
        floor = plugin_floor if is_plugin_logger(name) else harness_floor
        return record.levelno >= floor

    @staticmethod
    def _entry(record: logging.LogRecord) -> dict[str, Any]:
        name = record.name or ""
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{logging.Formatter().formatException(record.exc_info)}"
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "source": "plugin" if is_plugin_logger(name) else "harness",
            "level": level_name(record.levelno),
            "logger": name[:256],
            "message": clip(message, MAX_MESSAGE_CHARS),
        }
        attrs = {
            key: _json_safe(value)
            for key, value in vars(record).items()
            if key not in _STANDARD_ATTRS and not key.startswith("_")
        }
        if attrs:
            # A line whose attributes still exceed the bound keeps their NAMES only: a
            # push that can never fit the wire would otherwise block every line behind it.
            if len(json.dumps(attrs, ensure_ascii=False).encode()) > MAX_ATTRS_BYTES:
                attrs = {"clippedAttrs": clip(",".join(sorted(attrs)), MAX_ATTR_VALUE_CHARS)}
            entry["attrs"] = attrs
        return entry

    # ------------------------------------------------------------ drain

    def drain(
        self, max_entries: int = BATCH_ENTRIES, max_bytes: int = BATCH_BYTES
    ) -> list[dict[str, Any]]:
        """Take up to a batch, oldest first — always at least one line when any wait."""
        batch: list[dict[str, Any]] = []
        used = 0
        with self.lock:
            while self._queue and len(batch) < max_entries:
                size = self._queue[0]["_size"]
                if batch and used + size > max_bytes:
                    break
                entry = self._queue.popleft()
                self._bytes -= size
                used += size
                batch.append(entry)
        return batch

    def requeue(self, batch: list[dict[str, Any]]) -> None:
        """Put a batch back at the front — the companion was busy. The bounds hold."""
        with self.lock:
            for entry in reversed(batch):
                self._queue.appendleft(entry)
                self._bytes += entry["_size"]
            while len(self._queue) > MAX_ENTRIES or self._bytes > MAX_BYTES:
                dropped = self._queue.pop()
                self._bytes -= dropped["_size"]
                self._dropped += 1

    def discard(self, count: int) -> None:
        """Count lines a push lost for good — they are already out of the queue."""
        with self.lock:
            self._dropped += count

    def clear(self) -> None:
        """Forget everything — shipping is off."""
        with self.lock:
            self._queue.clear()
            self._bytes = 0

    def __len__(self) -> int:
        with self.lock:
            return len(self._queue)

    @property
    def size_bytes(self) -> int:
        with self.lock:
            return self._bytes

    @property
    def dropped(self) -> int:
        with self.lock:
            return self._dropped


#: The process's one handler (#2313), made on first use.
_shared: UfficiLogHandler | None = None


def shared_handler() -> UfficiLogHandler:
    """The process's one handler, installed at plugin load and used by every adapter.

    🔴 One per process, not per adapter: it is installed when Hermes loads the plugin —
    before Hermes logs its startup warnings, and long before it connects the plugin —
    and an adapter Hermes builds later takes the same one, so no line is captured twice
    and none logged between a disconnect and the next connect is missed. Hermes builds
    one adapter per platform; were two alive at once, each flusher would drain this one
    queue and the companion's ``busy`` would keep their pushes from overlapping.
    """
    global _shared
    if _shared is None:
        _shared = UfficiLogHandler()
    return _shared


def forget_shared_handler(root: logging.Logger | None = None) -> None:
    """Remove the process's handler and drop it — for tests, which load the plugin again."""
    global _shared
    if _shared is not None:
        uninstall(_shared, root)
    _shared = None


def install(handler: UfficiLogHandler, root: logging.Logger | None = None) -> None:
    """Add the handler to the root logger, once. 🔴 Never touches the root's level."""
    root = root or logging.getLogger()
    if handler not in root.handlers:
        root.addHandler(handler)


def uninstall(handler: UfficiLogHandler, root: logging.Logger | None = None) -> None:
    """Remove the handler; the queue survives for the next install."""
    (root or logging.getLogger()).removeHandler(handler)


class LogFlusher:
    """Hands the queue to the companion: every ten seconds, only while no call is
    in flight (or once a minute regardless), one push at a time."""

    def __init__(self, handler: UfficiLogHandler, calls: Any, *, clock=time.monotonic) -> None:
        self._handler = handler
        self._calls = calls
        self._clock = clock
        self._last_send = clock()
        self._hold_until = 0.0
        self._pushing = False
        #: Set while no push is in flight, so a stop can wait one out (#2313).
        self._idle = asyncio.Event()
        self._idle.set()

    async def run(self) -> None:
        """Loop until cancelled."""
        _IN_FLUSHER.set(True)
        while True:
            await asyncio.sleep(FLUSH_EVERY_SECONDS)
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — one bad tick must not end shipping for good
                self._hold_until = self._clock() + FAILURE_BACKOFF_SECONDS

    async def tick(self) -> None:
        """One look: push one batch when it is time and the tunnel is quiet."""
        token = _IN_FLUSHER.set(True)
        try:
            now = self._clock()
            if self._pushing or now < self._hold_until or not len(self._handler):
                return
            idle = getattr(self._calls, "in_flight", 0) == 0
            stale = now - self._last_send >= STALENESS_ESCAPE_SECONDS
            if not idle and not stale:
                return
            await self._push_one(now)
        finally:
            _IN_FLUSHER.reset(token)

    async def finish(self, task: Any) -> None:
        """Stop the periodic loop ``task`` and hand over what is left, once (#2313).

        🔴 A push already in flight is WAITED OUT, never cancelled: its frame has
        usually reached the companion, so a cancel put back a batch the companion had
        taken — the final push then sent it twice, or was answered ``busy`` and sent
        nothing. The caller bounds the whole of it; the loop is cancelled even when
        that bound cuts the wait.
        """
        try:
            await self._idle.wait()
        finally:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await self.flush_final()

    async def flush_final(self) -> None:
        """Hand over everything queued, now — once, as the plugin disconnects (#2313).

        🔴 The lines logged in the seconds before a stop are the ones that say why it
        stopped (the signal, the drain that timed out, the interrupted tool), and they
        sit inside the ten-second cadence: without this they died with the process.
        It ignores the cadence and the quiet-tunnel gate, stops at the first push not
        taken, and never bounds itself — the CALLER bounds it (``FINAL_FLUSH_SECONDS``),
        and a push cut there puts its batch back. Never call it while ``run`` is
        running: the companion takes one push at a time.
        """
        token = _IN_FLUSHER.set(True)
        try:
            while len(self._handler):
                if not await self._push_one(self._clock()):
                    return
        finally:
            _IN_FLUSHER.reset(token)

    async def _push_one(self, now: float) -> bool:
        """Push one batch and settle it; whether the companion took it."""
        batch = self._handler.drain()
        if not batch:
            return False
        self._pushing = True
        self._idle.clear()
        try:
            result = await self._calls.call(
                OP_LOGS_PUSH,
                {"entries": [{k: v for k, v in e.items() if k != "_size"} for e in batch]},
                deadline_ms=PUSH_DEADLINE_MS,
            )
        except BaseException:
            # 🔴 The batch is out of the queue already: a push that raised (a
            # dropped tunnel, a cancel at shutdown) puts it back rather than
            # losing it uncounted (#2259 review).
            self._handler.requeue(batch)
            raise
        finally:
            self._pushing = False
            self._idle.set()
        return self._settle(batch, result, now)

    def _settle(self, batch: list[dict[str, Any]], result: Any, now: float) -> bool:
        """Act on a push's answer; whether to push again now — the batch was taken and
        the companion asked for no wait."""
        if getattr(result, "error_class", None) == "malformed_payload":
            # 🔴 The wire refused the push as it stands: sending the same batch again
            # would be refused again, forever, with every later line queued behind it.
            self._handler.discard(len(batch))
            self._hold_until = now + FAILURE_BACKOFF_SECONDS
            return False
        if not getattr(result, "ok", False):
            # Failed or timed out: keep the lines, try again in a minute.
            self._handler.requeue(batch)
            self._hold_until = now + FAILURE_BACKOFF_SECONDS
            return False
        answer = result.payload or {}
        self._handler.set_levels(answer.get("harnessLevel"), answer.get("pluginLevel"))
        retry = answer.get("retryAfterSeconds")
        if answer.get("busy"):
            self._handler.requeue(batch)
            return False
        if answer.get("disabled"):
            self._handler.clear()
            wait = retry if isinstance(retry, (int, float)) and retry > 0 else DISABLED_WAIT_SECONDS
            self._hold_until = now + wait
            return False
        self._last_send = now
        if isinstance(retry, (int, float)) and retry > 0:
            self._hold_until = now + retry
            return False
        return True
