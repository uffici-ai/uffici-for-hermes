"""Hermes's log lines and the plugin's own, on their way to the companion (#2265)."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import pytest

from conftest import HERMES_DEFAULT_ROOT_LEVEL, HERMES_TURN_LOGGER, PlatformConfig
from uffici_for_hermes import log_shipping
from uffici_for_hermes.adapter import UfficiAdapter
from uffici_for_hermes.log_shipping import (
    MAX_ENTRIES,
    OP_LOGS_PUSH,
    LogFlusher,
    UfficiLogHandler,
    install,
    uninstall,
)


def record(name: str, level: int, message: str = "a line", **extra: Any) -> logging.LogRecord:
    rec = logging.LogRecord(name, level, __file__, 1, message, (), None)
    for key, value in extra.items():
        setattr(rec, key, value)
    return rec


@dataclass
class Answer:
    ok: bool = True
    payload: dict[str, Any] | None = None
    error_class: str | None = None


@dataclass
class FakeCalls:
    """The call client's surface the flusher uses — recording, answering as told."""

    in_flight: int = 0
    answer: Answer = field(
        default_factory=lambda: Answer(
            payload={
                "accepted": 1,
                "dropped": 0,
                "disabled": False,
                "busy": False,
                "harnessLevel": "warn",
                "pluginLevel": "info",
            }
        )
    )
    pushed: list[dict[str, Any]] = field(default_factory=list)
    during: Any = None

    async def call(self, op: str, payload: dict[str, Any], *, deadline_ms: int) -> Answer:
        assert op == OP_LOGS_PUSH and deadline_ms > 0
        self.pushed.append(payload)
        if self.during is not None:
            self.during()
        return self.answer


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# ---------------------------------------------------------------- the handler


class Untouchable:
    """A call client that records anything so much as looking at it. It records
    rather than raises for the same reason the traps below do: a caller that
    swallowed the exception — a ``getattr`` with a default — would hide it."""

    def __init__(self) -> None:
        self.touched: list[str] = []

    def __getattr__(self, name: str) -> Any:
        self.touched.append(name)
        raise AttributeError(name)


def test_emit_never_touches_the_wire_and_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """🔴 Structural: while emit runs, every way to do I/O or wait is trapped. A handler
    that swallowed the trap's exception would hide it, so the trap RECORDS instead."""
    import builtins
    import socket
    import time

    touched: list[str] = []

    def trap(name: str):
        def trapped(*_args: Any, **_kwargs: Any) -> Any:
            touched.append(name)
            raise AssertionError(f"emit touched {name}")

        return trapped

    handler = UfficiLogHandler()
    client = Untouchable()
    LogFlusher(handler, client)  # built beside it, as in the adapter
    for target, name in (
        (builtins, "open"),
        (socket, "socket"),
        (time, "sleep"),
        (asyncio, "get_running_loop"),
        (asyncio, "get_event_loop"),
        (asyncio, "ensure_future"),
    ):
        monkeypatch.setattr(target, name, trap(name))
    handler.handle(record("run_agent", logging.ERROR, "tool call failed"))
    monkeypatch.undo()
    assert touched == [], f"emit did I/O or waited: {touched}"
    assert client.touched == [], f"emit touched the call client: {client.touched}"
    assert len(handler) == 1

    class Exploding(logging.LogRecord):
        def getMessage(self) -> str:  # noqa: N802
            raise RuntimeError("a record that cannot render")

    handler.handle(Exploding("run_agent", logging.ERROR, __file__, 1, "x", (), None))
    assert len(handler) == 1  # swallowed, not raised


def test_two_floors_and_the_root_level_untouched() -> None:
    root = logging.getLogger()
    before = root.level
    handler = UfficiLogHandler()
    install(handler, root)
    install(handler, root)  # idempotent
    try:
        assert root.handlers.count(handler) == 1
        handler.handle(record("gateway.run", logging.INFO, "harness chatter"))
        handler.handle(record("gateway.run", logging.WARNING, "harness warning"))
        handler.handle(record("uffici_for_hermes.adapter", logging.INFO, "plugin info"))
        handler.handle(record("uffici_for_hermes.adapter", logging.DEBUG, "plugin debug"))
        assert [e["message"] for e in handler.drain()] == ["harness warning", "plugin info"]
        handler.set_levels("info", "debug")
        handler.handle(record("gateway.run", logging.INFO, "now kept"))
        handler.handle(record("uffici_for_hermes.adapter", logging.DEBUG, "plugin debug kept"))
        assert [e["message"] for e in handler.drain()] == ["now kept", "plugin debug kept"]
        handler.set_levels("bogus", None)  # an unknown level changes nothing
        assert handler.floors == (logging.INFO, logging.DEBUG)
        assert root.level == before, "the root logger's level must never change"
    finally:
        uninstall(handler, root)
    assert handler not in root.handlers


def test_an_entry_carries_its_source_level_extra_and_traceback() -> None:
    handler = UfficiLogHandler()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        rec = record("run_agent", logging.ERROR, "failed", requestId="req_1", obj=object())
        rec.exc_info = sys.exc_info()
    handler.handle(rec)
    handler.handle(record("uffici_for_hermes.calls", logging.WARNING, "plugin warn"))
    harness, plugin = handler.drain()
    assert harness["source"] == "harness" and harness["level"] == "error"
    assert harness["logger"] == "run_agent" and "ValueError: boom" in harness["message"]
    assert harness["attrs"]["requestId"] == "req_1" and isinstance(harness["attrs"]["obj"], str)
    assert harness["ts"].endswith("+00:00") and "." in harness["ts"]
    assert plugin["source"] == "plugin" and plugin["level"] == "warn" and "attrs" not in plugin


def test_the_wire_librarys_frame_dumps_are_never_kept() -> None:
    handler = UfficiLogHandler()
    handler.handle(record("websockets.client", logging.ERROR, "< TEXT '{...}'"))
    assert len(handler) == 0


def test_the_bounds_hold_and_count_what_they_drop() -> None:
    handler = UfficiLogHandler()
    for i in range(MAX_ENTRIES + 10):
        handler.handle(record("run_agent", logging.ERROR, f"line {i}"))
    assert len(handler) == MAX_ENTRIES and handler.dropped == 10
    assert handler.drain(1)[0]["message"] == "line 10"  # the oldest went first

    big = UfficiLogHandler()
    for _ in range(400):
        big.handle(record("run_agent", logging.ERROR, "x" * 8000))
    assert big.size_bytes <= log_shipping.MAX_BYTES and big.dropped > 0


def test_emit_and_drain_from_two_threads_lose_nothing_the_bounds_did_not_drop() -> None:
    handler = UfficiLogHandler()
    drained: list[int] = []
    stop = threading.Event()

    def drainer() -> None:
        while not stop.is_set():
            drained.append(len(handler.drain(50)))

    thread = threading.Thread(target=drainer)
    thread.start()
    for i in range(3000):
        handler.handle(record("run_agent", logging.ERROR, f"line {i}"))
    stop.set()
    thread.join()
    rest = len(handler.drain(10_000, 1 << 30))
    assert sum(drained) + rest + handler.dropped == 3000


def test_requeue_restores_order_within_the_bounds() -> None:
    handler = UfficiLogHandler()
    for i in range(3):
        handler.handle(record("run_agent", logging.ERROR, f"line {i}"))
    batch = handler.drain(2)
    handler.requeue(batch)
    assert [e["message"] for e in handler.drain()] == ["line 0", "line 1", "line 2"]
    handler.handle(record("run_agent", logging.ERROR, "again"))
    handler.clear()
    assert len(handler) == 0 and handler.size_bytes == 0


# ---------------------------------------------------------------- the flusher


def flusher_rig(**calls_kwargs: Any) -> tuple[UfficiLogHandler, FakeCalls, LogFlusher, Clock]:
    handler = UfficiLogHandler()
    calls = FakeCalls(**calls_kwargs)
    clock = Clock()
    return handler, calls, LogFlusher(handler, calls, clock=clock), clock


def test_no_push_while_a_call_is_in_flight_until_the_escape() -> None:
    handler, calls, flusher, clock = flusher_rig(in_flight=1)
    handler.handle(record("run_agent", logging.ERROR))
    asyncio.run(flusher.tick())
    assert calls.pushed == []
    clock.now += log_shipping.STALENESS_ESCAPE_SECONDS
    asyncio.run(flusher.tick())
    assert len(calls.pushed) == 1, "a busy harness's lines were starved past the escape"

    handler.handle(record("run_agent", logging.ERROR))
    calls.in_flight = 0
    asyncio.run(flusher.tick())
    assert len(calls.pushed) == 2, "idle, and still not sent"


def test_the_answer_sets_the_floors() -> None:
    handler, calls, flusher, _ = flusher_rig()
    calls.answer = Answer(
        payload={
            "accepted": 1,
            "dropped": 0,
            "disabled": False,
            "busy": False,
            "harnessLevel": "info",
            "pluginLevel": "debug",
        }
    )
    handler.handle(record("run_agent", logging.ERROR))
    asyncio.run(flusher.tick())
    handler.handle(record("gateway.run", logging.INFO, "harness info"))
    assert [e["message"] for e in handler.drain()] == ["harness info"]


def test_one_push_in_flight_and_its_own_lines_are_never_kept() -> None:
    handler, calls, flusher, _ = flusher_rig()
    handler.handle(record("run_agent", logging.ERROR, "the line to ship"))

    def during() -> None:
        # A second tick while the first is pushing does nothing…
        assert flusher._pushing  # noqa: SLF001
        # …and what the push path logs is not captured and shipped in turn.
        logging.getLogger("uffici_for_hermes.calls").warning(
            "uffici: call answered", extra={"op": OP_LOGS_PUSH}
        )
        logging.getLogger("gateway.run").error("logged on the flusher's path")

    calls.during = during
    root = logging.getLogger()
    install(handler, root)
    try:
        asyncio.run(flusher.tick())
    finally:
        uninstall(handler, root)
    assert len(calls.pushed) == 1 and len(handler) == 0
    # The call client's logs.push line is excluded by NAME too, outside the flusher.
    handler.handle(record("uffici_for_hermes.calls", logging.WARNING, "x", op=OP_LOGS_PUSH))
    assert len(handler) == 0
    pushed = calls.pushed[0]["entries"]
    assert [e["message"] for e in pushed] == ["the line to ship"] and "_size" not in pushed[0]


def test_busy_keeps_the_lines_for_the_next_tick() -> None:
    handler, calls, flusher, _ = flusher_rig()
    calls.answer = Answer(
        payload={
            "accepted": 0,
            "dropped": 0,
            "disabled": False,
            "busy": True,
            "harnessLevel": "warn",
            "pluginLevel": "info",
        }
    )
    handler.handle(record("run_agent", logging.ERROR))
    asyncio.run(flusher.tick())
    assert len(handler) == 1


def test_disabled_drops_the_queue_and_waits() -> None:
    handler, calls, flusher, clock = flusher_rig()
    calls.answer = Answer(
        payload={
            "accepted": 0,
            "dropped": 1,
            "disabled": True,
            "busy": False,
            "harnessLevel": "warn",
            "pluginLevel": "info",
        }
    )
    handler.handle(record("run_agent", logging.ERROR))
    asyncio.run(flusher.tick())
    assert len(handler) == 0
    handler.handle(record("run_agent", logging.ERROR))
    clock.now += log_shipping.DISABLED_WAIT_SECONDS - 1
    asyncio.run(flusher.tick())
    assert len(calls.pushed) == 1, "pushed again before the wait"
    clock.now += 1
    asyncio.run(flusher.tick())
    assert len(calls.pushed) == 2


def test_a_named_wait_and_a_failure_hold_the_next_push() -> None:
    handler, calls, flusher, clock = flusher_rig()
    calls.answer = Answer(
        payload={
            "accepted": 1,
            "dropped": 0,
            "disabled": False,
            "busy": False,
            "harnessLevel": "warn",
            "pluginLevel": "info",
            "retryAfterSeconds": 30,
        }
    )
    handler.handle(record("run_agent", logging.ERROR))
    asyncio.run(flusher.tick())
    handler.handle(record("run_agent", logging.ERROR))
    clock.now += 29
    asyncio.run(flusher.tick())
    assert len(calls.pushed) == 1
    calls.answer = Answer(ok=False)
    clock.now += 1
    asyncio.run(flusher.tick())
    assert len(calls.pushed) == 2 and len(handler) == 1, "a failed push must keep its lines"
    clock.now += log_shipping.FAILURE_BACKOFF_SECONDS - 1
    asyncio.run(flusher.tick())
    assert len(calls.pushed) == 2
    clock.now += 1
    asyncio.run(flusher.tick())
    assert len(calls.pushed) == 3


def test_an_empty_queue_sends_nothing() -> None:
    _, calls, flusher, _ = flusher_rig()
    asyncio.run(flusher.tick())
    assert calls.pushed == []


def test_run_ticks_on_its_cadence(monkeypatch: pytest.MonkeyPatch) -> None:
    handler, calls, flusher, _ = flusher_rig()
    handler.handle(record("run_agent", logging.ERROR))
    monkeypatch.setattr(log_shipping, "FLUSH_EVERY_SECONDS", 0)

    async def scenario() -> None:
        task = asyncio.ensure_future(flusher.run())
        for _ in range(20):
            await asyncio.sleep(0)
            if calls.pushed:
                break
        task.cancel()

    asyncio.run(scenario())
    assert len(calls.pushed) == 1


# ---------------------------------------------------------------- the adapter


def adapter_with(env: dict[str, str]) -> UfficiAdapter:
    async def connect_fn(_address: Any) -> Any:
        class Connection:
            async def send(self, _raw: str) -> None:
                return None

            async def close(self) -> None:
                return None

            def __aiter__(self):
                async def gen():
                    return
                    yield  # pragma: no cover

                return gen()

        return Connection()

    return UfficiAdapter(
        PlatformConfig(enabled=True),
        get_env=lambda name: {
            "UFFICI_WIRE_TOKEN": "t",
            "UFFICI_CONTEXT_WINDOW_TOKENS": "32000",
            **env,
        }.get(name),
        connect_fn=connect_fn,
    )


def test_the_adapter_installs_the_handler_once_and_keeps_it_past_a_stop() -> None:
    adapter = adapter_with({})

    async def scenario() -> None:
        assert await adapter.connect()
        handler = adapter._log_handler  # noqa: SLF001
        assert handler is log_shipping.shared_handler()
        assert handler in logging.getLogger().handlers
        assert adapter._log_flusher is not None  # noqa: SLF001
        await adapter.connect(is_reconnect=True)  # a reconnect keeps exactly one
        assert logging.getLogger().handlers.count(handler) == 1
        await adapter.disconnect()
        # 🔴 #2313: kept, so the lines between a disconnect and the next connect queue.
        assert logging.getLogger().handlers.count(handler) == 1
        assert adapter._log_flusher is None  # noqa: SLF001

    asyncio.run(scenario())


class AdapterCalls(FakeCalls):
    """The recording client, as the adapter's call tunnel."""

    def fail_all(self) -> None:
        return None


def adapter_with_calls(calls: FakeCalls) -> UfficiAdapter:
    adapter = adapter_with({})
    adapter._calls = calls  # noqa: SLF001
    return adapter


def test_disconnect_pushes_what_was_logged_just_before_it() -> None:
    """🔴 #2313: a restart's own lines — the SIGTERM context, the drain that timed out —
    are logged in the seconds before the stop, inside the flusher's ten-second cadence.
    The queue died with the process, so the Logs tab never saw why the harness went."""
    calls = AdapterCalls(in_flight=1)  # a call in flight holds back an ordinary tick

    async def scenario() -> None:
        adapter = adapter_with_calls(calls)
        assert await adapter.connect()
        logging.getLogger("gateway.run").warning("Gateway drain timed out after 0.0s")
        await adapter.disconnect()

    asyncio.run(scenario())
    shipped = [e["message"] for p in calls.pushed for e in p["entries"]]
    assert "Gateway drain timed out after 0.0s" in shipped
    assert len(log_shipping.shared_handler()) == 0


def test_a_hanging_final_push_never_holds_the_disconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    """🔴 #2313: the final push is bounded. A companion that never answers costs the
    stop that bound and no more, and the lines stay queued rather than lost uncounted."""
    monkeypatch.setattr(log_shipping, "FINAL_FLUSH_SECONDS", 0.05)
    calls = AdapterCalls()

    async def never(*_args: Any, **_kwargs: Any) -> Answer:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    calls.call = never  # type: ignore[method-assign]

    async def scenario() -> float:
        adapter = adapter_with_calls(calls)
        assert await adapter.connect()
        logging.getLogger("gateway.run").warning("Shutdown context: signal=SIGTERM")
        started = asyncio.get_running_loop().time()
        await adapter.disconnect()
        return asyncio.get_running_loop().time() - started

    assert asyncio.run(scenario()) < 1.0, "the disconnect waited on a push that never answered"
    assert [e["message"] for e in log_shipping.shared_handler().drain()] == [
        "Shutdown context: signal=SIGTERM"
    ]


def test_a_line_logged_before_the_first_connect_is_shipped() -> None:
    """🔴 #2313: Hermes logs its startup warnings after loading plugins and before
    connecting them. A handler installed at connect never saw them."""
    from test_registration import RecordingCtx
    from uffici_for_hermes import register

    register(RecordingCtx())
    logging.getLogger("gateway.run").warning("No env user allowlists configured")
    calls = AdapterCalls()

    async def scenario() -> None:
        adapter = adapter_with_calls(calls)
        assert await adapter.connect()
        await adapter.disconnect()

    asyncio.run(scenario())
    shipped = [e["message"] for p in calls.pushed for e in p["entries"]]
    assert shipped == ["No env user allowlists configured"]


def test_the_switch_off_installs_nothing_at_load(monkeypatch: pytest.MonkeyPatch) -> None:
    from test_registration import RecordingCtx
    from uffici_for_hermes import register

    monkeypatch.setenv("UFFICI_LOG_SHIPPING", "off")
    before = list(logging.getLogger().handlers)
    register(RecordingCtx())
    assert logging.getLogger().handlers == before


def test_the_final_flush_ignores_the_cadence_and_drains_every_batch() -> None:
    handler, calls, flusher, _ = flusher_rig(in_flight=1)
    for n in range(log_shipping.BATCH_ENTRIES + 3):
        handler.handle(record("run_agent", logging.ERROR, f"line {n}"))
    asyncio.run(flusher.flush_final())
    assert [len(p["entries"]) for p in calls.pushed] == [log_shipping.BATCH_ENTRIES, 3]
    assert len(handler) == 0


def test_the_final_flush_stops_at_the_first_push_not_taken() -> None:
    handler, calls, flusher, _ = flusher_rig()
    calls.answer = Answer(ok=False, error_class="upstream_failed")
    handler.handle(record("run_agent", logging.ERROR))
    asyncio.run(flusher.flush_final())
    assert len(calls.pushed) == 1 and len(handler) == 1


def test_the_switch_off_installs_nothing() -> None:
    adapter = adapter_with({"UFFICI_LOG_SHIPPING": "OFF"})

    async def scenario() -> None:
        before = list(logging.getLogger().handlers)
        assert await adapter.connect()
        assert adapter._log_handler is None  # noqa: SLF001
        assert logging.getLogger().handlers == before
        await adapter.disconnect()

    asyncio.run(scenario())


def test_a_hermes_turn_line_is_kept_only_once_raised() -> None:
    """Hermes's own turn logger at its default root level (the transcription the
    contract check pins against the real Hermes)."""
    handler = UfficiLogHandler()
    root = logging.getLogger()
    before = root.level
    root.setLevel(HERMES_DEFAULT_ROOT_LEVEL)
    install(handler, root)
    try:
        logging.getLogger(HERMES_TURN_LOGGER).info("a turn line")
        assert len(handler) == 0, "an INFO harness line shipped at the default Warning floor"
        handler.set_levels("info", "info")
        logging.getLogger(HERMES_TURN_LOGGER).info("a turn line")
        assert [e["source"] for e in handler.drain()] == ["harness"]
    finally:
        uninstall(handler, root)
        root.setLevel(before)


def _load_as_hermes_does():
    """This module, imported under the name Hermes gives a plugin (``hermes_plugins.…``)."""
    import importlib.util
    import sys

    name = "hermes_plugins.uffici_for_hermes.log_shipping"
    spec = importlib.util.spec_from_file_location(name, log_shipping.__file__)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_the_plugins_lines_are_the_plugins_under_hermess_namespace() -> None:
    # 🔴 #2267: Hermes loads the plugin as ``hermes_plugins.uffici_for_hermes``, so its
    # loggers carry that name. A literal prefix classified every one of them as the
    # harness's — held to the harness floor, and shipped under the wrong source.
    loaded = _load_as_hermes_does()
    assert loaded.PLUGIN_PACKAGE == "hermes_plugins.uffici_for_hermes"
    handler = loaded.UfficiLogHandler()
    handler.handle(record("hermes_plugins.uffici_for_hermes.adapter", logging.INFO, "plugin info"))
    handler.handle(record("gateway.run", logging.INFO, "harness info, below its floor"))
    handler.handle(record("uffici_for_hermes_lookalike", logging.WARNING, "not the plugin"))
    lines = handler.drain()
    assert [(line["source"], line["message"]) for line in lines] == [
        ("plugin", "plugin info"),
        ("harness", "not the plugin"),
    ]
    # And the push's own line is excluded under that name too.
    handler.handle(
        record("hermes_plugins.uffici_for_hermes.calls", logging.WARNING, "x", op=OP_LOGS_PUSH)
    )
    assert handler.drain() == []


# ---------------------------------------------------------------- the wire's bound (#2259 review)


def _wire_bytes(batch: list[dict[str, Any]]) -> int:
    entries = [{k: v for k, v in e.items() if k != "_size"} for e in batch]
    return len(json.dumps({"entries": entries}, ensure_ascii=False).encode())


def test_a_batch_of_wide_characters_stays_under_the_bound_in_bytes() -> None:
    # 🔴 Counting characters let six CJK messages estimate under 200 KiB and encode
    # past the wire's 256 KiB: a push the call client refuses, forever.
    handler = UfficiLogHandler()
    for _ in range(20):
        handler.handle(record("run_agent", logging.ERROR, "漢" * 15_000))
    batch = handler.drain()
    assert len(batch) >= 1
    assert _wire_bytes(batch) <= log_shipping.BATCH_BYTES + 64


def test_an_oversized_extra_keeps_its_names_and_not_its_bulk() -> None:
    handler = UfficiLogHandler()
    handler.handle(record("run_agent", logging.ERROR, "x", response="r" * 300_000))
    wide = {f"k{i}": "v" * 400 for i in range(40)}
    handler.handle(record("run_agent", logging.ERROR, "y", **wide))
    first, second = handler.drain()
    assert len(first["attrs"]["response"]) <= log_shipping.MAX_ATTR_VALUE_CHARS
    assert set(second["attrs"]) == {"clippedAttrs"}
    assert _wire_bytes([first, second]) < 16_000


def test_a_clip_never_leaves_part_of_a_token_for_the_redactor_to_miss() -> None:
    # 🔴 The companion redacts AFTER this clip: a key cut in half is a fragment no
    # pattern matches, so the cut takes the whole trailing token with it.
    key = "sk-or-v1-" + "a" * 64
    message = "w " * 7_990 + key
    clipped = log_shipping.clip(message, log_shipping.MAX_MESSAGE_CHARS)
    assert len(clipped) <= log_shipping.MAX_MESSAGE_CHARS
    assert "sk-" not in clipped and clipped.endswith(log_shipping.CLIP_MARK)
    assert log_shipping.clip("short", 100) == "short"
    # One token longer than the bound leaves nothing of it.
    assert log_shipping.clip("t" * 50, 20) == log_shipping.CLIP_MARK


def test_a_push_the_wire_refused_is_dropped_not_retried_forever() -> None:
    handler, calls, flusher, clock = flusher_rig()
    calls.answer = Answer(ok=False, error_class="malformed_payload")
    handler.handle(record("run_agent", logging.ERROR))
    asyncio.run(flusher.tick())
    assert len(handler) == 0 and handler.dropped == 1
    handler.handle(record("run_agent", logging.ERROR, "the next line"))
    clock.now += log_shipping.FAILURE_BACKOFF_SECONDS
    calls.answer = Answer(payload={"accepted": 1})
    asyncio.run(flusher.tick())
    assert calls.pushed[-1]["entries"][0]["message"] == "the next line"


def test_a_tick_that_raises_does_not_end_shipping(monkeypatch: pytest.MonkeyPatch) -> None:
    handler, calls, flusher, _ = flusher_rig()
    monkeypatch.setattr(log_shipping, "FLUSH_EVERY_SECONDS", 0)
    ticks = {"n": 0}

    async def flaky() -> None:
        ticks["n"] += 1
        if ticks["n"] == 1:
            raise ValueError("an unexpected answer")

    flusher.tick = flaky  # type: ignore[method-assign]

    async def scenario() -> None:
        task = asyncio.ensure_future(flusher.run())
        for _ in range(20):
            await asyncio.sleep(0)
        task.cancel()
        (outcome,) = await asyncio.gather(task, return_exceptions=True)
        assert isinstance(outcome, asyncio.CancelledError)

    asyncio.run(scenario())
    assert ticks["n"] > 1


def test_a_push_that_raises_puts_its_batch_back() -> None:
    """🔴 #2259 review: the batch is out of the queue before the push. A push that
    raised — a dropped tunnel, a cancel at shutdown — lost it, uncounted."""
    handler, calls, flusher, _ = flusher_rig()
    handler.handle(record("run_agent", logging.ERROR, "kept"))

    async def broken(*_args: Any, **_kwargs: Any) -> Answer:
        raise ConnectionError("the tunnel dropped")

    calls.call = broken  # type: ignore[method-assign]
    with pytest.raises(ConnectionError):
        asyncio.run(flusher.tick())
    assert len(handler) == 1 and handler.dropped == 0

    del calls.call  # the recording client again
    asyncio.run(flusher.tick())
    assert [e["message"] for e in calls.pushed[-1]["entries"]] == ["kept"]


def test_a_push_in_flight_at_disconnect_finishes_and_nothing_ships_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """🔴 #2313 review: a disconnect that CANCELLED the periodic push mid-flight put back
    a batch the companion had usually already taken, so the final push sent it again —
    or was answered busy and sent nothing. The disconnect waits for that push instead."""
    monkeypatch.setattr(log_shipping, "FLUSH_EVERY_SECONDS", 0)
    release = asyncio.Event()
    started = asyncio.Event()
    calls = AdapterCalls()
    in_flight = {"now": 0, "most": 0}
    original = calls.call

    async def slow(op: str, payload: dict[str, Any], *, deadline_ms: int) -> Answer:
        in_flight["now"] += 1
        in_flight["most"] = max(in_flight["most"], in_flight["now"])
        try:
            answer = await original(op, payload, deadline_ms=deadline_ms)
            started.set()
            await release.wait()
            return answer
        finally:
            in_flight["now"] -= 1

    calls.call = slow  # type: ignore[method-assign]

    async def scenario() -> None:
        adapter = adapter_with_calls(calls)
        assert await adapter.connect()
        logging.getLogger("gateway.run").warning("first")
        await asyncio.wait_for(started.wait(), 1.0)  # the periodic push is mid-flight
        logging.getLogger("gateway.run").warning("second")
        stopping = asyncio.ensure_future(adapter.disconnect())
        await asyncio.sleep(0.01)
        release.set()
        await asyncio.wait_for(stopping, 1.0)

    asyncio.run(scenario())
    shipped = [e["message"] for p in calls.pushed for e in p["entries"]]
    assert shipped == ["first", "second"], shipped
    assert in_flight["most"] == 1, "two pushes overlapped; the companion takes one at a time"
