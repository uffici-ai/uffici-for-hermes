"""Starting up: what the adapter reads, what it refuses, and what it declares."""

from __future__ import annotations

import asyncio
import logging

from conftest import PlatformConfig
from uffici_for_hermes import adapter as adapter_module
from uffici_for_hermes.adapter import UfficiAdapter
from uffici_for_hermes.budget import TOKENIZER_DRIFT_MARGIN


class FakeConnection:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, raw: str) -> None:
        self.sent.append(raw)

    async def close(self) -> None:
        return None

    def __aiter__(self):
        async def gen():
            return
            yield  # pragma: no cover

        return gen()


def adapter_with(**env) -> tuple[UfficiAdapter, list]:
    opened: list = []

    async def connect_fn(_address):
        connection = FakeConnection()
        opened.append(connection)
        return connection

    adapter = UfficiAdapter(
        PlatformConfig(enabled=True),
        get_env=lambda name: env.get(name),
        connect_fn=connect_fn,
    )
    return adapter, opened


def test_connect_declares_a_budget_derived_from_the_operators_window() -> None:
    async def scenario():
        adapter, opened = adapter_with(
            UFFICI_WIRE_TOKEN="t" * 40,
            UFFICI_CONTEXT_WINDOW_TOKENS="100000",
            UFFICI_CONTEXT_SHARE="0.5",
        )
        assert await adapter.connect() is True
        # Let the spawned wire loop reach its hello.
        for _ in range(50):
            await asyncio.sleep(0)
        await adapter.disconnect()

        assert opened, "connect() never opened a connection"
        import json

        hello = json.loads(opened[0].sent[0])
        assert hello["type"] == "hello"
        assert hello["contextBudgetTokens"] == int(100_000 * 0.5 * TOKENIZER_DRIFT_MARGIN)
        assert hello["harness"] == "hermes"

    asyncio.run(scenario())


def test_connect_refuses_without_a_token_and_the_refusal_names_the_variable(caplog) -> None:
    caplog.set_level(logging.ERROR)

    async def scenario():
        adapter, opened = adapter_with(UFFICI_CONTEXT_WINDOW_TOKENS="1000")
        assert await adapter.connect() is False
        assert not opened, "a connection was opened without a token"

    asyncio.run(scenario())
    assert "UFFICI_WIRE_TOKEN" in caplog.text


def test_connect_refuses_when_no_window_can_be_learned(caplog) -> None:
    """🔴 Rather than declaring a number it invented. The platform sizes its
    reads to what this says, and it would believe a guess."""
    caplog.set_level(logging.ERROR)

    async def scenario():
        adapter, opened = adapter_with(UFFICI_WIRE_TOKEN="t" * 40)
        assert await adapter.connect() is False
        assert not opened

    asyncio.run(scenario())
    assert "UFFICI_CONTEXT_WINDOW_TOKENS" in caplog.text


def test_the_refusal_carries_no_token_and_no_path(caplog, tmp_path) -> None:
    caplog.set_level(logging.DEBUG)
    secret_dir = tmp_path / "very-secret-home"
    secret_dir.mkdir()

    async def scenario():
        adapter, _ = adapter_with(UFFICI_WIRE_TOKEN_FILE=str(secret_dir / "wire-token"))
        assert await adapter.connect() is False

    asyncio.run(scenario())
    assert "very-secret-home" not in caplog.text


def test_the_window_comes_from_the_session_stores_configuration() -> None:
    """The first probe: Hermes's own configuration, as the adapter can reach it."""

    class Config:
        context_length = 64_000

    class Store:
        config = Config()

    async def scenario():
        adapter, opened = adapter_with(UFFICI_WIRE_TOKEN="t" * 40)
        adapter.set_session_store(Store())
        assert await adapter.connect() is True
        for _ in range(50):
            await asyncio.sleep(0)
        await adapter.disconnect()

        import json

        hello = json.loads(opened[0].sent[0])
        expected = int(64_000 * 0.25 * TOKENIZER_DRIFT_MARGIN)
        assert hello["contextBudgetTokens"] == expected

    asyncio.run(scenario())


def test_a_probe_reaching_a_module_that_is_not_there_is_survivable() -> None:
    """🔴 The second probe imports ``agent.model_metadata``, which exists only
    inside Hermes. Outside it the import raises, and that must cost a fallback
    rather than a refusal to start."""

    class Store:
        config = None

    adapter, _ = adapter_with(UFFICI_WIRE_TOKEN="t" * 40)
    adapter.set_session_store(Store())
    probes = adapter._window_probes()
    assert len(probes) == 2
    # The first answers None (no config), the second raises ImportError.
    assert probes[0]() is None
    try:
        probes[1]()
    except Exception as exc:
        assert isinstance(exc, ImportError | AttributeError | TypeError)
    else:  # pragma: no cover - only if a real Hermes is importable here
        pass


def test_the_harness_version_is_stated_unknown_rather_than_guessed() -> None:
    adapter, _ = adapter_with()
    # Hermes is not importable in this environment, which is the point.
    assert adapter._harness_version() == "unknown"


def test_a_session_reset_that_fails_does_not_take_the_turn_with_it() -> None:
    class Store:
        _entries: dict = {}

        def reset_session(self, key: str) -> None:
            raise RuntimeError("the store moved")

    async def scenario():
        adapter, _ = adapter_with()
        adapter.set_session_store(Store())
        adapter._reset_session(adapter.build_source(chat_id="conv_1", chat_type="dm"))

    asyncio.run(scenario())


def test_a_store_without_a_reset_is_not_an_error() -> None:
    adapter, _ = adapter_with()
    adapter.set_session_store(object())
    adapter._reset_session(adapter.build_source(chat_id="conv_1", chat_type="dm"))


def test_a_store_exposing_get_entry_instead_of_entries_is_read() -> None:
    class Store:
        def get_entry(self, key: str):
            return object() if "conv_known" in key else None

    adapter, _ = adapter_with()
    adapter.set_session_store(Store())
    known = adapter.build_source(chat_id="conv_known", chat_type="dm")
    unknown = adapter.build_source(chat_id="conv_other", chat_type="dm")
    assert adapter._session_exists(known) is True
    assert adapter._session_exists(unknown) is False


def test_no_store_at_all_means_fresh() -> None:
    adapter, _ = adapter_with()
    assert adapter._session_exists(adapter.build_source(chat_id="c", chat_type="dm")) is False


# --------------------------------------------------------------- readiness


def test_the_harness_is_not_ready_until_its_message_handler_is_installed() -> None:
    """🔴 The signal, and Hermes's own words for why it is the right one: with
    no message handler, "every inbound message is discarded" on an adapter that
    is otherwise connected and can send."""
    adapter, _ = adapter_with()
    assert adapter.harness_is_ready() is False

    adapter._message_handler = lambda event: None
    assert adapter.harness_is_ready() is True


def test_ready_is_sent_only_once_the_harness_can_answer() -> None:
    """🔴 THE FRAME'S WHOLE POINT (#2014).

    `connect()` runs while Hermes is still starting. Announcing readiness there
    would be the same lie the frame exists to stop — the companion would deliver
    the first brief of the run into a window where nothing can answer it.
    """

    async def scenario():
        adapter, _ = adapter_with(
            UFFICI_WIRE_TOKEN="t" * 40, UFFICI_CONTEXT_WINDOW_TOKENS="32000"
        )
        # The poll is a real sleep; spin it so the test does not wait on a clock.
        adapter_module.READY_POLL_SECONDS = 0
        sent: list[dict] = []

        class Wire:
            async def send(self, frame: dict) -> bool:
                sent.append(frame)
                return True

            async def stop(self) -> None:
                return None

        assert await adapter.connect() is True
        adapter._client = Wire()
        # 🔴 The announcement hangs off the COMPANION'S HELLO, not off connect:
        # a frame sent before the wire is up is dropped, which is exactly what
        # happened on the first live run — `no wire connection; dropping frame`,
        # and the companion then timed out waiting for a plugin that believed it
        # had already said it was ready.
        await adapter._on_companion_hello({"v": 1, "type": "hello"})

        # The harness is still starting: nothing is announced.
        for _ in range(20):
            await asyncio.sleep(0)
        assert [f for f in sent if f.get("type") == "ready"] == [], (
            "ready went out before the harness could answer"
        )

        # Hermes wires the adapter up.
        adapter._message_handler = lambda event: None
        for _ in range(200):
            await asyncio.sleep(0)
            if any(f.get("type") == "ready" for f in sent):
                break

        ready = [f for f in sent if f.get("type") == "ready"]
        assert ready, "ready never went out, so the companion holds every ring for ever"
        assert ready[0]["v"] == 1
        await adapter.disconnect()

    asyncio.run(scenario())


def test_a_harness_that_never_becomes_ready_says_so_rather_than_hanging_quietly(caplog) -> None:
    """A held ring answered late is recoverable; a silent daemon is not."""
    caplog.set_level(logging.ERROR)

    async def scenario():
        adapter, _ = adapter_with()
        original = adapter_module.READY_TIMEOUT_SECONDS
        adapter_module.READY_TIMEOUT_SECONDS = 0.0
        try:
            await adapter._announce_ready_when_able()
        finally:
            adapter_module.READY_TIMEOUT_SECONDS = original

    asyncio.run(scenario())
    assert "did not become able to answer" in caplog.text
