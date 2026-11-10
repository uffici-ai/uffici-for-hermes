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


#: The pod shape's configured address: every real caller names one (#3119), and a test
#: that is about something else connects the way the e2e pod does.
POD_URL = "http://127.0.0.1:7411"
ADDRESS_ENV = ("UFFICI_WIRE_SOCKET", "UFFICI_COMPANION_URL", "UFFICI_WIRE_TOKEN_FILE")


def adapter_with(*, address: bool = True, **env) -> tuple[UfficiAdapter, list]:
    """An adapter whose wire is a fake; ``address=False`` configures no address at all."""
    opened: list = []
    if address and not any(name in env for name in ADDRESS_ENV):
        env["UFFICI_COMPANION_URL"] = POD_URL

    async def connect_fn(address):
        connection = FakeConnection()
        connection.address = address
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
        # 🔴 #2236: the installed version, never a constant — a source checkout says so.
        installed = adapter_module.plugin_version()
        assert hello["pluginVersion"] == installed == adapter_module.DEV_VERSION

    asyncio.run(scenario())


def test_no_tcp_dial_happens_without_a_configured_address(tmp_path, caplog) -> None:
    """🔴 #3119: with no address configured the plugin dials the companion's socket beside
    the token file, and with no token file either it refuses — it never dials
    ``127.0.0.1:7411``, a port any local user could bind first and so receive the wire
    token and feed the harness forged rings."""
    caplog.set_level(logging.ERROR)
    token = tmp_path / "wire-token"
    token.write_text("t" * 40, encoding="utf-8")

    async def scenario(**env):
        adapter, opened = adapter_with(address=False, UFFICI_CONTEXT_WINDOW_TOKENS="100000", **env)
        connected = await adapter.connect()
        for _ in range(50):
            await asyncio.sleep(0)
        await adapter.disconnect()
        return connected, [connection.address for connection in opened]

    connected, dialled = asyncio.run(scenario(UFFICI_WIRE_TOKEN_FILE=str(token)))
    assert connected is True and dialled, "connect() never dialled the companion"
    assert {address.kind for address in dialled} == {"socket"}, "a TCP address was dialled"
    assert dialled[0].socket_path == str(tmp_path / "wire.sock")

    connected, dialled = asyncio.run(scenario(UFFICI_WIRE_TOKEN="t" * 40))
    assert connected is False and dialled == [], "dialled with no address configured"
    assert "no wire address" in caplog.text


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


def _stand_in_hermes_config(monkeypatch, model, resolved: list, window=272_000) -> None:
    """``hermes_cli.config.load_config`` and ``agent.model_metadata`` as a real install has them.

    Transcribed from the machine #2230 was found on: ``load_config()["model"]`` is
    ``{default, provider, base_url[, context_length]}``, and ``get_model_context_length``
    takes the model positionally and the rest by keyword.
    """
    import sys
    import types

    config = types.ModuleType("hermes_cli.config")
    # Only the read-only loader answers, so a probe that stopped preferring it would
    # read nothing and the tests below would fail.
    config.load_config_readonly = lambda: {"model": model}
    config.load_config = lambda: {}
    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.config = config
    metadata = types.ModuleType("agent.model_metadata")

    def get_model_context_length(
        model,
        base_url="",
        api_key="",
        config_context_length=None,
        provider="",
        custom_providers=None,
    ):
        resolved.append(
            {
                "model": model,
                "base_url": base_url,
                "provider": provider,
                "config_context_length": config_context_length,
            }
        )
        return config_context_length or window

    metadata.get_model_context_length = get_model_context_length
    # The cache probe the plugin used to rely on answers nothing, as it did on a real install
    # for the session store's model (#2230).
    metadata.get_cached_context_length = lambda model, base_url="": None
    agent = types.ModuleType("agent")
    agent.model_metadata = metadata
    for name, module in {
        "hermes_cli": hermes_cli,
        "hermes_cli.config": config,
        "agent": agent,
        "agent.model_metadata": metadata,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_with_no_override_the_window_is_the_one_hermes_resolves_for_its_own_model(
    monkeypatch,
) -> None:
    """🔴 #2230: on a real install the plugin never started.

    Both old probes read the model from the gateway's SESSION-STORE config, which
    carries no model, so neither ever asked Hermes. Hermes's own configuration names
    the model, and its own resolver answers for it — the window Hermes compacts
    against, so declaring it invents nothing.
    """
    resolved: list = []
    _stand_in_hermes_config(
        monkeypatch,
        {"default": "gpt-6-luna", "provider": "openai-codex", "base_url": "https://codex.example"},
        resolved,
    )

    async def scenario():
        adapter, opened = adapter_with(UFFICI_WIRE_TOKEN="t" * 40)
        assert await adapter.connect() is True
        for _ in range(50):
            await asyncio.sleep(0)
        await adapter.disconnect()

        import json

        hello = json.loads(opened[0].sent[0])
        assert hello["contextBudgetTokens"] == int(272_000 * 0.25 * TOKENIZER_DRIFT_MARGIN)

    asyncio.run(scenario())
    assert resolved == [
        {
            "model": "gpt-6-luna",
            "base_url": "https://codex.example",
            "provider": "openai-codex",
            "config_context_length": None,
        }
    ]


def test_hermes_configured_context_length_reaches_its_resolver(monkeypatch) -> None:
    """``model.context_length`` is Hermes's own override; the resolver is handed it, as
    Hermes's agent does, and it is what Hermes answers."""
    resolved: list = []
    _stand_in_hermes_config(
        monkeypatch,
        {"default": "mock-model", "provider": "modelmock", "context_length": "32000"},
        resolved,
    )
    adapter, _ = adapter_with()
    assert adapter._window_probes()[0]() == 32_000
    assert resolved[0]["config_context_length"] == 32_000


def test_a_bare_model_string_in_hermes_config_is_still_asked_about(monkeypatch) -> None:
    resolved: list = []
    _stand_in_hermes_config(monkeypatch, "anthropic/claude-x", resolved, window=200_000)
    adapter, _ = adapter_with()
    assert adapter._window_probes()[0]() == 200_000
    assert resolved[0]["model"] == "anthropic/claude-x"


def test_a_hermes_config_naming_no_model_is_a_fallback_not_a_guess(monkeypatch) -> None:
    resolved: list = []
    _stand_in_hermes_config(monkeypatch, {"provider": "openrouter"}, resolved)
    adapter, _ = adapter_with()
    assert adapter._window_probes()[0]() is None
    assert resolved == []


def test_the_operators_override_still_beats_hermes(monkeypatch) -> None:
    """The override is for CORRECTING Hermes, and correcting means winning."""
    resolved: list = []
    _stand_in_hermes_config(monkeypatch, {"default": "m", "provider": "p"}, resolved)

    async def scenario():
        adapter, opened = adapter_with(
            UFFICI_WIRE_TOKEN="t" * 40, UFFICI_CONTEXT_WINDOW_TOKENS="100000"
        )
        assert await adapter.connect() is True
        for _ in range(50):
            await asyncio.sleep(0)
        await adapter.disconnect()
        import json

        assert json.loads(opened[0].sent[0])["contextBudgetTokens"] == int(
            100_000 * 0.25 * TOKENIZER_DRIFT_MARGIN
        )

    asyncio.run(scenario())
    assert resolved == []


def test_a_probe_reaching_a_module_that_is_not_there_is_survivable() -> None:
    """🔴 The second probe imports ``agent.model_metadata``, which exists only
    inside Hermes. Outside it the import raises, and that must cost a fallback
    rather than a refusal to start."""

    class Store:
        config = None

    adapter, _ = adapter_with(UFFICI_WIRE_TOKEN="t" * 40)
    adapter.set_session_store(Store())
    probes = adapter._window_probes()
    assert len(probes) == 3
    # Hermes's own configuration and its resolver exist only inside Hermes: the
    # first probe raises ImportError here, the session store answers None (no
    # config), and the cache probe raises ImportError.
    for probe in (probes[0], probes[2]):
        try:
            probe()
        except Exception as exc:
            assert isinstance(exc, ImportError | AttributeError | TypeError)
    assert probes[1]() is None
    try:
        probes[2]()
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


def test_a_store_that_can_only_be_looked_up_is_never_read_as_resumed() -> None:
    """🔴 #2414: an index lookup is not Hermes's decision. A store offering ``get_entry``
    (or ``_entries``) and no routing call cannot say whether the session survives the
    route, so the answer is the safe one."""

    class Store:
        _entries = {"agent:main:uffici:dm:conv_known": object()}

        def get_entry(self, key: str):
            return object()

    adapter, _ = adapter_with()
    adapter.set_session_store(Store())
    known = adapter.build_source(chat_id="conv_known", chat_type="dm")
    assert asyncio.run(adapter._session_is_fresh(known)) is True


def test_no_store_at_all_means_fresh() -> None:
    adapter, _ = adapter_with()
    source = adapter.build_source(chat_id="c", chat_type="dm")
    assert asyncio.run(adapter._session_is_fresh(source)) is True


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
        adapter, _ = adapter_with(UFFICI_WIRE_TOKEN="t" * 40, UFFICI_CONTEXT_WINDOW_TOKENS="32000")
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
