"""The harness's drift, read for the hello (#2406): each kind, its bounds, and never a raise."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest

from test_connect import adapter_with
from test_memory import setup_config
from test_wire_client import FakeCompanion
from uffici_for_hermes import drift
from uffici_for_hermes import memory as mem
from uffici_for_hermes.adapter import PLATFORM_NAME
from uffici_for_hermes.wire_client import WireAddress, WireClient, plugin_hello

WIRE_V1 = Path(drift.__file__).parent / "wire" / "v1"


def drifted_config() -> dict:
    """The issue's fixture: an extra MCP server, a second enabled platform, and Hermes's own
    memory switched back on — over what `uffici setup` writes."""
    config = setup_config()
    config["mcp_servers"] = {"github": {"url": "https://example.invalid/mcp"}}
    config["memory"]["memory_enabled"] = True
    return config


def fake_mcp(config: dict) -> list[str]:
    """Hermes's reading, for the fixture: every server whose ``enabled`` is not off."""
    servers = config.get("mcp_servers") or {}
    return [name for name, server in servers.items() if server.get("enabled", True) is not False]


def test_each_kind_is_read_from_the_fixture_config() -> None:
    report = drift.read_drift(
        PLATFORM_NAME,
        load_config=drifted_config,
        mcp_servers=fake_mcp,
        platforms=lambda: [PLATFORM_NAME, "telegram"],
    )
    assert report == {
        "mcpServers": ["github"],
        "channels": ["telegram"],
        "localMemory": ["memory.memory_enabled"],
    }


def test_a_clean_harness_reports_every_kind_read_and_empty() -> None:
    report = drift.read_drift(
        PLATFORM_NAME,
        load_config=setup_config,
        mcp_servers=fake_mcp,
        platforms=lambda: [PLATFORM_NAME],
    )
    assert report == {"mcpServers": [], "channels": [], "localMemory": []}


def test_uffici_is_never_a_channel_of_drift() -> None:
    report = drift.read_drift(
        PLATFORM_NAME,
        load_config=setup_config,
        mcp_servers=fake_mcp,
        platforms=lambda: [PLATFORM_NAME],
    )
    assert report["channels"] == []


def test_a_source_that_cannot_be_read_leaves_only_its_kind_out(caplog) -> None:
    """🔴 Absent, never empty: the doctor says "not reported" for an absent kind, and an
    empty list would read as a clean harness."""
    caplog.set_level(logging.INFO)

    def broken(*_args):
        raise RuntimeError("secret-looking detail https://example.invalid/x")

    report = drift.read_drift(
        PLATFORM_NAME, load_config=drifted_config, mcp_servers=broken, platforms=lambda: []
    )
    assert "mcpServers" not in report
    assert report["channels"] == [] and report["localMemory"] == ["memory.memory_enabled"]

    report = drift.read_drift(PLATFORM_NAME, load_config=broken, platforms=lambda: ["slack"])
    assert report == {"channels": ["slack"]}, "an unreadable config leaves its two kinds out"

    assert drift.read_drift(PLATFORM_NAME, load_config=broken, platforms=broken) == {}
    assert "RuntimeError" in caplog.text
    assert "example.invalid" not in caplog.text, "an error's text never reaches a log line"


def test_outside_hermes_nothing_can_be_read_and_nothing_raises() -> None:
    """The real readers, with no Hermes to import: every kind unread, no exception."""
    assert drift.read_drift(PLATFORM_NAME) == {}


def test_names_are_flattened_clipped_deduplicated_and_capped() -> None:
    names = ["b", "a", "a", "  ", "tab\there", "bell\x07", "x" * 100]
    assert drift.bounded_names(names) == ["a", "b", "bell", "tab here", "x" * 64]
    assert drift.bounded_names([f"s{i:03}" for i in range(50)]) == [f"s{i:03}" for i in range(32)]


def test_the_bounds_are_the_wires() -> None:
    """🔴 The plugin clips so the companion never refuses a hello over drift: its bounds are
    the schema's, read from the plugin's own copy of the wire."""
    schema = json.loads((WIRE_V1 / "schema" / "hello.schema.json").read_text(encoding="utf-8"))
    kinds = schema["$defs"]["pluginHello"]["properties"]["drift"]["properties"]
    for kind in ("mcpServers", "channels"):
        assert kinds[kind]["maxItems"] == drift.MAX_NAMES
        assert kinds[kind]["items"]["maxLength"] == drift.MAX_NAME_CHARS
    assert kinds["localMemory"]["maxItems"] == drift.MAX_LOCAL_MEMORY
    assert kinds["localMemory"]["items"]["maxLength"] >= max(
        len(key) for key in (drift.PROVIDER_KEY, *drift.BUILTIN_STORE_KEYS)
    )


def test_the_hello_carries_drift_only_when_read() -> None:
    report = {"mcpServers": ["github"], "channels": [], "localMemory": []}
    hello = plugin_hello("t" * 40, "0.21.5", "0.1.0", 1000, report)
    golden = json.loads((WIRE_V1 / "golden" / "valid" / "hello-plugin-with-drift.json").read_text())
    assert hello["drift"] == report and set(golden["drift"]) == set(report)
    assert "drift" not in plugin_hello("t" * 40, "0.21.5", "0.1.0", 1000)


@pytest.mark.real_drift_read
@pytest.mark.parametrize("reader", ["reports", "raises"])
def test_connect_sends_the_drift_and_a_failing_read_never_stops_it(reader: str) -> None:
    """🔴 Requirement 4: drift never blocks a connection."""

    def read(own: str) -> dict:
        assert own == PLATFORM_NAME
        if reader == "raises":
            raise RuntimeError("boom")
        return {"mcpServers": ["github"], "channels": [], "localMemory": []}

    async def scenario():
        adapter, opened = adapter_with(
            UFFICI_WIRE_TOKEN="t" * 40, UFFICI_CONTEXT_WINDOW_TOKENS="100000"
        )
        adapter._drift_reader = read  # noqa: SLF001
        assert await adapter.connect() is True
        # The hello is the event: the read runs on a thread, so no count of loop ticks is it.
        for _ in range(500):
            if opened and opened[0].sent:
                break
            await asyncio.sleep(0.01)
        await adapter.disconnect()
        return json.loads(opened[0].sent[0])

    hello = asyncio.run(scenario())
    want = (
        {} if reader == "raises" else {"mcpServers": ["github"], "channels": [], "localMemory": []}
    )
    assert hello["drift"] == want


def test_every_hello_reads_the_drift_again() -> None:
    """A reconnect after the harness was fixed reports the fix, not the first read."""
    reads = iter([{"mcpServers": ["github"]}, {"mcpServers": []}])
    opened: list[FakeCompanion] = []
    order: list[str] = []

    async def read_drift() -> dict:
        order.append("read")
        return next(reads)

    async def connect(_address):
        order.append("dial")
        opened.append(FakeCompanion())
        if len(opened) == 2:
            await client.stop()
        return opened[-1]

    async def no_frame(_frame) -> None:
        return None

    client = WireClient(
        address=WireAddress(kind="socket", socket_path="/tmp/x", url="ws://companion/v1/ring"),
        token="t" * 40,
        harness_version="0.21.5",
        plugin_version="0.1.0",
        budget_tokens=1000,
        connect_fn=connect,
        on_frame=no_frame,
        drift=read_drift,
        sleep_fn=lambda _s: asyncio.sleep(0),
    )
    asyncio.run(client.run())
    hellos = [companion.sent[0] for companion in opened]
    assert [hello["drift"] for hello in hellos] == [{"mcpServers": ["github"]}, {"mcpServers": []}]
    # 🔴 Before each dial: the companion claims its one plugin slot on accept.
    assert order == ["read", "dial", "read", "dial"]


def test_another_provider_is_local_memory_and_no_provider_is_not() -> None:
    """No provider keeps no notes of Hermes's own — only Uffici's memory goes unrecalled,
    which the connect-time warning says; another provider is a memory of its own."""
    config = setup_config()
    config["memory"]["provider"] = "honcho"
    assert drift.local_memory_keys(config) == [drift.PROVIDER_KEY]
    del config["memory"]["provider"]
    assert drift.local_memory_keys(config) == []
    assert "memory.provider is unset" in mem.memory_drift(config)[0]
