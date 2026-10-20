"""The agent's Uffici memory in Hermes: the provider, prefetch, the four tools, the rendezvous."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from typing import Any

import pytest

from conftest import MemoryProvider, ProviderCollector, RecallStatus, set_session_vars
from test_platform_tools import Registry
from uffici_for_hermes import memory as mem
from uffici_for_hermes import register
from uffici_for_hermes.calls import CallResult
from uffici_for_hermes.memory import (
    COMPANION_WITHOUT_MEMORY,
    DEGRADED_EMPTY,
    DEGRADED_NOTE,
    NO_REQUEST,
    NOT_CONNECTED,
    PREFETCH_DEADLINE_MS,
    PREFETCH_WAIT_SECONDS,
    RECALL_HEADING,
    TOOL_OPS,
    TOOL_SCHEMAS,
    UNREADABLE,
    MemoryTools,
    build_provider,
    is_memory_collector,
    render_recall,
)
from uffici_for_hermes.platform_tools import OUTSIDE_A_REQUEST, TOOLSET
from uffici_for_hermes.wire import WIRE_V1

REQUEST = "11111111-1111-4111-8111-111111111111"
CONVERSATION = "conv-1"
FACT = "urn:uffici:fact:9c8b7a6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"


def op_golden(name: str) -> dict:
    return json.loads((WIRE_V1 / "golden" / "ops" / "valid" / f"{name}.json").read_text())


def request_schema(op: str) -> dict:
    return json.loads((WIRE_V1 / "ops" / f"{op}.request.schema.json").read_text())


LEVELS = op_golden("memory.recall.result-levels")
DEGRADED = op_golden("memory.recall.result-degraded")


class Tunnel:
    """The call client's ``call``: records each call, answers from a table."""

    def __init__(self, answers: dict[str, CallResult] | None = None) -> None:
        self.calls: list[tuple[str, dict, int]] = []
        self.answers = answers or {}

    async def call(self, op: str, payload: dict | None = None, *, deadline_ms: int) -> CallResult:
        self.calls.append((op, payload or {}, deadline_ms))
        if op in self.answers:
            return self.answers[op]
        if op == "memory.recall":
            return CallResult(call_id="c", payload=LEVELS)
        return CallResult(call_id="c", payload={"urn": FACT})


def failed(error_class: str, instruction: str | None = None) -> CallResult:
    return CallResult(call_id="c", error_class=error_class, instruction=instruction)


def memory_for(
    *,
    registry: Registry | None = None,
    tunnel: Tunnel | None = None,
    ledger: dict[str, str] | None = None,
    available: bool = True,
    connected: bool = True,
    session: Any = None,
) -> tuple[MemoryTools, Registry, Tunnel, dict[str, str]]:
    registry = registry or Registry()
    tunnel = tunnel or Tunnel()
    ledger = {CONVERSATION: REQUEST} if ledger is None else ledger
    flags = {"available": available, "connected": connected}
    tools = MemoryTools(
        registry.register_tool,
        call=tunnel.call,
        request_for=ledger.get,
        available=lambda: flags["available"],
        connected=lambda: flags["connected"],
        session_reader=session or (lambda: CONVERSATION),
    )
    tools.flags = flags  # type: ignore[attr-defined]
    return tools, registry, tunnel, ledger


@pytest.fixture
def loop_thread():
    """A running event loop on its own thread — the gateway's loop, seen from Hermes's
    worker thread, which is where every handler and prefetch runs."""
    loop = asyncio.new_event_loop()
    runner = threading.Thread(target=loop.run_forever, daemon=True)
    runner.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    runner.join(timeout=5)
    loop.close()


# ------------------------------------------------------------------ rendering


def test_a_recall_renders_each_hit_with_its_level_and_urn_under_one_heading() -> None:
    text, count = render_recall(LEVELS)
    assert count == 1
    assert text.splitlines() == [
        RECALL_HEADING,
        f"- [conversation] The Lisbon client is now invoiced on the 10th. ({FACT})",
    ]


def test_the_rendered_recall_carries_no_fence_hermes_adds_its_own() -> None:
    text, _ = render_recall(LEVELS)
    assert "<memory-context>" not in text and "</memory-context>" not in text


def test_a_multi_line_fact_stays_under_its_own_bullet() -> None:
    hit = {**LEVELS["hits"][0], "text": "First line.\nSecond line."}
    text, _ = render_recall({"hits": [hit], "degraded": False})
    assert text.splitlines()[1:] == [f"- [conversation] First line. ({FACT})", "  Second line."]


def test_a_degraded_recall_says_words_stood_in_for_meaning() -> None:
    text, count = render_recall(DEGRADED)
    assert count == 1 and text.endswith(DEGRADED_NOTE)
    assert "- [folder] The Porto warehouse closes at 18:00." in text


def test_nothing_matched_is_empty_and_a_degraded_nothing_says_it_may_have_missed() -> None:
    assert render_recall({"hits": [], "degraded": False}) == ("", 0)
    assert render_recall({"hits": [], "degraded": True}) == (DEGRADED_EMPTY, 0)
    assert render_recall({"hits": [], "degraded": False, "missing": ["embeddings"]}) == (
        DEGRADED_EMPTY,
        0,
    )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"hits": []},
        {"hits": "x", "degraded": False},
        {"hits": ["x"], "degraded": False},
        {"hits": [{"level": "folder"}], "degraded": False},
    ],
)
def test_a_result_outside_the_wire_is_not_rendered(payload: Any) -> None:
    assert render_recall(payload) is None


# ------------------------------------------------------------------ prefetch


def test_prefetch_recalls_on_the_turns_request_inside_the_plugins_deadline(loop_thread) -> None:
    tools, _registry, tunnel, _ = memory_for()
    tools.bind_loop(loop_thread)
    text, count = tools.prefetch(CONVERSATION, "When is Lisbon invoiced?")
    assert count == 1 and FACT in text
    op, payload, deadline = tunnel.calls[0]
    assert op == "memory.recall" and deadline == PREFETCH_DEADLINE_MS
    assert payload == {
        "requestId": REQUEST,
        "query": "When is Lisbon invoiced?",
        "deadlineMs": PREFETCH_DEADLINE_MS,
    }


def test_the_prefetch_budget_sits_inside_hermess_bound() -> None:
    # 🔴 Hermes gives an external provider 8 s and renders NOTHING when it fires; every
    # bound of the plugin's must fire first, so the turn reads an instruction instead.
    assert PREFETCH_DEADLINE_MS / 1000 < PREFETCH_WAIT_SECONDS < mem.HERMES_PREFETCH_BOUND_SECONDS


def test_a_long_turn_is_recalled_by_its_end(loop_thread) -> None:
    tools, _r, tunnel, _ = memory_for()
    tools.bind_loop(loop_thread)
    tools.prefetch(CONVERSATION, "a" * 9_000 + "THE END")
    query = tunnel.calls[0][1]["query"]
    assert len(query) == mem.MAX_QUERY_CHARS and query.endswith("THE END")


@pytest.mark.parametrize(
    ("error_class", "instruction"),
    [
        ("upstream_failed", "Memory was unavailable because of a system problem; tell the person."),
        ("timed_out", None),
        ("not_permitted", "This memory request names a request you are not answering."),
    ],
)
def test_a_failed_recall_is_the_instruction_never_empty(
    loop_thread, error_class: str, instruction: str | None
) -> None:
    tunnel = Tunnel({"memory.recall": failed(error_class, instruction)})
    tools, *_ = memory_for(tunnel=tunnel)
    tools.bind_loop(loop_thread)
    text, count = tools.prefetch(CONVERSATION, "anything")
    assert count == 0 and text.startswith("Uffici memory could not be checked")
    assert (instruction or mem.INSTRUCTIONS[error_class]) in text


def test_a_recall_answer_outside_the_wire_reads_as_unreadable(loop_thread) -> None:
    tunnel = Tunnel({"memory.recall": CallResult(call_id="c", payload={"hits": "x"})})
    tools, *_ = memory_for(tunnel=tunnel)
    tools.bind_loop(loop_thread)
    assert tools.prefetch(CONVERSATION, "anything") == (UNREADABLE, 0)


def test_a_disconnected_or_memoryless_companion_or_no_request_each_say_so(loop_thread) -> None:
    tools, _r, tunnel, ledger = memory_for(connected=False)
    tools.bind_loop(loop_thread)
    assert tools.prefetch(CONVERSATION, "q") == (NOT_CONNECTED, 0)
    tools.flags["connected"], tools.flags["available"] = True, False  # type: ignore[attr-defined]
    assert tools.prefetch(CONVERSATION, "q") == (COMPANION_WITHOUT_MEMORY, 0)
    tools.flags["available"] = True  # type: ignore[attr-defined]
    ledger.clear()
    assert tools.prefetch(CONVERSATION, "q") == (NO_REQUEST, 0)
    assert tunnel.calls == []


def test_an_empty_query_recalls_nothing_and_asks_nothing(loop_thread) -> None:
    tools, _r, tunnel, _ = memory_for()
    tools.bind_loop(loop_thread)
    assert tools.prefetch(CONVERSATION, "   ") == ("", 0)
    assert tunnel.calls == []


def test_a_recall_that_outlives_the_wait_is_the_timed_out_instruction(monkeypatch) -> None:
    loop = asyncio.new_event_loop()
    runner = threading.Thread(target=loop.run_forever, daemon=True)
    runner.start()

    class Stuck(Tunnel):
        async def call(self, op: str, payload: dict | None = None, *, deadline_ms: int):
            await asyncio.sleep(60)
            raise AssertionError("never answers")

    try:
        monkeypatch.setattr(mem, "PREFETCH_WAIT_SECONDS", 0.05)
        tools, *_ = memory_for(tunnel=Stuck())
        tools.bind_loop(loop)
        text, count = tools.prefetch(CONVERSATION, "q")
        assert count == 0 and mem.INSTRUCTIONS["timed_out"] in text
    finally:
        loop.call_soon_threadsafe(loop.stop)
        runner.join(timeout=5)
        loop.close()


def test_prefetch_without_a_loop_still_answers_an_instruction() -> None:
    tools, *_ = memory_for()
    text, count = tools.prefetch(CONVERSATION, "q")
    assert count == 0 and text.startswith("Uffici memory could not be checked")


# ------------------------------------------------------------------ the provider


def test_the_provider_is_hermess_kind_and_does_nothing_but_prefetch() -> None:
    provider = build_provider()
    assert isinstance(provider, MemoryProvider)
    assert provider.name == "uffici" and provider.is_available() is True
    assert provider.get_tool_schemas() == []
    assert provider.system_prompt_block() == ""
    assert provider.on_pre_compress([{"role": "user", "content": "x"}]) == ""
    assert provider.sync_turn("u", "a") is None
    assert provider.initialize("session", hermes_home="/tmp", platform="uffici") is None
    # 🔴 Not opted into the fail-closed compression checkpoint.
    assert getattr(provider, "pre_compress_checkpoint_api_version", 1) == 1


def test_another_platforms_turn_recalls_nothing(monkeypatch) -> None:
    called: list[Any] = []
    monkeypatch.setattr(mem.rendezvous(), "prefetch", lambda *a: called.append(a) or ("x", 1))

    def turn() -> str:
        set_session_vars(platform="telegram", chat_id="chat")
        return build_provider().prefetch("hello there")

    assert run_in_thread(turn) == "" and called == []


def test_a_uffici_turn_with_no_adapter_published_is_told_the_companion_is_not_connected(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mem.rendezvous(), "prefetch", None)
    provider = build_provider()

    def turn() -> str:
        set_session_vars(platform="uffici", chat_id=CONVERSATION)
        return provider.prefetch("hello there")

    assert run_in_thread(turn) == NOT_CONNECTED
    assert provider.recall_status() is None


def test_the_provider_reaches_the_published_prefetch_and_reports_the_last_count(
    monkeypatch,
) -> None:
    seen: list[tuple[str, str]] = []

    def bridge(conversation: str, query: str) -> tuple[str, int]:
        seen.append((conversation, query))
        return "recalled", 3

    monkeypatch.setattr(mem.rendezvous(), "prefetch", bridge)
    provider = build_provider()

    def turn() -> str:
        set_session_vars(platform="uffici", chat_id=CONVERSATION)
        return provider.prefetch("what did we decide?")

    assert run_in_thread(turn) == "recalled"
    assert seen == [(CONVERSATION, "what did we decide?")]
    assert provider.recall_status() == RecallStatus(provider_label="Uffici", count=3)
    # 🔴 Only the LAST prefetch: one that recalled nothing reports nothing.
    monkeypatch.setattr(mem.rendezvous(), "prefetch", lambda *_a: ("", 0))
    run_in_thread(turn)
    assert provider.recall_status() is None


def run_in_thread(fn) -> Any:
    box: dict[str, Any] = {}
    worker = threading.Thread(target=lambda: box.setdefault("value", fn()))
    worker.start()
    worker.join(timeout=10)
    return box.get("value")


# ------------------------------------------------------------------ the rendezvous


def test_the_rendezvous_is_one_module_whatever_name_the_package_was_imported_under() -> None:
    first = mem.rendezvous()
    assert sys.modules[mem.RENDEZVOUS_MODULE] is first
    assert not mem.RENDEZVOUS_MODULE.startswith(mem.__name__.split(".")[0] + ".")
    assert mem.rendezvous() is first


def test_withdraw_removes_only_the_prefetch_still_published() -> None:
    def a(*_args: Any) -> tuple[str, int]:
        return "a", 1

    def b(*_args: Any) -> tuple[str, int]:
        return "b", 1

    mem.publish(a)
    mem.withdraw(b)
    assert mem.rendezvous().prefetch is a
    mem.withdraw(a)
    assert mem.rendezvous().prefetch is None


# ------------------------------------------------------------------ registration


def test_the_memory_collector_registers_the_provider_and_never_the_platform_again() -> None:
    collector = ProviderCollector()
    assert is_memory_collector(collector)
    register(collector)
    assert isinstance(collector.provider, MemoryProvider)
    assert collector.forwarded == []


def test_a_plugin_context_is_not_mistaken_for_the_memory_collector() -> None:
    class Context:
        def register_platform(self, **_: Any) -> None:
            pass

        def register_memory_provider(self, _provider: Any) -> None:
            pass

    assert not is_memory_collector(Context())
    assert not is_memory_collector(object())


# ------------------------------------------------------------------ the tools


def test_the_four_tools_register_in_the_uffici_toolset_when_the_companion_has_memory() -> None:
    tools, registry, *_ = memory_for()
    tools.sync()
    assert set(registry.tools) == set(TOOL_SCHEMAS) == set(TOOL_OPS)
    assert {entry["toolset"] for entry in registry.tools.values()} == {TOOLSET}
    tools.sync()
    assert len(registry.calls) == 4, "a second hello re-registers nothing"


def test_the_tools_are_withdrawn_when_a_companion_without_memory_says_hello() -> None:
    tools, registry, *_ = memory_for()
    tools.sync()
    tools.flags["available"] = False  # type: ignore[attr-defined]
    tools.sync()
    assert registry.tools == {} and tools.offered == frozenset()


def test_a_tool_name_hermes_already_holds_is_skipped_and_the_rest_offered() -> None:
    tools, registry, *_ = memory_for(registry=Registry(taken={"uffici_recall"}))
    tools.sync()
    assert tools.offered == frozenset(TOOL_SCHEMAS) - {"uffici_recall"}


def test_outside_hermes_nothing_is_registered() -> None:
    tools = MemoryTools(
        None,
        call=Tunnel().call,
        request_for={}.get,
        available=lambda: True,
        connected=lambda: True,
    )
    tools.sync()
    assert tools.offered == frozenset()


def test_every_description_names_this_organizations_memory_and_the_urns_it_takes() -> None:
    for name, schema in TOOL_SCHEMAS.items():
        assert "this organization's Uffici memory" in schema["description"], name
    assert "never an appraisal" in TOOL_SCHEMAS["uffici_remember"]["description"]
    assert "a fact URN" in TOOL_SCHEMAS["uffici_forget"]["description"]
    assert "statement URN" in TOOL_SCHEMAS["uffici_unfold"]["description"]


@pytest.mark.parametrize(
    ("tool", "args", "expected"),
    [
        (
            "uffici_recall",
            {"query": "Lisbon", "limit": 3},
            {"query": "Lisbon", "limit": 3, "deadlineMs": mem.TOOL_RECALL_DEADLINE_MS},
        ),
        (
            "uffici_remember",
            {"text": "  Lisbon is invoiced on the 10th. "},
            {"text": "Lisbon is invoiced on the 10th."},
        ),
        (
            "uffici_forget",
            {"urn": FACT, "replacementUrn": FACT},
            {"urn": FACT, "replacementUrn": FACT},
        ),
        ("uffici_unfold", {"urn": FACT, "depth": 2, "ignored": True}, {"urn": FACT, "depth": 2}),
    ],
)
def test_each_tool_is_forwarded_as_its_op_on_the_turns_request(
    loop_thread, tool: str, args: dict, expected: dict
) -> None:
    tools, registry, tunnel, _ = memory_for()
    tools.bind_loop(loop_thread)
    tools.sync()
    answered = json.loads(registry.tools[tool]["handler"](args))
    op, payload, _deadline = tunnel.calls[0]
    assert op == TOOL_OPS[tool] and payload == {"requestId": REQUEST, **expected}
    schema = request_schema(op)
    assert set(payload) <= set(schema["properties"]) and set(schema["required"]) <= set(payload)
    assert "urn" in answered or "hits" in answered


def test_a_tool_outside_a_uffici_request_does_nothing_and_says_why(loop_thread) -> None:
    tools, registry, tunnel, _ = memory_for(ledger={})
    tools.bind_loop(loop_thread)
    tools.sync()
    answered = json.loads(registry.tools["uffici_remember"]["handler"]({"text": "x"}))
    assert answered == {"error": OUTSIDE_A_REQUEST} and tunnel.calls == []


def test_a_tool_with_its_argument_missing_is_refused_before_the_tunnel(loop_thread) -> None:
    tools, registry, tunnel, _ = memory_for()
    tools.bind_loop(loop_thread)
    tools.sync()
    for tool in TOOL_SCHEMAS:
        answered = json.loads(registry.tools[tool]["handler"]({}))
        assert answered["errorClass"] == "malformed_payload", tool
    assert tunnel.calls == []


def test_a_failed_tool_call_carries_the_platforms_class_and_sentence(loop_thread) -> None:
    refused = "That fact is not one you may forget; tell the person."
    tunnel = Tunnel({"memory.forget": failed("upstream_failed", refused)})
    tools, registry, *_ = memory_for(tunnel=tunnel)
    tools.bind_loop(loop_thread)
    tools.sync()
    answered = json.loads(registry.tools["uffici_forget"]["handler"]({"urn": FACT}))
    assert answered == {"error": refused, "errorClass": "upstream_failed"}


def test_a_tool_called_after_the_companion_lost_memory_is_unknown_op(loop_thread) -> None:
    tools, registry, tunnel, _ = memory_for()
    tools.bind_loop(loop_thread)
    tools.sync()
    tools.flags["available"] = False  # type: ignore[attr-defined]
    answered = json.loads(registry.tools["uffici_recall"]["handler"]({"query": "q"}))
    assert answered["errorClass"] == "unknown_op" and tunnel.calls == []


def test_no_tool_logs_a_fact_or_a_query(loop_thread, caplog) -> None:
    tools, registry, *_ = memory_for()
    tools.bind_loop(loop_thread)
    tools.sync()
    caplog.set_level("DEBUG")
    registry.tools["uffici_remember"]["handler"]({"text": "SECRET-FACT-TEXT"})
    tools.prefetch(CONVERSATION, "SECRET-QUERY-TEXT")
    logged = "\n".join(r.getMessage() + repr(r.__dict__) for r in caplog.records)
    assert "SECRET-FACT-TEXT" not in logged and "SECRET-QUERY-TEXT" not in logged
    assert "Lisbon" not in logged, "a recalled fact reached a log line"


# ------------------------------------------------------------------ the adapter


def test_the_companions_hello_decides_whether_the_memory_tools_are_offered() -> None:
    from conftest import PlatformConfig
    from uffici_for_hermes.adapter import UfficiAdapter

    registry = Registry()
    ctx = type("Ctx", (), {"register_tool": registry.register_tool})()
    adapter = UfficiAdapter(PlatformConfig(enabled=True), get_env=lambda _: None, plugin_ctx=ctx)

    async def scenario() -> None:
        await adapter._on_companion_hello(  # noqa: SLF001
            {"v": 1, "type": "hello", "features": ["tools", "memory"]}
        )
        assert set(registry.tools) == set(TOOL_SCHEMAS)
        await adapter._on_companion_hello({"v": 1, "type": "hello", "features": ["tools"]})  # noqa: SLF001
        assert registry.tools == {}
        runner = adapter._ready_runner  # noqa: SLF001
        if runner is not None:
            runner.cancel()

    asyncio.run(scenario())


def test_connect_publishes_the_adapters_prefetch_and_disconnect_withdraws_it() -> None:
    from test_connect import adapter_with

    async def scenario() -> None:
        adapter, _opened = adapter_with(
            UFFICI_WIRE_TOKEN="t" * 40, UFFICI_CONTEXT_WINDOW_TOKENS="100000"
        )
        assert await adapter.connect() is True
        assert mem.rendezvous().prefetch == adapter._memory.prefetch  # noqa: SLF001
        await adapter.disconnect()
        assert mem.rendezvous().prefetch is None

    asyncio.run(scenario())


def test_a_connected_adapter_answers_a_prefetch_from_hermess_thread_end_to_end() -> None:
    """The provider copy → the rendezvous → the adapter → the tunnel → back, with only the
    companion doubled: the path a real turn takes."""
    from test_connect import adapter_with

    async def scenario() -> tuple[str, list[dict]]:
        adapter, opened = adapter_with(
            UFFICI_WIRE_TOKEN="t" * 40, UFFICI_CONTEXT_WINDOW_TOKENS="100000"
        )
        assert await adapter.connect() is True
        sent: list[dict] = []

        async def companion(frame: dict) -> bool:
            sent.append(frame)
            asyncio.get_running_loop().call_soon(
                adapter._calls.on_result,  # noqa: SLF001
                {"v": 1, "type": "result", "callId": frame["callId"], "payload": LEVELS},
            )
            return True

        adapter._calls._send = companion  # noqa: SLF001
        adapter._companion_features = frozenset({"memory"})  # noqa: SLF001
        adapter._hello_event().set()  # noqa: SLF001
        adapter._open_request[CONVERSATION] = REQUEST  # noqa: SLF001

        def turn() -> str:
            set_session_vars(platform="uffici", chat_id=CONVERSATION)
            return build_provider().prefetch("When is Lisbon invoiced?")

        text = await asyncio.to_thread(turn)
        await adapter.disconnect()
        return text, sent

    text, sent = asyncio.run(scenario())
    assert text.startswith(RECALL_HEADING) and FACT in text
    assert sent[0]["op"] == "memory.recall" and sent[0]["payload"]["requestId"] == REQUEST


# ------------------------------------------------------------------ drift


def setup_config() -> dict:
    """What the manifest's ``config`` lines produce for the memory section."""
    manifest = json.loads((mem_root() / "uffici-plugin.json").read_text())
    config: dict = {}
    for entry in manifest["config"]:
        *parents, leaf = entry["key"].split(".")
        node = config
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = {"true": True, "false": False}.get(entry["value"], entry["value"])
    return config


def mem_root():
    from pathlib import Path

    return Path(__file__).resolve().parent.parent


def test_the_setups_own_keys_carry_no_drift() -> None:
    assert mem.memory_drift(setup_config()) == []


def test_another_provider_or_none_is_drift() -> None:
    config = setup_config()
    config["memory"]["provider"] = "honcho"
    assert "memory.provider is honcho" in mem.memory_drift(config)[0]
    assert "memory.provider is unset" in mem.memory_drift({})[0]


@pytest.mark.parametrize("key", ["memory_enabled", "user_profile_enabled"])
def test_hermess_own_memory_switched_back_on_or_absent_is_drift(key: str) -> None:
    config = setup_config()
    config["memory"][key] = True
    assert f"memory.{key}" in mem.memory_drift(config)[-1]
    del config["memory"][key]
    assert f"memory.{key}" in mem.memory_drift(config)[-1], "absent is ON in Hermes"
    config["memory"][key] = "off"
    assert mem.memory_drift(config) == []


def test_drift_is_logged_and_an_unreadable_config_is_not_an_error(caplog) -> None:
    caplog.set_level("INFO")
    assert mem.report_memory_drift(lambda: {"memory": {}})
    assert any(
        r.levelname == "WARNING" and "memory.provider" in r.getMessage() for r in caplog.records
    )

    def broken() -> dict:
        raise OSError("no config")

    assert mem.report_memory_drift(broken) == []
    assert mem.report_memory_drift() == [], "outside Hermes there is no config to read"


# ------------------------------------------------------------------ never a raise


def test_a_bridge_that_raises_is_an_instruction_never_nothing(monkeypatch) -> None:
    """🔴 Hermes swallows a provider's exception and the turn reads nothing remembered."""

    def broken(*_args: Any) -> tuple[str, int]:
        raise RuntimeError("the loop went away")

    monkeypatch.setattr(mem.rendezvous(), "prefetch", broken)
    provider = build_provider()

    def turn() -> str:
        set_session_vars(platform="uffici", chat_id=CONVERSATION)
        return provider.prefetch("what was decided?")

    assert run_in_thread(turn) == mem.BRIDGE_FAILED


def test_a_call_the_gateway_cancels_is_answered_not_raised(loop_thread) -> None:
    from uffici_for_hermes.platform_tools import run_on_loop

    async def cancelled() -> str:
        raise asyncio.CancelledError

    answered = json.loads(run_on_loop(loop_thread, cancelled, wait_seconds=5, tool="x"))
    assert answered["errorClass"] == "upstream_failed"


def test_a_loop_that_closed_before_the_hand_off_is_answered_not_raised() -> None:
    from uffici_for_hermes.platform_tools import run_on_loop

    # A real loop that closed after the plugin's check said it was open.
    loop = asyncio.new_event_loop()
    loop.close()
    loop.is_closed = lambda: False  # type: ignore[method-assign]

    async def never() -> str:  # pragma: no cover - never scheduled
        return ""

    coroutine = never()
    try:
        answered = json.loads(run_on_loop(loop, lambda: coroutine, wait_seconds=1, tool="x"))
    finally:
        coroutine.close()
        # Hand the loop its real answer back, or its finalizer closes it a second time.
        del loop.is_closed
    assert answered["errorClass"] == "unknown_op"


def test_a_boolean_is_never_passed_as_a_count(loop_thread) -> None:
    tools, registry, tunnel, _ = memory_for()
    tools.bind_loop(loop_thread)
    tools.sync()
    registry.tools["uffici_recall"]["handler"]({"query": "q", "limit": True})
    registry.tools["uffici_unfold"]["handler"]({"urn": FACT, "depth": True})
    assert "limit" not in tunnel.calls[0][1] and "depth" not in tunnel.calls[1][1]
