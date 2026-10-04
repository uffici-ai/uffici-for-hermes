"""The agent's Uffici tools as Hermes tools: the catalog, the calls, what a post cites."""

from __future__ import annotations

import asyncio
import contextvars
import json
import threading
from typing import Any

import pytest

from conftest import PlatformConfig, set_session_vars
from test_adapter import FakeStore, FakeWire
from test_adapter import golden as adapter_golden
from uffici_for_hermes import platform_tools as pt
from uffici_for_hermes.adapter import UfficiAdapter
from uffici_for_hermes.calls import CallResult
from uffici_for_hermes.platform_tools import (
    ASK_SENTENCE,
    NO_LONGER_AVAILABLE,
    OUTSIDE_A_REQUEST,
    TOOLSET,
    PlatformTools,
    catalog_entries,
    hermes_schema,
    on_post_tool_call,
)
from uffici_for_hermes.wire import WIRE_V1


def op_golden(name: str) -> dict:
    return json.loads((WIRE_V1 / "golden" / "ops" / "valid" / f"{name}.json").read_text())


BOARDS = op_golden("tools.list.result-boards")
RAN = op_golden("tools.call.result-ran")
NEEDS_APPROVAL = op_golden("tools.call.result-needs-approval")


class Handle:
    def __init__(self, registry: Registry, name: str) -> None:
        self._registry, self.name, self.disposed = registry, name, False

    def dispose(self) -> None:
        self.disposed = True
        self._registry.tools.pop(self.name, None)


class Registry:
    """``PluginContext.register_tool`` as much as this plugin uses: keyword arguments,
    a handle whose ``dispose`` removes the tool, and None for a name another toolset
    already holds (``hermes_cli/plugins.py``)."""

    def __init__(self, taken: set[str] | None = None) -> None:
        self.tools: dict[str, dict[str, Any]] = {}
        self.calls: list[dict[str, Any]] = []
        self._taken = taken or set()

    def register_tool(self, **kwargs: Any) -> Handle | None:
        self.calls.append(kwargs)
        if kwargs["name"] in self._taken:
            return None
        self.tools[kwargs["name"]] = kwargs
        return Handle(self, kwargs["name"])


class Tunnel:
    """The call client's ``call``: records each call, answers from a table."""

    def __init__(self, answers: dict[str, CallResult] | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.answers = answers or {}

    async def call(self, op: str, payload: dict | None = None, *, deadline_ms: int) -> CallResult:
        self.calls.append((op, payload or {}))
        if op in self.answers:
            return self.answers[op]
        if op == "tools.list":
            return CallResult(call_id="c", payload=BOARDS)
        return CallResult(call_id="c", payload=RAN)


def tools_for(
    *,
    registry: Registry | None = None,
    tunnel: Tunnel | None = None,
    ledger: dict[str, str] | None = None,
    enabled: bool = True,
    clock=None,
    session=None,
) -> tuple[PlatformTools, Registry, Tunnel, dict[str, str]]:
    registry = registry or Registry()
    tunnel = tunnel or Tunnel()
    ledger = {} if ledger is None else ledger
    kwargs: dict[str, Any] = {}
    if clock is not None:
        kwargs["clock"] = clock
    if session is not None:
        kwargs["session_reader"] = session
    tools = PlatformTools(
        registry.register_tool,
        call=tunnel.call,
        request_for=ledger.get,
        enabled=lambda: enabled,
        **kwargs,
    )
    return tools, registry, tunnel, ledger


# ------------------------------------------------------------------ the catalog


def test_the_catalog_is_read_in_the_wires_shape() -> None:
    entries = catalog_entries(BOARDS)
    assert entries is not None and set(entries) == {"list_boards", "delete_board"}
    assert entries["delete_board"]["permission"] == "ask"
    assert catalog_entries(op_golden("tools.list.result-empty")) == {}


@pytest.mark.parametrize("payload", [None, {}, {"tools": "x"}, []])
def test_a_result_that_is_not_a_catalog_is_none(payload: Any) -> None:
    assert catalog_entries(payload) is None


def test_an_entry_outside_the_wire_is_skipped_never_repaired() -> None:
    good = BOARDS["tools"][0]
    payload = {
        "tools": [
            good,
            {**good, "name": "has space"},
            {**good, "name": "deny_one", "permission": "deny"},
            {**good, "name": "no_params", "parameters": []},
            {**good, "name": "long", "description": "x" * 16_001},
            {**good},  # a duplicate name
            "not an object",
        ]
    }
    assert set(catalog_entries(payload) or {}) == {"list_boards"}


def test_the_schema_keeps_the_platforms_name_description_and_parameters() -> None:
    entries = catalog_entries(BOARDS) or {}
    allow = hermes_schema(entries["list_boards"])
    assert allow == {
        "name": "list_boards",
        "description": BOARDS["tools"][0]["description"],
        "parameters": BOARDS["tools"][0]["parameters"],
    }


def test_an_ask_tool_says_a_person_approves_each_call() -> None:
    entries = catalog_entries(BOARDS) or {}
    ask = hermes_schema(entries["delete_board"])
    assert ask["description"].startswith(BOARDS["tools"][1]["description"])
    assert ask["description"].endswith(ASK_SENTENCE)
    assert ASK_SENTENCE not in hermes_schema(entries["list_boards"])["description"]
    empty = hermes_schema({**entries["delete_board"], "description": ""})
    assert empty["description"] == ASK_SENTENCE


def test_every_tool_is_registered_in_the_uffici_toolset() -> None:
    tools, registry, _, _ = tools_for()
    tools.apply(catalog_entries(BOARDS) or {})
    assert set(registry.tools) == {"list_boards", "delete_board"}
    for registered in registry.tools.values():
        assert registered["toolset"] == TOOLSET == "uffici"
        assert registered["schema"]["name"] == registered["name"]
        assert callable(registered["handler"])
    assert tools.offered == {"list_boards", "delete_board"}


def test_a_tool_that_left_the_catalog_is_unregistered() -> None:
    tools, registry, _, _ = tools_for()
    tools.apply(catalog_entries(BOARDS) or {})
    tools.apply({"list_boards": (catalog_entries(BOARDS) or {})["list_boards"]})
    assert set(registry.tools) == {"list_boards"}
    assert tools.offered == {"list_boards"}


def test_an_unchanged_tool_is_not_registered_again_and_a_changed_one_is_replaced() -> None:
    tools, registry, _, _ = tools_for()
    entries = catalog_entries(BOARDS) or {}
    tools.apply(entries)
    tools.apply(entries)
    assert len(registry.calls) == 2, "an unchanged catalog re-registered its tools"
    changed = {**entries, "list_boards": {**entries["list_boards"], "description": "new"}}
    tools.apply(changed)
    assert len(registry.calls) == 3
    assert registry.tools["list_boards"]["schema"]["description"] == "new"


def test_a_name_hermes_refuses_is_not_offered() -> None:
    tools, registry, _, _ = tools_for(registry=Registry(taken={"list_boards"}))
    tools.apply(catalog_entries(BOARDS) or {})
    assert tools.offered == {"delete_board"}

    def boom(**_: Any) -> None:
        raise PermissionError("no")

    raising = PlatformTools(boom, call=Tunnel().call, request_for={}.get, enabled=lambda: True)
    raising.apply(catalog_entries(BOARDS) or {})
    assert raising.offered == frozenset()


def test_without_a_plugin_context_nothing_is_registered() -> None:
    tools = PlatformTools(None, call=Tunnel().call, request_for={}.get, enabled=lambda: True)
    assert tools.enabled() is False
    tools.apply(catalog_entries(BOARDS) or {})
    assert tools.offered == frozenset()


def test_a_dispose_that_raises_still_takes_the_tool_away() -> None:
    tools, registry, _, _ = tools_for()
    tools.apply(catalog_entries(BOARDS) or {})

    class Stuck:
        def dispose(self) -> None:
            raise RuntimeError("stuck")

    tools._handles["list_boards"] = Stuck()  # noqa: SLF001 — a Hermes that will not remove it
    tools.apply({})
    assert tools.offered == frozenset()


# ------------------------------------------------------------------ the refresh


def test_the_catalog_is_asked_for_the_conversation_and_registered() -> None:
    async def scenario() -> tuple:
        tools, registry, tunnel, _ = tools_for()
        tools.kick("conv-1")
        assert tools.refreshing()
        await tools.wait()
        return registry, tunnel

    registry, tunnel = asyncio.run(scenario())
    assert tunnel.calls == [("tools.list", {"conversationId": "conv-1"})]
    assert set(registry.tools) == {"list_boards", "delete_board"}


def test_the_catalog_is_asked_for_at_most_once_a_minute() -> None:
    now = [1000.0]

    async def scenario() -> Tunnel:
        tools, _, tunnel, _ = tools_for(clock=lambda: now[0])
        tools.kick("conv-1")
        await tools.wait()
        now[0] += 59
        tools.kick("conv-1")
        await tools.wait()
        assert len(tunnel.calls) == 1, "asked again inside the minute"
        now[0] += 1
        tools.kick("conv-2")
        await tools.wait()
        return tunnel

    tunnel = asyncio.run(scenario())
    assert [payload for _, payload in tunnel.calls] == [
        {"conversationId": "conv-1"},
        {"conversationId": "conv-2"},
    ]


def test_no_catalog_is_asked_for_without_the_companions_feature_or_a_conversation() -> None:
    async def scenario() -> Tunnel:
        tools, _, tunnel, _ = tools_for(enabled=False)
        tools.kick("conv-1")
        await tools.wait()
        on, _, on_tunnel, _ = tools_for()
        on.kick(None)
        await on.wait()
        assert not on_tunnel.calls
        return tunnel

    assert asyncio.run(scenario()).calls == []


@pytest.mark.parametrize(
    "answer",
    [
        CallResult(call_id="c", error_class="timed_out", instruction="x"),
        CallResult(call_id="c", payload={"tools": "nope"}),
    ],
)
def test_a_failed_read_keeps_the_tools_already_offered(answer: CallResult) -> None:
    now = [0.0]

    async def scenario() -> PlatformTools:
        tunnel = Tunnel()
        tools, _, _, _ = tools_for(tunnel=tunnel, clock=lambda: now[0])
        tools.kick("conv-1")
        await tools.wait()
        tunnel.answers["tools.list"] = answer
        now[0] += 60
        tools.kick("conv-1")
        await tools.wait()
        return tools

    assert asyncio.run(scenario()).offered == {"list_boards", "delete_board"}


def test_a_tunnel_that_raises_fails_no_turn() -> None:
    async def boom(*_: Any, **__: Any) -> CallResult:
        raise RuntimeError("gone")

    async def scenario() -> PlatformTools:
        tools = PlatformTools(
            Registry().register_tool, call=boom, request_for={}.get, enabled=lambda: True
        )
        tools.kick("conv-1")
        await tools.wait()
        await tools.wait()  # nothing in flight: returns at once
        return tools

    assert asyncio.run(scenario()).offered == frozenset()


# ------------------------------------------------------------------ the calls


def offered(**kwargs: Any) -> tuple[PlatformTools, Tunnel, dict[str, str]]:
    tools, _, tunnel, ledger = tools_for(**kwargs)
    tools.apply(catalog_entries(BOARDS) or {})
    return tools, tunnel, ledger


def test_a_call_goes_out_on_the_conversations_open_request() -> None:
    tools, tunnel, ledger = offered(ledger={"conv-1": "req-1"})
    out = asyncio.run(tools.invoke("conv-1", "list_boards", {"x": 1}))
    assert tunnel.calls == [
        ("tools.call", {"requestId": "req-1", "name": "list_boards", "arguments": {"x": 1}})
    ]
    assert json.loads(out) == {"callUrn": RAN["callUrn"], "result": RAN["result"]}


def test_arguments_that_are_not_an_object_go_as_an_empty_one() -> None:
    tools, tunnel, _ = offered(ledger={"conv-1": "req-1"})
    asyncio.run(tools.invoke("conv-1", "list_boards", ["not", "an", "object"]))
    assert tunnel.calls[0][1]["arguments"] == {}


@pytest.mark.parametrize("conversation", [None, "conv-without-a-request"])
def test_outside_a_request_the_model_is_told_and_nothing_is_called(conversation) -> None:
    """🔴 Refused WITHOUT the tunnel: a call with no request would have to guess one."""
    tools, tunnel, _ = offered(ledger={"conv-1": "req-1"})
    out = json.loads(asyncio.run(tools.invoke(conversation, "list_boards", {})))
    assert out == {"error": OUTSIDE_A_REQUEST}
    assert "cannot act outside a request" in out["error"]
    assert tunnel.calls == []


def test_a_tool_that_left_the_catalog_answers_that_it_is_no_longer_available() -> None:
    tools, tunnel, _ = offered(ledger={"conv-1": "req-1"})
    tools.apply({})
    out = json.loads(asyncio.run(tools.invoke("conv-1", "list_boards", {})))
    assert out == {"error": NO_LONGER_AVAILABLE}
    assert tunnel.calls == []


def test_a_companion_without_the_feature_is_never_called() -> None:
    tools, tunnel, _ = offered(ledger={"conv-1": "req-1"}, enabled=False)
    out = json.loads(asyncio.run(tools.invoke("conv-1", "list_boards", {})))
    assert out["errorClass"] == "unknown_op"
    assert tunnel.calls == []


def test_a_failed_call_relays_its_instruction() -> None:
    failed = CallResult(call_id="c", error_class="not_permitted", instruction="Tell them.")
    tools, _, _ = offered(ledger={"conv-1": "r"}, tunnel=Tunnel({"tools.call": failed}))
    out = json.loads(asyncio.run(tools.invoke("conv-1", "list_boards", {})))
    assert out == {"error": "Tell them.", "errorClass": "not_permitted"}


@pytest.mark.parametrize(
    "payload",
    [{"callUrn": None}, {"callUrn": "", "result": {}}, {"callUrn": 7, "result": {}}],
)
def test_a_result_outside_the_wire_is_malformed(payload: dict) -> None:
    answer = CallResult(call_id="c", payload=payload)
    tools, _, _ = offered(ledger={"conv-1": "r"}, tunnel=Tunnel({"tools.call": answer}))
    out = json.loads(asyncio.run(tools.invoke("conv-1", "list_boards", {})))
    assert out["errorClass"] == "malformed_payload"


def test_a_call_that_needs_approval_is_a_result_and_cites_nothing() -> None:
    answer = CallResult(call_id="c", payload=NEEDS_APPROVAL)
    tools, _, _ = offered(ledger={"conv-1": "r"}, tunnel=Tunnel({"tools.call": answer}))
    out = json.loads(asyncio.run(tools.invoke("conv-1", "delete_board", {"board": "b"})))
    assert out == {"callUrn": None, "result": NEEDS_APPROVAL["result"]}
    assert tools.take_extras("r") == {}


def test_the_calls_that_ran_are_cited_once_each_and_bounded() -> None:
    tools, tunnel, _ = offered(ledger={"conv-1": "r"})

    async def scenario() -> None:
        for n in range(1, 25):
            tunnel.answers["tools.call"] = CallResult(
                call_id="c", payload={"callUrn": f"urn:uffici:tool_call:{n}", "result": {}}
            )
            await tools.invoke("conv-1", "list_boards", {})
        await tools.invoke("conv-1", "list_boards", {})  # the last urn again

    asyncio.run(scenario())
    extras = tools.take_extras("r")
    assert extras["citations"] == [f"urn:uffici:tool_call:{n}" for n in range(1, 21)]
    assert tools.take_extras("r") == {}, "a citation was handed over twice"


# ------------------------------------------------------------------ the session


class Loop:
    """An event loop on its own thread — the gateway's, where the tunnel runs."""

    def __enter__(self) -> asyncio.AbstractEventLoop:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        return self.loop

    def __exit__(self, *_: Any) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        pending = asyncio.all_tasks(self.loop)
        for task in pending:
            task.cancel()
        if pending:
            self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self.loop.close()


def in_turn(platform: str, chat_id: str, fn):
    """What Hermes does: the turn's tools run in a worker thread carrying its session."""

    def run() -> Any:
        set_session_vars(platform=platform, chat_id=chat_id)
        return fn()

    return contextvars.Context().run(run)


def test_two_concurrent_sessions_each_act_on_their_own_request() -> None:
    """🔴 THE binding. One Hermes serves many conversations at once, and a call must act
    on the request of the conversation its turn belongs to — read in the worker thread
    the tool runs in, from Hermes's session ContextVars, never on the event loop."""
    gate = threading.Barrier(2)
    tunnel = Tunnel()
    tools, _, _ = offered(ledger={"conv-a": "req-a", "conv-b": "req-b"}, tunnel=tunnel)
    results: dict[str, str] = {}
    with Loop() as loop:
        tools.bind_loop(loop)

        def worker(conversation: str) -> None:
            def call() -> str:
                gate.wait(timeout=5)
                return tools.run_from_thread("list_boards", {"who": conversation})

            results[conversation] = in_turn("uffici", conversation, call)

        threads = [threading.Thread(target=worker, args=(c,)) for c in ("conv-a", "conv-b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
    sent = {payload["arguments"]["who"]: payload["requestId"] for _, payload in tunnel.calls}
    assert sent == {"conv-a": "req-a", "conv-b": "req-b"}
    assert all("callUrn" in json.loads(out) for out in results.values())


def test_a_turn_on_another_platform_cannot_act() -> None:
    tools, tunnel, _ = offered(ledger={"conv-a": "req-a"})
    with Loop() as loop:
        tools.bind_loop(loop)
        out = in_turn("telegram", "conv-a", lambda: tools.run_from_thread("list_boards", {}))
    assert json.loads(out) == {"error": OUTSIDE_A_REQUEST}
    assert tunnel.calls == []


def test_without_a_loop_the_handler_answers_rather_than_hangs() -> None:
    tools, tunnel, _ = offered(ledger={"conv-a": "req-a"})
    out = in_turn("uffici", "conv-a", lambda: tools.run_from_thread("list_boards", {}))
    assert json.loads(out)["errorClass"] == "unknown_op"
    assert tunnel.calls == []


def test_a_handler_run_on_the_loop_itself_is_refused_not_deadlocked() -> None:
    async def scenario() -> str:
        tools, _, _ = offered(ledger={"conv-a": "req-a"})
        tools.bind_loop(asyncio.get_running_loop())
        return tools.run_from_thread("list_boards", {})

    assert json.loads(asyncio.run(scenario())) == {"error": pt.CANNOT_WAIT_HERE}


def test_a_call_that_outlives_its_wait_is_a_timed_out_answer(monkeypatch) -> None:
    monkeypatch.setattr(pt, "CALL_DEADLINE_MS", 1)
    monkeypatch.setattr(pt, "THREAD_GRACE_SECONDS", 0.05)

    async def never(*_: Any, **__: Any) -> CallResult:
        await asyncio.sleep(5)
        raise AssertionError("unreachable")

    tools = PlatformTools(
        Registry().register_tool,
        call=never,
        request_for={"conv-a": "r"}.get,
        enabled=lambda: True,
    )
    tools.apply(catalog_entries(BOARDS) or {})
    with Loop() as loop:
        tools.bind_loop(loop)
        out = in_turn("uffici", "conv-a", lambda: tools.run_from_thread("list_boards", {}))
    assert json.loads(out)["errorClass"] == "timed_out"


def test_the_registered_handler_is_the_one_hermes_dispatches() -> None:
    tools, registry, tunnel, _ = tools_for(ledger={"conv-a": "req-a"})
    tools.apply(catalog_entries(BOARDS) or {})
    handler = registry.tools["list_boards"]["handler"]
    with Loop() as loop:
        tools.bind_loop(loop)
        in_turn("uffici", "conv-a", lambda: handler({}))
    assert tunnel.calls[0][1]["requestId"] == "req-a"


# ------------------------------------------------------------------ the post


def test_hermess_own_tool_calls_are_reported_by_name_only() -> None:
    tools, _, _ = offered(ledger={"conv-a": "req-a"})
    with Loop() as loop:
        tools.bind_loop(loop)
        for name, status in (
            ("terminal", "ok"),
            ("web_search", "error"),
            ("list_boards", "ok"),  # ours: cited, not reported
            ("tool_search", "ok"),  # Hermes's bridge, not something the agent did
            ("", "ok"),
        ):
            in_turn(
                "uffici",
                "conv-a",
                lambda n=name, s=status: on_post_tool_call(
                    tool_name=n,
                    args={"secret": "never read"},
                    result="never read",
                    status=s,
                ),
            )
        in_turn("telegram", "conv-a", lambda: on_post_tool_call(tool_name="terminal"))
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(timeout=5)
    assert tools.take_extras("req-a")["activity"] == [
        {"tool": "terminal", "summary": pt.ACTIVITY_SUMMARY_OK},
        {"tool": "web_search", "summary": pt.ACTIVITY_SUMMARY_FAILED},
    ]


def test_activity_is_bounded_and_outside_a_request_dropped() -> None:
    tools, _, ledger = offered(ledger={"conv-a": "req-a"})
    for n in range(25):
        tools._record_activity("conv-a", "t" * 70 + str(n), "ok")  # noqa: SLF001
    tools._record_activity("conv-none", "terminal", "ok")  # noqa: SLF001
    done = tools.take_extras("req-a")["activity"]
    assert len(done) == pt.MAX_ACTIVITY and all(len(d["tool"]) == 64 for d in done)
    assert tools.take_extras("conv-none") == {}


def test_without_the_feature_a_post_carries_nothing_new() -> None:
    tools, _, _ = offered(ledger={"conv-a": "r"}, enabled=False)
    tools._citations["r"] = ["urn:uffici:tool_call:1"]  # noqa: SLF001
    assert tools.take_extras("r") == {}


def test_a_post_that_did_not_go_gives_its_extras_back() -> None:
    tools, _, _ = offered(ledger={"conv-a": "r"})
    tools._citations["r"] = ["urn:uffici:tool_call:2"]  # noqa: SLF001
    tools._record_activity("conv-a", "terminal", "ok")  # noqa: SLF001
    extras = tools.take_extras("r")
    tools._citations["r"] = ["urn:uffici:tool_call:3"]  # noqa: SLF001
    tools.give_back("r", extras)
    again = tools.take_extras("r")
    assert again["citations"] == ["urn:uffici:tool_call:2", "urn:uffici:tool_call:3"]
    assert again["activity"] == extras["activity"]
    tools.give_back("r", {})
    tools.forget("r")
    tools.forget_requests()
    assert tools.take_extras("r") == {}


def test_a_hook_on_a_closed_loop_is_ignored() -> None:
    tools, _, _ = offered(ledger={"conv-a": "r"})
    loop = asyncio.new_event_loop()
    loop.close()
    tools.bind_loop(loop)
    in_turn("uffici", "conv-a", lambda: tools.on_tool_called("terminal", "ok"))
    assert tools.take_extras("r") == {}


def test_a_hook_that_raises_never_fails_the_tool_call(monkeypatch) -> None:
    tools, _, _ = offered()

    def boom(*_: Any) -> None:
        raise RuntimeError("observer broke")

    monkeypatch.setattr(tools, "on_tool_called", boom)
    on_post_tool_call(tool_name="terminal", status="ok")


def test_the_session_reader_answers_only_for_a_uffici_turn() -> None:
    assert in_turn("uffici", "conv-z", pt.session_conversation) == "conv-z"
    assert in_turn("telegram", "conv-z", pt.session_conversation) is None
    assert in_turn("uffici", "", pt.session_conversation) is None


# ------------------------------------------------------------------ the adapter


def test_the_adapter_registers_through_the_plugins_own_context() -> None:
    registry = Registry()
    ctx = type("Ctx", (), {"register_tool": registry.register_tool})()
    adapter = UfficiAdapter(PlatformConfig(enabled=True), get_env=lambda _: None, plugin_ctx=ctx)
    adapter._companion_features = frozenset({"tools"})  # noqa: SLF001
    adapter._tools.apply(catalog_entries(BOARDS) or {})  # noqa: SLF001
    assert set(registry.tools) == {"list_boards", "delete_board"}


class CallingWire(FakeWire):
    """The companion, answering the plugin's calls: the catalog, then each tool call.

    ``hold`` keeps the catalog's answer back until ``release``, so a test can act while
    the plugin is still waiting on it.
    """

    def __init__(self, opener: dict | None = None, *, hold: bool = False) -> None:
        super().__init__(opener=opener)
        self.hold = hold
        self._held: list[dict] = []

    def _answer(self, frame: dict) -> None:
        payload = BOARDS if frame["op"] == "tools.list" else RAN
        self.adapter._calls.on_result(  # noqa: SLF001
            {"v": 1, "type": "result", "callId": frame["callId"], "payload": payload}
        )

    async def send(self, frame: dict) -> bool:
        if frame.get("type") == "call" and frame.get("op", "").startswith("tools."):
            self.sent.append(frame)
            if self.hold and frame["op"] == "tools.list":
                self._held.append(frame)
            else:
                asyncio.get_running_loop().call_soon(self._answer, frame)
            return True
        return await super().send(frame)

    def release(self) -> None:
        for frame in self._held:
            self._answer(frame)
        self._held.clear()

    def calls(self, op: str) -> list[dict]:
        return [f for f in self.sent if f.get("type") == "call" and f.get("op") == op]


TOOLS_HELLO = {"v": 1, "type": "hello", "companionVersion": "0.7.0", "features": ["tools"]}


async def attached(*, features: bool = True, hold: bool = False):
    registry = Registry()
    ctx = type("Ctx", (), {"register_tool": registry.register_tool})()
    adapter = UfficiAdapter(PlatformConfig(enabled=True), get_env=lambda _: None, plugin_ctx=ctx)
    wire = CallingWire(opener=adapter_golden("opener"), hold=hold)
    wire.adapter = adapter
    adapter._client = wire  # noqa: SLF001
    adapter.set_session_store(FakeStore())
    adapter._tools.bind_loop(asyncio.get_running_loop())  # noqa: SLF001
    if features:
        adapter._companion_features = frozenset(TOOLS_HELLO["features"])  # noqa: SLF001
    return adapter, wire, registry


async def spin(times: int = 50) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


def test_a_brief_waits_for_its_tools_before_hermes_has_it() -> None:
    """🔴 Hermes freezes a session's tool list when it builds that session's agent, on its
    first turn — so the turn is handed over only once the catalog is registered."""

    async def scenario():
        adapter, wire, registry = await attached(hold=True)
        brief = adapter_golden("brief")
        await adapter._on_frame(brief)  # noqa: SLF001
        await spin()
        assert wire.calls("tools.list") == [
            {
                "v": 1,
                "type": "call",
                "callId": wire.calls("tools.list")[0]["callId"],
                "op": "tools.list",
                "deadlineMs": pt.LIST_DEADLINE_MS,
                "payload": {"conversationId": brief["conversationId"]},
            }
        ]
        assert adapter.handled == [], "the turn reached Hermes before its tools did"
        wire.release()
        await spin()
        return adapter, registry

    adapter, registry = asyncio.run(scenario())
    assert len(adapter.handled) == 1
    assert set(registry.tools) == {"list_boards", "delete_board"}


def test_without_the_feature_the_brief_is_handed_over_at_once_and_nothing_is_asked() -> None:
    async def scenario():
        adapter, wire, registry = await attached(features=False)
        await adapter._on_frame(adapter_golden("brief"))  # noqa: SLF001
        return adapter, wire, registry

    adapter, wire, registry = asyncio.run(scenario())
    assert len(adapter.handled) == 1
    assert wire.calls("tools.list") == [] and registry.tools == {}


def test_a_brief_withdrawn_while_it_waited_never_reaches_hermes_and_stops_nothing() -> None:
    async def scenario():
        adapter, wire, _ = await attached(hold=True)
        brief = adapter_golden("brief")
        await adapter._on_frame(brief)  # noqa: SLF001
        await spin()
        await adapter._on_frame(  # noqa: SLF001
            {"v": 1, "type": "cancel", "requestId": brief["requestId"], "reason": "superseded"}
        )
        wire.release()
        await spin()
        return adapter

    adapter = asyncio.run(scenario())
    assert adapter.handled == [], "a withdrawn brief was handed over, or a /stop was sent"


def test_a_tool_call_is_cited_on_the_next_post_and_only_there() -> None:
    async def scenario():
        adapter, wire, _ = await attached()
        brief = adapter_golden("brief")
        await adapter._on_frame(brief)  # noqa: SLF001
        await spin()
        await adapter._tools.invoke(brief["conversationId"], "list_boards", {})  # noqa: SLF001
        await adapter.send(brief["conversationId"], "Here are the boards.")
        await adapter.send(brief["conversationId"], "Anything else?")
        return wire, brief

    wire, brief = asyncio.run(scenario())
    tool_call = wire.calls("tools.call")[0]["payload"]
    assert tool_call == {"requestId": brief["requestId"], "name": "list_boards", "arguments": {}}
    posts = wire.of_type("post")
    assert posts[0]["citations"] == [RAN["callUrn"]]
    assert "citations" not in posts[1]


def test_without_the_feature_a_post_carries_no_citation_field() -> None:
    async def scenario():
        adapter, wire, _ = await attached(features=False)
        brief = adapter_golden("brief")
        await adapter._on_frame(brief)  # noqa: SLF001
        adapter._tools._citations[brief["requestId"]] = [RAN["callUrn"]]  # noqa: SLF001
        await adapter.send(brief["conversationId"], "Hi.")
        return wire

    post = asyncio.run(scenario()).of_type("post")[0]
    assert "citations" not in post and "activity" not in post


def test_a_post_that_failed_keeps_its_citations_for_the_next() -> None:
    async def scenario():
        adapter, wire, _ = await attached()
        brief = adapter_golden("brief")
        await adapter._on_frame(brief)  # noqa: SLF001
        await spin()
        await adapter._tools.invoke(brief["conversationId"], "list_boards", {})  # noqa: SLF001
        original = wire.send

        async def drop(frame: dict) -> bool:
            return False if frame.get("type") == "post" else await original(frame)

        wire.send = drop
        assert (await adapter.send(brief["conversationId"], "lost")).success is False
        wire.send = original
        await adapter.send(brief["conversationId"], "found")
        return wire

    assert asyncio.run(scenario()).of_type("post")[0]["citations"] == [RAN["callUrn"]]


def test_ready_asks_for_the_catalog_of_the_conversation_last_seen() -> None:
    async def scenario():
        adapter, wire, registry = await attached()
        adapter._known_conversation = "conv-earlier"  # noqa: SLF001
        adapter._message_handler = object()  # noqa: SLF001 — Hermes wired the adapter up
        await adapter._announce_ready_when_able()  # noqa: SLF001
        await spin()
        return wire, registry

    wire, registry = asyncio.run(scenario())
    assert [c["payload"] for c in wire.calls("tools.list")] == [{"conversationId": "conv-earlier"}]
    assert set(registry.tools) == {"list_boards", "delete_board"}


def test_the_plugin_observes_hermess_tool_calls_and_hands_its_context_to_the_adapter() -> None:
    from uffici_for_hermes import register

    class Ctx:
        def __init__(self) -> None:
            self.hooks: list[tuple[str, Any]] = []
            self.registry = Registry()
            self.register_tool = self.registry.register_tool

        def register_hook(self, name: str, callback: Any) -> None:
            self.hooks.append((name, callback))

        def register_platform(self, **entry: Any) -> None:
            self.entry = entry

    ctx = Ctx()
    register(ctx)
    assert ctx.hooks == [("post_tool_call", on_post_tool_call)]
    adapter = ctx.entry["adapter_factory"](PlatformConfig(enabled=True))
    adapter._companion_features = frozenset({"tools"})  # noqa: SLF001
    adapter._tools.apply(catalog_entries(BOARDS) or {})  # noqa: SLF001
    assert set(ctx.registry.tools) == {"list_boards", "delete_board"}
