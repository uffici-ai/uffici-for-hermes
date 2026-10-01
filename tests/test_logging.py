"""🔴 THE LOG-CLASSIFICATION GUARD (#2012 requirement 9).

A whole turn is driven through the adapter with distinctive content in every
place content can live — a member's message, another agent's, a tool line, an
opener entry, a reply the agent posts, the wire token — and the log that turn
produced is searched for all of it.

End to end rather than per call site, because the risk is not one line: it is
the adapter, the wire client, the renderer and whatever a future field adds.
Running the real objects is what makes the absence mean anything.
"""

from __future__ import annotations

import asyncio
import json
import logging

from conftest import PlatformConfig, ProcessingOutcome
from uffici_for_hermes import adapter as adapter_module
from uffici_for_hermes.adapter import UfficiAdapter

#: Every distinctive string this turn carries. If one of these reaches a log
#: line, transcript content or a credential has been published.
MEMBER_TEXT = "the Lisbon invoice for 48,000 EUR is unpaid"
AGENT_TEXT = "I asked Ana about the Lisbon account last Tuesday"
TOOL_TEXT = "queried table invoices where status = overdue"
OPENER_TEXT = "earlier we agreed to escalate to Marta Nogueira"
REPLY_TEXT = "yes, it was filed under 2026-Q3"
MEMBER_NAME = "Ana Marques"
AGENT_HANDLE = "@archer"
TOKEN = "wire-token-value-that-must-never-be-logged-0123456789"

SECRETS = (
    MEMBER_TEXT, AGENT_TEXT, TOOL_TEXT, OPENER_TEXT, REPLY_TEXT,
    MEMBER_NAME, AGENT_HANDLE, TOKEN,
)


def entry(entry_id: str, cursor: str, kind: str, text: str, author: dict) -> dict:
    return {
        "id": entry_id, "cursor": cursor, "at": "2026-09-17T14:01:00Z",
        "kind": kind, "text": text, "author": author,
    }


MEMBER = {"type": "member", "id": "m1", "label": MEMBER_NAME, "self": False}
AGENT = {"type": "agent", "id": "a7", "label": "Archer", "handle": AGENT_HANDLE, "self": False}

BRIEF = {
    "v": 1, "type": "brief", "requestId": "req_1", "conversationId": "conv_1",
    "reason": "mentioned", "cause": {"entryId": "e2", "authorId": "m1"},
    "deadline": "2026-09-17T14:10:00Z", "sinceCursor": "c0001", "throughCursor": "c0003",
    "resetSession": False,
    "delta": [
        entry("e2", "c0002", "message", MEMBER_TEXT, MEMBER),
        entry("e3", "c0003", "tool_line", TOOL_TEXT, AGENT),
    ],
}

OPENER = {
    "v": 1, "type": "opener", "requestId": "req_1",
    "entries": [
        entry("e0", "c0000", "message", OPENER_TEXT, MEMBER),
        entry("e1", "c0001", "message", AGENT_TEXT, AGENT),
    ],
}


class LoudWire:
    """A wire that LOGS every frame it is handed, the way a debugging operator
    would — so the guard covers the frames the adapter emits, not only the ones
    it receives."""

    def __init__(self, logger: logging.Logger) -> None:
        self.log = logger
        self.sent: list[dict] = []

    async def send(self, frame: dict) -> bool:
        self.sent.append(frame)
        from uffici_for_hermes.wire_client import describe_frame

        self.log.info("uffici: sent a frame", extra=describe_frame(frame))
        return True

    async def stop(self) -> None:
        return None


def test_no_transcript_content_and_no_token_reaches_a_log_line(caplog) -> None:
    caplog.set_level(logging.DEBUG)

    async def scenario():
        sent_hello: list[str] = []

        async def connect_fn(_address):
            class Connection:
                async def send(self, raw: str) -> None:
                    sent_hello.append(raw)

                async def close(self) -> None:
                    return None

                def __aiter__(self):
                    async def gen():
                        return
                        yield  # pragma: no cover

                    return gen()

            return Connection()

        adapter = UfficiAdapter(
            PlatformConfig(enabled=True),
            get_env=lambda name: {
                "UFFICI_WIRE_TOKEN": TOKEN,
                "UFFICI_CONTEXT_WINDOW_TOKENS": "32000",
            }.get(name),
            connect_fn=connect_fn,
        )
        wire = LoudWire(logging.getLogger("uffici_for_hermes.test"))

        # 🔴 CONNECT, so the token is actually read and put in a real `hello`.
        # The first cut handed the token to `get_env` and never called connect,
        # so `resolve_token` and `plugin_hello` never ran — it asserted that a
        # string nobody had read was absent from the log.
        await adapter.connect()
        for _ in range(60):
            await asyncio.sleep(0)

        adapter._client = wire
        await adapter._on_frame(BRIEF)
        await adapter._on_frame(OPENER)
        await adapter.send("conv_1", REPLY_TEXT)
        await adapter.on_processing_complete(adapter.handled[0], ProcessingOutcome.SUCCESS)
        # The close settles; drive it so the `done` frame is actually produced.
        adapter_module.DONE_SETTLE_SECONDS = 0
        for _ in range(200):
            await asyncio.sleep(0)
            if not adapter._settling:
                break

        # 🔴 The unknown-frame path, with CONTENT in it. An unknown frame is
        # exactly where a whole-frame dump is most tempting ("we do not know
        # what this is, so log all of it") and it is still a frame: a companion
        # newer than this plugin carries transcript in fields we cannot name.
        await adapter._on_frame(
            {
                "v": 1, "type": "mystery", "requestId": "req_1",
                "text": MEMBER_TEXT, "entries": OPENER["entries"],
                "note": AGENT_TEXT,
            }
        )
        await adapter.send("conv_gone", REPLY_TEXT)
        # What the plugin would hand the companion as `logs.push` (#2265): every line
        # its root handler kept over the turn, at the DEFAULT floors.
        shipped = adapter._log_handler.drain(10_000, 1 << 30)
        await adapter.disconnect()
        return wire, sent_hello, shipped

    wire, sent_hello, shipped = asyncio.run(scenario())

    # 🔴 Non-vacuity, both halves: the turn must have HAPPENED, and the log must
    # exist — a guard over an empty log asserts nothing at all.
    assert len(wire.sent) >= 3, f"the turn produced only {len(wire.sent)} frame(s)"
    assert caplog.records, "the run produced no log records, so this guard looked at nothing"
    # 🔴 And the token REALLY was read and sent: without this, "absent from the
    # log" is a fact about a string nobody touched.
    assert any(TOKEN in raw for raw in sent_hello), (
        "the token never reached a hello frame, so its absence from the log proves nothing"
    )

    rendered = "\n".join(
        f"{record.getMessage()} {json.dumps(record.__dict__, default=str)}"
        for record in caplog.records
    )
    for secret in SECRETS:
        assert secret not in rendered, (
            f"the log contains {secret!r} — transcript content, display names and the wire "
            f"token are never logged"
        )

    # 🔴 The same turn's logs.push payload (#2265): the lines the plugin ships carry none
    # of it either. Non-vacuity: the plugin's own INFO lines are shipped.
    payload = json.dumps(shipped)
    assert shipped, "the turn shipped no line, so the absence below proves nothing"
    for secret in SECRETS:
        assert secret not in payload, f"a logs.push payload carries {secret!r}"


def test_the_wire_token_is_in_no_logs_push_payload_at_any_level(caplog) -> None:
    """At the MOST verbose floors an owner can set, the token still never ships."""
    caplog.set_level(logging.DEBUG)

    sent: list[str] = []

    async def scenario():
        async def connect_fn(_address):
            class Connection:
                async def send(self, raw: str) -> None:
                    sent.append(raw)

                async def close(self) -> None:
                    return None

                def __aiter__(self):
                    async def gen():
                        return
                        yield  # pragma: no cover

                    return gen()

            return Connection()

        adapter = UfficiAdapter(
            PlatformConfig(enabled=True),
            get_env=lambda name: {
                "UFFICI_WIRE_TOKEN": TOKEN, "UFFICI_CONTEXT_WINDOW_TOKENS": "32000",
            }.get(name),
            connect_fn=connect_fn,
        )
        adapter._log_handler.set_levels("debug", "debug")
        await adapter.connect()
        for _ in range(60):
            await asyncio.sleep(0)
        adapter._client = LoudWire(logging.getLogger("uffici_for_hermes.test"))
        await adapter._on_frame(BRIEF)
        await adapter._on_frame(OPENER)
        await adapter.send("conv_1", REPLY_TEXT)
        shipped = adapter._log_handler.drain(10_000, 1 << 30)
        await adapter.disconnect()
        return shipped

    shipped = asyncio.run(scenario())
    # Non-vacuity: lines DID ship at these floors, and the token really was read and
    # put on the wire — so its absence is a fact about the shipping, not an idle run.
    assert shipped, "nothing shipped at debug, so this proves nothing"
    assert any(TOKEN in raw for raw in sent), "the token never reached a hello frame"
    assert TOKEN not in json.dumps(shipped)


def test_the_content_really_was_in_play(caplog) -> None:
    """The companion piece to the absence above: prove the same strings DID
    travel through the adapter, so "absent from the log" is a fact about the
    logging and not about an empty run."""

    async def scenario():
        adapter = UfficiAdapter(PlatformConfig(enabled=True), get_env=lambda _n: None)

        class Silent:
            def __init__(self) -> None:
                self.sent: list[dict] = []

            async def send(self, frame: dict) -> bool:
                self.sent.append(frame)
                return True

            async def stop(self) -> None:
                return None

        wire = Silent()
        adapter._client = wire
        await adapter._on_frame(BRIEF)
        await adapter._on_frame(OPENER)
        await adapter.send("conv_1", REPLY_TEXT)
        return adapter, wire

    adapter, wire = asyncio.run(scenario())
    event = adapter.handled[0]
    assert MEMBER_TEXT in event.text and TOOL_TEXT in event.text
    assert OPENER_TEXT in (event.channel_context or "")
    assert REPLY_TEXT in json.dumps(wire.sent)


def test_no_log_call_in_the_package_can_publish_a_whole_frame() -> None:
    """🔴 The structural half, and the first cut of it was mis-scoped three ways.

    It only looked at ``extra=`` on calls whose receiver was a bare name
    ``logger`` — so it passed a POSITIONAL dump (``logger.warning("%s", frame)``,
    the footgun `logging.md` names first), any other receiver
    (``self.log.info(...)``, which this very file uses), and ``print``.
    """
    import ast
    import pathlib

    #: Names that hold a whole wire frame or a run of entries.
    containers = {"frame", "raw", "entry", "entries", "delta", "event", "held", "hello"}
    #: Every method a logger answers to.
    levels = {"debug", "info", "warning", "error", "exception", "critical", "fatal", "log"}

    def is_describe_frame(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "describe_frame"
        )

    def is_container(node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id in containers

    root = pathlib.Path(__file__).resolve().parents[1] / "src"
    offenders: list[str] = []
    calls = 0
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            where = f"{path.name}:{node.lineno}"

            # 🔴 `print` in a package that runs inside somebody else's daemon is
            # an unclassified write to their stdout.
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                offenders.append(f"{where} (print)")
                continue

            # Any receiver, not just a bare `logger`.
            if not (isinstance(node.func, ast.Attribute) and node.func.attr in levels):
                continue
            calls += 1

            for argument in node.args:
                if is_container(argument):
                    offenders.append(f"{where} (positional whole frame)")
            for keyword in node.keywords:
                if keyword.arg != "extra":
                    continue
                value = keyword.value
                if is_describe_frame(value):
                    continue
                # A dict literal is not a free pass: ``extra={"frame": frame}``
                # publishes exactly what ``extra=frame`` would.
                if isinstance(value, ast.Dict) and all(
                    not is_container(item) for item in value.values
                ):
                    continue
                offenders.append(f"{where} (unclassified extra)")

    # Non-vacuity: there are log calls to inspect.
    assert calls > 0, "no log calls found; this scan walked the wrong tree"
    assert not offenders, f"a log call can publish a whole frame: {offenders}"
