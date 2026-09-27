"""The plugin's half of the call tunnel (#2028).

A ``call`` goes out with a plugin-chosen ``callId``; the one ``result`` carrying
that id comes back, in any order relative to the others. These are the rules
only the plugin can see: which call a result answers, what happens to a result
that answers nothing, and that a call never waits past its own deadline.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest

from uffici_for_hermes.call_contract import OP_PATTERN, encode_frame, encoded_size
from uffici_for_hermes.calls import (
    CALL_ERROR_CLASSES,
    MAX_CALL_PAYLOAD_BYTES,
    CallClient,
)
from uffici_for_hermes.wire import WIRE_V1


class Outbox:
    """Captures what the client sends, and can refuse to send."""

    def __init__(self, *, deliver: bool = True) -> None:
        self.frames: list[dict] = []
        self.deliver = deliver

    async def send(self, frame: dict) -> bool:
        self.frames.append(frame)
        return self.deliver


def result(call_id: str, **body) -> dict:
    return {"v": 1, "type": "result", "callId": call_id, **body}


def golden(kind: str, name: str) -> dict:
    return json.loads((WIRE_V1 / "golden" / kind / name).read_text(encoding="utf-8"))


def test_the_call_frame_is_the_contracts_shape() -> None:
    async def scenario() -> None:
        outbox = Outbox()
        client = CallClient(outbox.send)
        task = asyncio.create_task(client.call("memory.recall", {"query": "q"}, deadline_ms=5000))
        await asyncio.sleep(0)
        frame = outbox.frames[0]
        assert frame["v"] == 1 and frame["type"] == "call"
        assert frame["op"] == "memory.recall" and frame["deadlineMs"] == 5000
        assert frame["payload"] == {"query": "q"}
        # A UUID v4, as the contract recommends — and at most 64 characters.
        assert uuid.UUID(frame["callId"]).version == 4
        client.on_result(result(frame["callId"], payload={}))
        assert (await task).ok

    asyncio.run(scenario())


def test_a_call_with_no_payload_omits_it() -> None:
    async def scenario() -> None:
        outbox = Outbox()
        client = CallClient(outbox.send)
        task = asyncio.create_task(client.call("conversation.context", deadline_ms=5000))
        await asyncio.sleep(0)
        assert "payload" not in outbox.frames[0]
        client.on_result(result(outbox.frames[0]["callId"], payload={"ok": True}))
        answered = await task
        assert answered.ok and answered.payload == {"ok": True}

    asyncio.run(scenario())


def test_two_calls_resolve_independently_and_out_of_order() -> None:
    async def scenario() -> None:
        outbox = Outbox()
        client = CallClient(outbox.send)
        first = asyncio.create_task(client.call("slow", deadline_ms=5000))
        second = asyncio.create_task(client.call("fast", deadline_ms=5000))
        await asyncio.sleep(0)
        first_id, second_id = (f["callId"] for f in outbox.frames)
        assert first_id != second_id

        # The second call's result arrives first, and resolves only the second.
        client.on_result(result(second_id, payload={"which": "fast"}))
        assert (await second).payload == {"which": "fast"}
        assert not first.done()
        client.on_result(result(first_id, payload={"which": "slow"}))
        assert (await first).payload == {"which": "slow"}

    asyncio.run(scenario())


def test_an_error_result_carries_its_class_instruction_and_upstream() -> None:
    async def scenario() -> None:
        outbox = Outbox()
        client = CallClient(outbox.send)
        task = asyncio.create_task(client.call("memory.recall", deadline_ms=5000))
        await asyncio.sleep(0)
        client.on_result(result(outbox.frames[0]["callId"], error={
            "class": "not_permitted", "instruction": "Memory is off; tell the person.",
            "upstream": {"class": "memory_disabled"},
        }))
        answered = await task
        assert not answered.ok
        assert answered.error_class == "not_permitted"
        assert answered.instruction == "Memory is off; tell the person."
        assert answered.upstream == {"class": "memory_disabled"}

    asyncio.run(scenario())


def test_the_valid_result_goldens_resolve_the_call_they_name() -> None:
    async def scenario() -> None:
        for name in ("result-success.json", "result-unknown-op.json", "result-timed-out.json"):
            frame = golden("valid", name)
            outbox = Outbox()
            client = CallClient(outbox.send, new_id=lambda call_id=frame["callId"]: call_id)
            task = asyncio.create_task(client.call("op", deadline_ms=5000))
            await asyncio.sleep(0)
            assert client.on_result(frame) is True, name
            answered = await task
            assert answered.ok == ("payload" in frame), name

    asyncio.run(scenario())


def test_a_result_answering_nothing_is_discarded_the_unknown_call_golden() -> None:
    """``unknown_call--`` is the one invalid golden only the plugin can see."""
    frame = golden("invalid", "unknown_call--result-echoes-nothing-sent.json")
    client = CallClient(Outbox().send)
    assert client.on_result(frame) is False  # discarded, not raised


def test_a_result_carrying_both_payload_and_error_fails_its_call_malformed() -> None:
    async def scenario() -> None:
        frame = golden("invalid", "malformed_frame--result-payload-and-error.json")
        outbox = Outbox()
        client = CallClient(outbox.send, new_id=lambda: frame["callId"])
        task = asyncio.create_task(client.call("op", deadline_ms=5000))
        await asyncio.sleep(0)
        client.on_result(frame)
        # Not left to hang until the deadline: the call it names learns now.
        assert (await task).error_class == "malformed_payload"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "body",
    [
        {},  # neither
        {"payload": "not an object"},
        {"error": {"class": "no_such_class", "instruction": "x"}},
        {"error": {"class": "timed_out"}},  # no instruction
        {"error": "not an object"},
    ],
)
def test_a_result_breaking_its_shape_fails_its_call_malformed(body: dict) -> None:
    async def scenario() -> None:
        outbox = Outbox()
        client = CallClient(outbox.send, new_id=lambda: "c-1")
        task = asyncio.create_task(client.call("op", deadline_ms=5000))
        await asyncio.sleep(0)
        client.on_result(result("c-1", **body))
        assert (await task).error_class == "malformed_payload"

    asyncio.run(scenario())


def test_a_call_over_its_bound_is_never_sent() -> None:
    """The bound holds in both directions; the plugin enforces its own side."""

    async def scenario() -> None:
        outbox = Outbox()
        client = CallClient(outbox.send)
        answered = await client.call(
            "memory.remember", {"text": "x" * MAX_CALL_PAYLOAD_BYTES}, deadline_ms=5000
        )
        assert outbox.frames == []
        assert answered.error_class == "malformed_payload"

    asyncio.run(scenario())


def test_a_call_that_cannot_be_sent_fails_upstream_at_once() -> None:
    async def scenario() -> None:
        client = CallClient(Outbox(deliver=False).send)
        answered = await client.call("op", deadline_ms=5000)
        assert answered.error_class == "upstream_failed"

    asyncio.run(scenario())


def test_a_call_never_waits_past_its_own_deadline() -> None:
    async def scenario() -> None:
        client = CallClient(Outbox().send, grace_seconds=0.01)
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        answered = await client.call("op", deadline_ms=30)
        assert answered.error_class == "timed_out"
        assert loop.time() - started_at < 1.0
        # A result arriving after the plugin gave up is discarded, not raised.
        assert client.on_result(result(answered.call_id, payload={})) is False

    asyncio.run(scenario())


def test_losing_the_connection_fails_every_call_in_flight() -> None:
    async def scenario() -> None:
        outbox = Outbox()
        client = CallClient(outbox.send)
        tasks = [asyncio.create_task(client.call("op", deadline_ms=300_000)) for _ in range(3)]
        await asyncio.sleep(0)
        client.fail_all()
        for task in tasks:
            assert (await task).error_class == "upstream_failed"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("op", "deadline_ms"),
    [
        ("", 5000),
        ("x" * 129, 5000),
        ("Memory.Recall", 5000),  # a name is lower-case
        ("memory recall", 5000),  # and never free text: `op` is a log field
        ("op", 0),
        ("op", 300_001),
        ("op", True),
    ],
)
def test_a_call_outside_the_contract_is_a_programming_error(op: str, deadline_ms: int) -> None:
    with pytest.raises(ValueError):
        asyncio.run(CallClient(Outbox().send).call(op, deadline_ms=deadline_ms))


def test_the_classes_are_the_schemas_closed_five() -> None:
    schema = json.loads((WIRE_V1 / "schema" / "result.schema.json").read_text(encoding="utf-8"))
    enum = schema["properties"]["error"]["properties"]["class"]["enum"]
    assert list(CALL_ERROR_CLASSES) == enum


def test_the_bound_is_the_contracts() -> None:
    assert MAX_CALL_PAYLOAD_BYTES == 262_144


def test_a_call_id_over_its_bound_is_a_programming_error() -> None:
    client = CallClient(Outbox().send, new_id=lambda: "c" * 65)
    with pytest.raises(ValueError):
        asyncio.run(client.call("op", deadline_ms=5000))


def _call_of_exactly(size: int) -> tuple[CallClient, Outbox, dict]:
    """A call whose ENCODED frame is exactly ``size`` bytes, in multi-byte text.

    🔴 Multi-byte on purpose: "ç" is two bytes in UTF-8 and six as a ``\\u00e7``
    escape, so a client that measured one encoding and sent another would pass
    an ASCII boundary test and break this one.
    """
    outbox = Outbox()
    client = CallClient(outbox.send, new_id=lambda: "c-bound")
    frame = {"v": 1, "type": "call", "callId": "c-bound", "op": "memory.remember",
             "deadlineMs": 5000, "payload": {"text": ""}}
    base = encoded_size(frame)
    text = "ç" * ((size - base) // 2) + "x" * ((size - base) % 2)
    payload = {"text": text}
    assert encoded_size({**frame, "payload": payload}) == size
    return client, outbox, payload


def test_a_call_of_exactly_the_bound_is_sent_and_one_byte_more_is_not() -> None:
    async def scenario() -> None:
        client, outbox, payload = _call_of_exactly(MAX_CALL_PAYLOAD_BYTES)
        task = asyncio.create_task(client.call("memory.remember", payload, deadline_ms=5000))
        await asyncio.sleep(0)
        assert len(outbox.frames) == 1, "a call of exactly the bound must be sent"
        # And what is sent is exactly what was measured.
        assert len(encode_frame(outbox.frames[0]).encode("utf-8")) == MAX_CALL_PAYLOAD_BYTES
        client.on_result({"v": 1, "type": "result", "callId": "c-bound", "payload": {}})
        assert (await task).ok

        client, outbox, payload = _call_of_exactly(MAX_CALL_PAYLOAD_BYTES + 1)
        answered = await client.call("memory.remember", payload, deadline_ms=5000)
        assert outbox.frames == [] and answered.error_class == "malformed_payload"

    asyncio.run(scenario())


def test_a_send_that_stalls_cannot_hold_a_call_past_its_deadline() -> None:
    """A send awaits the socket's drain; a companion that stopped reading must not
    hold the call any longer than a slow answer could."""

    async def stalled(_frame: dict) -> bool:
        await asyncio.sleep(3600)
        return True

    async def scenario() -> None:
        client = CallClient(stalled, grace_seconds=0.01)
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        answered = await client.call("op", deadline_ms=30)
        assert answered.error_class == "timed_out"
        assert loop.time() - started_at < 1.0
        assert client.in_flight == 0

    asyncio.run(scenario())


def test_the_op_pattern_is_the_schemas() -> None:
    schema = json.loads((WIRE_V1 / "schema" / "call.schema.json").read_text(encoding="utf-8"))
    assert OP_PATTERN.pattern == schema["properties"]["op"]["pattern"]
