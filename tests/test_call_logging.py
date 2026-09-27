"""🔴 THE CALL LOG-CLASSIFICATION GUARD (#2028 requirement 7).

A ``call``'s payload and its ``result``'s payload carry transcript content and
memory rows. Calls and results carrying a recognisable marker are driven
through the real objects — the adapter, the call client, the wire client's
decoder and its send-failure path — and every log record the run produced,
extras included, is searched for the marker.
"""

from __future__ import annotations

import asyncio
import json
import logging

from conftest import PlatformConfig
from uffici_for_hermes.adapter import UfficiAdapter
from uffici_for_hermes.calls import MAX_CALL_PAYLOAD_BYTES
from uffici_for_hermes.wire_client import WireAddress, WireClient

MARKER = "the Lisbon invoice for 48,000 EUR is unpaid"


class CapturingWire:
    """Stands in for the wire client: records what the adapter sends."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, frame: dict) -> bool:
        self.sent.append(frame)
        return True

    async def stop(self) -> None:
        return None


class BrokenConnection:
    """A connection whose send fails — the path that logs the frame it dropped."""

    async def send(self, _raw: str) -> None:
        raise ConnectionResetError("the socket went away")


def test_no_call_or_result_payload_reaches_a_log_line(caplog) -> None:
    caplog.set_level(logging.DEBUG)

    async def scenario() -> tuple[list[dict], list[bool]]:
        adapter = UfficiAdapter(PlatformConfig(enabled=True), get_env=lambda _n: None)
        wire = CapturingWire()
        adapter._client = wire
        answered: list[bool] = []

        # An answered call: the marker goes out in the payload and comes back.
        echo = asyncio.create_task(
            adapter.calls.call("memory.recall", {"text": MARKER}, deadline_ms=5000)
        )
        await asyncio.sleep(0)
        call_id = wire.sent[-1]["callId"]
        await adapter._on_frame(
            {"v": 1, "type": "result", "callId": call_id, "payload": {"text": MARKER}}
        )
        answered.append((await echo).payload == {"text": MARKER})

        # A refused call whose upstream error carries the marker.
        refused = asyncio.create_task(
            adapter.calls.call("memory.recall", {"text": MARKER}, deadline_ms=5000)
        )
        await asyncio.sleep(0)
        await adapter._on_frame({
            "v": 1, "type": "result", "callId": wire.sent[-1]["callId"],
            "error": {"class": "upstream_failed", "instruction": "Try again shortly.",
                      "upstream": {"detail": MARKER}},
        })
        answered.append((await refused).upstream == {"detail": MARKER})

        # A malformed result for a call in flight, carrying the marker twice.
        malformed = asyncio.create_task(adapter.calls.call("memory.recall", deadline_ms=5000))
        await asyncio.sleep(0)
        await adapter._on_frame({
            "v": 1, "type": "result", "callId": wire.sent[-1]["callId"],
            "payload": {"text": MARKER}, "error": {"class": "timed_out", "instruction": MARKER},
        })
        answered.append((await malformed).error_class == "malformed_payload")

        # A result answering nothing — discarded, with the marker in it.
        await adapter._on_frame(
            {"v": 1, "type": "result", "callId": "c-nobody", "payload": {"text": MARKER}}
        )

        # An oversized result, decoded by the wire client's own decoder.
        big = json.dumps({"v": 1, "type": "result", "callId": "c-big",
                          "payload": {"text": MARKER + "x" * MAX_CALL_PAYLOAD_BYTES}})
        decoded = WireClient._decode(big)
        answered.append(decoded is not None and decoded["error"]["class"] == "malformed_payload")

        # An oversized call, refused before it is sent.
        over = await adapter.calls.call(
            "memory.remember", {"text": MARKER + "x" * MAX_CALL_PAYLOAD_BYTES}, deadline_ms=5000
        )
        answered.append(over.error_class == "malformed_payload")

        # The send-failure paths, which log the frame they dropped.
        adapter._client = None
        dropped = await adapter.calls.call("memory.recall", {"text": MARKER}, deadline_ms=5000)
        answered.append(dropped.error_class == "upstream_failed")
        client = WireClient(
            address=WireAddress(kind="tcp", url="ws://127.0.0.1:7411/v1/ring"), token="t",
            harness_version="0.21.4", plugin_version="0.1.0", budget_tokens=48000,
            connect_fn=None, on_frame=None,  # type: ignore[arg-type]
        )
        client._connection = BrokenConnection()
        answered.append(
            await client.send({"v": 1, "type": "call", "callId": "c-x", "op": "memory.recall",
                               "deadlineMs": 5000, "payload": {"text": MARKER}}) is False
        )
        return wire.sent, answered

    sent, answered = asyncio.run(scenario())

    # 🔴 Non-vacuity: the marker really travelled, every path really ran, and a
    # log exists — otherwise its absence proves nothing.
    assert any(MARKER in json.dumps(frame) for frame in sent), "the marker never left in a call"
    assert all(answered), f"a path did not run as intended: {answered}"
    assert caplog.records, "the run produced no log records, so this guard looked at nothing"

    rendered = "\n".join(
        f"{record.getMessage()} {json.dumps(record.__dict__, default=str)}"
        for record in caplog.records
    )
    assert MARKER not in rendered and "Lisbon" not in rendered, (
        "a call or result payload reached a log line — payloads carry transcript content "
        "and memory rows, and only the op, the callId and the class may be logged"
    )
    assert "c-nobody" in rendered, "the discarded result's callId should be what its line names"
