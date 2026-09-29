"""The wire, from the plugin's side, against an in-process fake companion."""

import asyncio
import json

import pytest

from uffici_for_hermes.calls import MAX_CALL_PAYLOAD_BYTES
from uffici_for_hermes.wire_client import (
    ALLOW_REMOTE_ENV,
    BACKOFF_MAX_SECONDS,
    DEFAULT_URL,
    SOCKET_ENV,
    TOKEN_ENV,
    TOKEN_FILE_ENV,
    URL_ENV,
    WireAddress,
    WireClient,
    WireConfigError,
    backoff_seconds,
    describe_frame,
    plugin_hello,
    resolve_address,
    resolve_token,
)


def env(**values: str):
    return lambda name: values.get(name)


# --------------------------------------------------------------------------- address


def test_a_named_socket_wins() -> None:
    address = resolve_address(env(**{SOCKET_ENV: "/tmp/u/wire.sock", URL_ENV: "http://127.0.0.1:9"}))
    assert address.kind == "socket" and address.socket_path == "/tmp/u/wire.sock"


def test_the_default_is_the_companions_own_loopback_port() -> None:
    assert resolve_address(env()).url == f"ws://{DEFAULT_URL.split('//')[1]}/v1/ring"


def test_a_remote_companion_is_refused_and_the_refusal_names_the_switch() -> None:
    """🔴 The wire's only authentication is a local token. Pointing it at another
    machine sends that token across a network it was never meant for."""
    with pytest.raises(WireConfigError) as caught:
        resolve_address(env(**{URL_ENV: "http://companion.example.com:7411"}))
    assert ALLOW_REMOTE_ENV in str(caught.value)


def test_a_remote_companion_is_allowed_when_the_operator_says_so() -> None:
    address = resolve_address(
        env(**{URL_ENV: "http://host.docker.internal:7411", ALLOW_REMOTE_ENV: "1"})
    )
    assert address.kind == "tcp" and "host.docker.internal" in address.url


def test_a_scheme_that_is_not_the_local_wire_is_refused() -> None:
    with pytest.raises(WireConfigError):
        resolve_address(env(**{URL_ENV: "ftp://127.0.0.1:7411"}))


def test_the_address_describes_itself_without_its_path() -> None:
    """🔴 The socket lives inside ``$UFFICI_HOME``, beside the token file. Its
    path is a map to the token's location, so it never reaches a log line."""
    address = resolve_address(env(**{SOCKET_ENV: "/home/someone/.uffici/wire.sock"}))
    assert "/home/someone" not in address.describe()
    assert address.describe() == "<socket>"


# --------------------------------------------------------------------------- token


def test_the_token_comes_from_its_file(tmp_path) -> None:
    path = tmp_path / "wire-token"
    path.write_text("t" * 40 + "\n")
    assert resolve_token(env(**{TOKEN_FILE_ENV: str(path)})) == "t" * 40


def test_an_unreadable_token_file_is_reported_without_its_path(tmp_path) -> None:
    """🔴 ``OSError`` renders the path; the path is inside the token's directory."""
    missing = tmp_path / "secrets-dir" / "wire-token"
    with pytest.raises(WireConfigError) as caught:
        resolve_token(env(**{TOKEN_FILE_ENV: str(missing)}))
    assert "secrets-dir" not in str(caught.value)


def test_an_empty_token_file_is_refused(tmp_path) -> None:
    path = tmp_path / "wire-token"
    path.write_text("   \n")
    with pytest.raises(WireConfigError):
        resolve_token(env(**{TOKEN_FILE_ENV: str(path)}))


def test_the_environment_carries_the_token_for_a_container() -> None:
    assert resolve_token(env(**{TOKEN_ENV: "x" * 40})) == "x" * 40


def test_no_token_anywhere_names_both_ways_to_supply_one() -> None:
    with pytest.raises(WireConfigError) as caught:
        resolve_token(env())
    assert TOKEN_FILE_ENV in str(caught.value) and TOKEN_ENV in str(caught.value)


# --------------------------------------------------------------------------- logging


def test_a_frame_description_carries_ids_and_counts_and_no_content() -> None:
    """🔴 The allow-list is the guard. A deny-list would publish every field the
    wire gains until somebody noticed."""
    described = describe_frame(
        {
            "v": 1, "type": "brief", "requestId": "req_1", "conversationId": "conv_1",
            "reason": "mentioned", "token": "SECRET-TOKEN-VALUE",
            "delta": [{"text": "the Lisbon invoice is unpaid"}],
            "text": "a reply nobody should see in a log",
        }
    )
    rendered = json.dumps(described)
    for secret in ("SECRET-TOKEN-VALUE", "Lisbon", "nobody should see"):
        assert secret not in rendered, f"describe_frame published {secret!r}: {rendered}"
    assert described["deltaEntryCount"] == 1
    assert described["textChars"] == len("a reply nobody should see in a log")
    assert described["requestId"] == "req_1"


def test_hello_carries_the_token_and_the_budget() -> None:
    hello = plugin_hello("t" * 40, "0.21.3", "0.1.0", 12345)
    assert hello["harness"] == "hermes"
    assert hello["contextBudgetTokens"] == 12345
    assert hello["token"] == "t" * 40


def test_hello_names_this_harness_process_and_the_name_is_stable_within_it() -> None:
    """🔴 #2234: the companion tells a RESTARTED harness from a reconnect by this id.

    A restart must be handed what its predecessor acknowledged; a reconnect of the same
    process must not be (it is still answering). So the id is minted once per process —
    every hello this process sends carries the same one — and valid against the schema.
    """
    import re

    from uffici_for_hermes.wire import WIRE_V1

    first = plugin_hello("t" * 40, "0.21.3", "0.1.0", 12345)
    second = plugin_hello("t" * 40, "0.21.3", "0.1.0", 12345)
    assert first["processId"] == second["processId"]
    schema = json.loads((WIRE_V1 / "schema" / "hello.schema.json").read_text())
    rule = schema["$defs"]["pluginHello"]["properties"]["processId"]
    assert re.fullmatch(rule["pattern"], first["processId"])
    assert rule["minLength"] <= len(first["processId"]) <= rule["maxLength"]


# --------------------------------------------------------------------------- backoff


def test_backoff_grows_and_is_capped() -> None:
    assert backoff_seconds(1) < backoff_seconds(2) < backoff_seconds(3)
    assert backoff_seconds(99) == BACKOFF_MAX_SECONDS
    assert backoff_seconds(0) == backoff_seconds(1)


# --------------------------------------------------------------------------- the loop


class FakeCompanion:
    """An in-process companion: it records what it was sent and replays frames."""

    def __init__(self, script: list[dict] | None = None, fail_after: int | None = None) -> None:
        self.sent: list[dict] = []
        self._script = list(script or [])
        self._fail_after = fail_after
        self.closed = False

    async def send(self, raw: str) -> None:
        if self._fail_after is not None and len(self.sent) >= self._fail_after:
            raise ConnectionResetError("the companion went away")
        self.sent.append(json.loads(raw))

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self):
        async def gen():
            for frame in self._script:
                yield json.dumps(frame)

        return gen()


async def drive(client: WireClient, collected: list | None = None, want: int = 0) -> None:
    """Run the client until it has produced ``want`` items, then stop it.

    🔴 Not "run for N milliseconds". The client RECONNECTS, and a fake whose
    script replays on every reconnect will produce thousands of frames in a
    fixed window — which is the client working, and a test that counted them
    would be asserting on the speed of the machine.
    """
    task = asyncio.create_task(client.run())
    for _ in range(2000):
        if collected is not None and len(collected) >= want:
            break
        await asyncio.sleep(0)
    await client.stop()
    task.cancel()


def build(companion: FakeCompanion, received: list, **kwargs) -> WireClient:
    async def connect(_address):
        return companion

    async def on_frame(frame):
        received.append(frame)

    return WireClient(
        address=WireAddress(kind="socket", socket_path="/tmp/x", url="ws://companion/v1/ring"),
        token="t" * 40, harness_version="0.21.3", plugin_version="0.1.0", budget_tokens=1000,
        connect_fn=connect, on_frame=on_frame, sleep_fn=lambda _s: asyncio.sleep(0), **kwargs
    )


@pytest.mark.parametrize("_run", [1])
def test_hello_goes_first_and_briefs_reach_the_handler(_run: int) -> None:
    async def scenario():
        received: list[dict] = []
        companion = FakeCompanion(
            [
                {"v": 1, "type": "hello", "companionVersion": "0.1.0"},
                {"v": 1, "type": "brief", "requestId": "req_1", "conversationId": "conv_1"},
            ]
        )
        client = build(companion, received)
        await drive(client, received, want=1)
        assert companion.sent and companion.sent[0]["type"] == "hello"
        assert received, "no frame reached the handler at all"
        assert {f["type"] for f in received} == {"brief"}, (
            "the companion's own hello must be consumed by the client, not handed on"
        )

    asyncio.run(scenario())


def test_a_frame_of_another_version_is_discarded() -> None:
    async def scenario():
        received: list[dict] = []
        companion = FakeCompanion([{"v": 2, "type": "brief", "requestId": "req_1"}])
        client = build(companion, received)
        task = asyncio.create_task(client.run())
        for _ in range(200):
            await asyncio.sleep(0)
        await client.stop()
        task.cancel()
        assert received == [], "a frame of another protocol version reached the handler"

    asyncio.run(scenario())


@pytest.mark.parametrize("raw", ["not json", '"a string"', "[]"])
def test_a_frame_that_is_not_an_object_is_discarded(raw: str) -> None:
    assert WireClient._decode(raw) is None


def test_bytes_are_decoded() -> None:
    frame = WireClient._decode(b'{"v": 1, "type": "brief"}')
    assert frame is not None and frame["type"] == "brief"


def test_sending_without_a_connection_is_a_dropped_frame_not_a_raise() -> None:
    async def scenario():
        client = build(FakeCompanion(), [])
        assert await client.send({"v": 1, "type": "post", "requestId": "r", "text": "x"}) is False

    asyncio.run(scenario())


def test_a_send_that_fails_is_reported_rather_than_raised() -> None:
    """🔴 Every caller is a lifecycle hook. A post that cannot be delivered is a
    dropped post, never a failed turn for the harness."""

    async def scenario():
        companion = FakeCompanion([{"v": 1, "type": "hello", "companionVersion": "0.1.0"}],
                                  fail_after=1)
        received: list[dict] = []
        client = build(companion, received)
        task = asyncio.create_task(client.run())
        await asyncio.sleep(0.05)
        sent = await client.send({"v": 1, "type": "post", "requestId": "r", "text": "x"})
        await client.stop()
        task.cancel()
        assert sent is False

    asyncio.run(scenario())


def test_the_connection_reconnects_until_stopped() -> None:
    async def scenario():
        attempts = []

        async def connect(_address):
            attempts.append(1)
            return FakeCompanion([{"v": 1, "type": "hello", "companionVersion": "0.1.0"}])

        client = WireClient(
            address=WireAddress(kind="tcp", url="ws://127.0.0.1:7411/v1/ring"),
            token="t" * 40, harness_version="0.21.3", plugin_version="0.1.0", budget_tokens=10,
            connect_fn=connect, on_frame=lambda _f: asyncio.sleep(0),
            sleep_fn=lambda _s: asyncio.sleep(0),
        )
        task = asyncio.create_task(client.run())
        await asyncio.sleep(0.05)
        await client.stop()
        task.cancel()
        assert len(attempts) > 1, "the client connected once and gave up"

    asyncio.run(scenario())


def test_the_connected_callback_sees_the_companions_hello() -> None:
    async def scenario():
        seen: list[dict] = []
        companion = FakeCompanion(
            [{"v": 1, "type": "hello", "companionVersion": "0.1.0", "agentSlug": "smith"}]
        )
        client = build(companion, [], on_connected=lambda f: (seen.append(f), asyncio.sleep(0))[1])
        await drive(client, seen, want=1)
        assert seen and seen[0]["agentSlug"] == "smith"

    asyncio.run(scenario())


def test_a_token_file_that_is_not_text_is_refused_without_its_bytes(tmp_path) -> None:
    """🔴 ``UnicodeDecodeError`` renders the offending bytes — a fragment of the
    token — and is neither an ``OSError`` nor anything the adapter catches, so it
    would escape into whatever the harness logs at its platform boundary."""
    path = tmp_path / "wire-token"
    path.write_bytes(b"\xff\xfe secret-ish bytes \x00")
    with pytest.raises(WireConfigError) as caught:
        resolve_token(env(**{TOKEN_FILE_ENV: str(path)}))
    message = str(caught.value)
    assert "UTF-8" in message
    for leaked in ("secret-ish", "0xff", "\\xff"):
        assert leaked not in message, f"the refusal carries {leaked!r}: {message}"


# ── The call tunnel's side of the wire client (#2028) ─────────────────────────


def test_losing_the_connection_is_reported_so_calls_in_flight_can_fail() -> None:
    async def scenario():
        lost: list[int] = []
        companion = FakeCompanion([{"v": 1, "type": "hello", "companionVersion": "0.1.0"}])
        client = build(companion, [], on_disconnected=lambda: lost.append(1))
        await drive(client, lost, want=1)
        assert lost, "a connection ended and nobody was told — its calls would hang to the deadline"

    asyncio.run(scenario())


def test_an_oversized_result_is_answered_malformed_payload_for_its_call() -> None:
    raw = json.dumps({"v": 1, "type": "result", "callId": "c-1",
                      "payload": {"text": "x" * MAX_CALL_PAYLOAD_BYTES}})
    decoded = WireClient._decode(raw)
    assert decoded == {
        "v": 1, "type": "result", "callId": "c-1",
        "error": {"class": "malformed_payload", "instruction": decoded["error"]["instruction"]},
    }
    assert decoded["error"]["instruction"], "the agent must have a sentence to relay"


def test_a_result_at_its_bound_passes_through_untouched() -> None:
    frame = {"v": 1, "type": "result", "callId": "c-2", "payload": {"text": ""}}
    frame["payload"]["text"] = "x" * (MAX_CALL_PAYLOAD_BYTES - len(json.dumps(frame)))
    raw = json.dumps(frame)
    assert len(raw.encode("utf-8")) == MAX_CALL_PAYLOAD_BYTES
    assert WireClient._decode(raw) == frame


def test_an_oversized_frame_that_is_not_a_result_is_not_rewritten() -> None:
    # Only a result names a call to answer; anything else keeps its own rules.
    frame = {"v": 1, "type": "brief", "requestId": "r", "note": "x" * MAX_CALL_PAYLOAD_BYTES}
    assert WireClient._decode(json.dumps(frame)) == frame


def test_a_call_or_result_description_names_ids_and_class_never_the_payload() -> None:
    call = describe_frame({"v": 1, "type": "call", "callId": "c-1", "op": "memory.recall",
                           "deadlineMs": 5000, "payload": {"text": "the Lisbon invoice"}})
    assert call == {"type": "call", "callId": "c-1", "op": "memory.recall"}
    result = describe_frame({"v": 1, "type": "result", "callId": "c-1", "error": {
        "class": "upstream_failed", "instruction": "x", "upstream": {"detail": "Lisbon"}}})
    assert result == {"type": "result", "callId": "c-1", "callErrorClass": "upstream_failed"}
