"""The adapter, against a fake companion replaying the golden frames."""

from __future__ import annotations

import asyncio
import json

import pytest

from conftest import (
    MessageType,
    PlatformConfig,
    ProcessingOutcome,
    SessionSource,
    build_session_key,
)
from uffici_for_hermes import adapter as adapter_module
from uffici_for_hermes.adapter import CHAT_TYPE, OPENER_HEADER, PLATFORM_NAME, UfficiAdapter
from uffici_for_hermes.wire import WIRE_V1


def golden(name: str) -> dict:
    return json.loads((WIRE_V1 / "golden" / "valid" / f"{name}.json").read_text())


class FakeWire:
    """The companion's side, replaying the ORDER the real one enforces.

    🔴 It sends the opener only after an accepted `fresh` ack, because that is
    what `ack.schema.json` says and what `manager.go` does — `OnAck` is the only
    caller of `SendOpener`. A fake that replayed a fixed script instead let the
    first cut of this adapter wait for an opener that its own missing ack was
    holding up, and the test PINNED that deadlock as correct. A real Hermes
    against a real companion then delivered three briefs, zero openers and
    fourteen unmet expectations.
    """

    def __init__(self, opener: dict | None = None) -> None:
        self.sent: list[dict] = []
        self.received: list[dict] = []
        #: What to answer a fresh ack with, if anything.
        self._opener = opener
        #: Filled in by the test so the wire can hand the opener back.
        self.adapter = None

    async def send(self, frame: dict) -> bool:
        self.sent.append(frame)
        if (
            self._opener is not None
            and frame.get("type") == "ack"
            and frame.get("outcome") == "accepted"
            and frame.get("session") == "fresh"
            and self.adapter is not None
        ):
            await self.adapter._on_frame({**self._opener, "requestId": frame["requestId"]})
        return True

    async def stop(self) -> None:
        return None

    def of_type(self, kind: str) -> list[dict]:
        return [f for f in self.sent if f.get("type") == kind]


class FakeStore:
    """Hermes's session store, as much of it as this plugin touches."""

    def __init__(self, keys: set[str] | None = None) -> None:
        self._entries = dict.fromkeys(keys or set(), object())
        self.resets: list[str] = []

    def reset_session(self, key: str) -> None:
        self.resets.append(key)
        self._entries.pop(key, None)


def build(
    store: FakeStore | None = None, *, opener: dict | None = None, **env
) -> tuple[UfficiAdapter, FakeWire]:
    adapter = UfficiAdapter(PlatformConfig(enabled=True), get_env=lambda name: env.get(name))
    wire = FakeWire(opener=opener)
    wire.adapter = adapter
    adapter._client = wire
    if store is not None:
        adapter.set_session_store(store)
    return adapter, wire


def run(coro):
    return asyncio.run(coro)


async def settle(adapter: UfficiAdapter) -> None:
    """Let the close's settle window elapse.

    🔴 `done` no longer fires inside `on_processing_complete`: a reply can still
    arrive after that hook, so the close waits for the replies to stop (see
    `DONE_SETTLE_SECONDS`). Tests drive that window to zero and spin.
    """
    # 🔴 BOTH windows. A turn that produced nothing waits the longer
    # `EMPTY_SETTLE_SECONDS` (#2014), so zeroing only the short one leaves every
    # silent-turn test spinning for fifteen real seconds and then failing on a
    # close that was going to happen.
    previous = (adapter_module.DONE_SETTLE_SECONDS, adapter_module.EMPTY_SETTLE_SECONDS)
    adapter_module.DONE_SETTLE_SECONDS = 0
    adapter_module.EMPTY_SETTLE_SECONDS = 0
    try:
        for _ in range(200):
            await asyncio.sleep(0)
            if not adapter._settling:
                return
    finally:
        adapter_module.DONE_SETTLE_SECONDS, adapter_module.EMPTY_SETTLE_SECONDS = previous


# --------------------------------------------------------------------------- policy


def test_authorization_is_declared_upstream() -> None:
    """🔴 The companion authenticates the wire and the platform decided
    participation before it rang, so Hermes's default-deny allowlist is asking
    the wrong question. Delegated, never absent."""
    adapter, _ = build()
    assert adapter.authorization_is_upstream is True


def test_the_plugin_never_reaches_for_an_allow_all_switch() -> None:
    """An allow-all environment variable would widen access to a conversation
    the platform never granted. There is nowhere in this package that names one."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src"
    sources = list(root.rglob("*.py"))
    assert sources, "no sources found; this scan looked at nothing"
    for path in sources:
        text = path.read_text()
        for forbidden in ("ALLOW_ALL_USERS", "allow_all_env", "allowed_users_env"):
            assert forbidden not in text, f"{path.name} names {forbidden}"


def test_code_blocks_are_supported() -> None:
    adapter, _ = build()
    assert adapter.supports_code_blocks is True


# --------------------------------------------------------------------------- session keying


def test_one_conversation_is_one_session_whoever_wrote() -> None:
    """🔴 The acceptance criterion. Two different people writing in one Uffici
    conversation must reach ONE Hermes session — the conversation IS the shared
    transcript. A ``group`` chat_type would key per user and split it in two,
    each half seeing a different conversation.
    """
    adapter, _ = build()
    first = adapter.build_source(chat_id="conv_1", chat_type=CHAT_TYPE, user_id="member_1")
    second = adapter.build_source(chat_id="conv_1", chat_type=CHAT_TYPE, user_id="member_2")
    assert build_session_key(first) == build_session_key(second)
    # Non-vacuity: the two sources really do differ in the field being ignored.
    assert first.user_id != second.user_id


def test_two_conversations_are_two_sessions() -> None:
    adapter, _ = build()
    one = adapter.build_source(chat_id="conv_1", chat_type=CHAT_TYPE, user_id="member_1")
    two = adapter.build_source(chat_id="conv_2", chat_type=CHAT_TYPE, user_id="member_1")
    assert build_session_key(one) != build_session_key(two)


# --------------------------------------------------------------------------- the brief


def test_a_fresh_brief_is_ACKED_first_so_the_opener_can_arrive() -> None:
    """🔴 THE ORDER, and the one that was wrong.

    The companion sends the opener only after an accepted `fresh` ack — it is
    `OnAck` that calls `SendOpener`. A plugin that held the brief waiting for the
    opener deadlocks both sides until the turn expires, and the first cut of this
    adapter did exactly that: a real Hermes against a real companion produced
    three briefs, zero openers and fourteen unmet expectations.

    The event is still HELD until the opener lands, so the recalled history
    enters the session with the first message. What moves first is the ack.
    """

    async def scenario():
        adapter, wire = build(FakeStore(), opener=golden("opener"))
        await adapter._on_frame(golden("brief"))

        acks = wire.of_type("ack")
        assert acks, "no ack was sent, so the companion would never send the opener"
        assert acks[0] == {
            "v": 1, "type": "ack", "requestId": "req_01J8ZQ7X4K2M9P0Q",
            "outcome": "accepted", "session": "fresh",
        }
        # And the opener the wire sent back in answer got the event admitted.
        assert len(adapter.handled) == 1, "the held event was never handed to the harness"

    run(scenario())


def test_the_event_is_not_handed_over_until_the_opener_lands() -> None:
    """The other half: the ack goes first, but the EVENT still waits, so the
    recalled history is stored with the first message rather than after it."""

    async def scenario():
        # No opener configured: the wire acks and sends nothing back.
        adapter, wire = build(FakeStore())
        await adapter._on_frame(golden("brief"))
        assert wire.of_type("ack"), "the ack must not wait for the opener"
        assert adapter.handled == [], "the event was handed over before its opener"

        await adapter._on_frame(golden("opener"))
        assert len(adapter.handled) == 1

    run(scenario())


def test_a_redelivered_brief_is_acked_again_and_admitted_once() -> None:
    """🔴 A redelivery is ordinary — the wire client reconnects on its own and
    the companion resends whatever it never got an ack for. Admitting it twice
    puts the same delta in the session twice and posts two replies to one turn.
    """

    async def scenario():
        adapter, wire = build(FakeStore(), opener=golden("opener"))
        await adapter._on_frame(golden("brief"))
        assert len(adapter.handled) == 1

        await adapter._on_frame(golden("brief"))
        assert len(adapter.handled) == 1, "the redelivered brief was admitted a second time"
        assert len(wire.of_type("ack")) == 2, "the redelivery was not acked again"

    run(scenario())


def test_the_opener_is_placed_user_side_exactly_once() -> None:
    """🔴 One of the five named mutations: placing it in ``channel_prompt``
    (the SYSTEM-side slot) must fail this test. That slot is for what is stable
    across a session — a per-turn opener there rewrites the system prompt every
    turn and costs the provider's cache."""

    async def scenario():
        adapter, _ = build(FakeStore())
        await adapter._on_frame(golden("brief"))
        await adapter._on_frame(golden("opener"))
        event = adapter.handled[0]

        assert event.channel_prompt is None, (
            "the opener reached the system-side slot; it is per-turn content"
        )
        assert event.channel_context is not None
        assert event.channel_context.count(OPENER_HEADER) == 1, "the opener was placed twice"
        assert "I will take a look." in event.channel_context, "the opener's entries are missing"

    run(scenario())


def test_a_second_brief_on_the_same_conversation_is_resumed_and_asks_for_no_opener() -> None:
    """The acceptance criterion: two briefs, one session, first fresh then
    resumed — and the resumed one is admitted immediately."""

    async def scenario():
        adapter, wire = build(FakeStore())
        await adapter._on_frame(golden("brief"))
        await adapter._on_frame(golden("opener"))

        # Hermes now has the session the first turn created.
        source = adapter.build_source(chat_id="conv_01J8ZQ7X4K2M9P0Q", chat_type=CHAT_TYPE)
        adapter._session_store._entries[build_session_key(source)] = object()

        second = {**golden("brief"), "requestId": "req_2"}
        await adapter._on_frame(second)

        assert len(adapter.handled) == 2, "the resumed brief was held for an opener"
        sessions = [a["session"] for a in wire.of_type("ack")]
        assert sessions == ["fresh", "resumed"]

    run(scenario())


def test_reset_session_forces_fresh_and_resets_first() -> None:
    async def scenario():
        store = FakeStore()
        adapter, wire = build(store)
        source = adapter.build_source(chat_id="conv_01J8ZQ7X4K2M9P0Q", chat_type=CHAT_TYPE)
        store._entries[build_session_key(source)] = object()

        await adapter._on_frame({**golden("brief"), "resetSession": True})
        assert store.resets == [build_session_key(source)], "the session was not reset"
        await adapter._on_frame(golden("opener"))
        assert wire.of_type("ack")[0]["session"] == "fresh"

    run(scenario())


def test_an_unreadable_session_store_means_fresh() -> None:
    """🔴 The asymmetry, stated: a wrong "resumed" costs the model its whole
    recalled history; a wrong "fresh" costs one extra read."""

    class Broken:
        @property
        def _entries(self):
            raise RuntimeError("the store moved")

    async def scenario():
        adapter, wire = build(Broken())
        await adapter._on_frame(golden("brief"))
        await adapter._on_frame(golden("opener"))
        assert wire.of_type("ack")[0]["session"] == "fresh"

    run(scenario())


def test_the_brief_never_lets_transcript_drive_the_harness() -> None:
    """🔴 ``allow_gateway_control=False`` on every brief. A member who types
    "/stop" into a conversation is talking to the agent, not to the gateway."""

    async def scenario():
        adapter, _ = build(FakeStore())
        await adapter._on_frame({**golden("brief"), "resetSession": False})
        await adapter._on_frame(golden("opener"))
        event = adapter.handled[0]
        assert event.allow_gateway_control is False
        assert event.message_type is MessageType.TEXT

    run(scenario())


def test_the_sender_is_the_causes_author() -> None:
    async def scenario():
        adapter, _ = build(FakeStore())
        await adapter._on_frame(golden("brief"))
        await adapter._on_frame(golden("opener"))
        event = adapter.handled[0]
        assert event.user_id == "member_2" and event.user_name == "Ana Marques"

    run(scenario())


def test_a_brief_without_ids_is_discarded() -> None:
    async def scenario():
        adapter, wire = build(FakeStore())
        await adapter._on_frame({"v": 1, "type": "brief"})
        assert adapter.handled == [] and wire.sent == []

    run(scenario())


def test_an_unknown_frame_is_ignored_rather_than_fatal() -> None:
    async def scenario():
        adapter, wire = build(FakeStore())
        await adapter._on_frame({"v": 1, "type": "something_new", "requestId": "r"})
        assert wire.sent == []

    run(scenario())


def test_an_opener_for_no_held_request_is_ignored() -> None:
    async def scenario():
        adapter, wire = build(FakeStore())
        await adapter._on_frame(golden("opener"))
        assert adapter.handled == [] and wire.sent == []

    run(scenario())


def test_a_refused_resumed_brief_is_acked_then_closed_with_a_done() -> None:
    """🔴 BOTH paths ack on receipt, so `rejected` is unreachable and a refusal
    is a `done` either way.

    The wire says the doorbell is acknowledged on receipt, and the resumed path
    used to wait for the harness to take the event first. Under a real model
    that wait exceeded the companion's ack timeout and the brief was
    redelivered — a turn answered twice because the acknowledgement was late.
    """

    async def scenario():
        store = FakeStore()
        adapter, wire = build(store)
        source = adapter.build_source(chat_id="conv_01J8ZQ7X4K2M9P0Q", chat_type=CHAT_TYPE)
        store._entries[build_session_key(source)] = object()

        async def refuse(_event):
            return False

        adapter.handle_message = refuse
        await adapter._on_frame(golden("brief"))

        ack = wire.of_type("ack")[0]
        assert ack["outcome"] == "accepted" and ack["session"] == "resumed", (
            "the resumed ack must go out on receipt, before the harness is asked"
        )
        done = wire.of_type("done")[0]
        assert done["outcome"] == "error" and done["errorClass"] == "harness_error"

    run(scenario())


def test_a_refused_fresh_brief_is_closed_with_a_done_because_it_was_already_acked() -> None:
    """🔴 Its ack is what earned the opener, so a refusal afterwards cannot be a
    retroactive ack. `done {error}` is the frame for "this turn is over"."""

    async def scenario():
        adapter, wire = build(FakeStore(), opener=golden("opener"))

        async def refuse(_event):
            return False

        adapter.handle_message = refuse
        await adapter._on_frame(golden("brief"))

        assert wire.of_type("ack")[0]["outcome"] == "accepted"
        done = wire.of_type("done")[0]
        assert done["outcome"] == "error" and done["errorClass"] == "harness_error"

    run(scenario())


def test_a_harness_that_raises_never_puts_its_message_on_the_wire() -> None:
    async def scenario():
        adapter, wire = build(FakeStore(), opener=golden("opener"))

        async def explode(_event):
            raise RuntimeError("the provider said: your key sk-live-1234 is invalid")

        adapter.handle_message = explode
        await adapter._on_frame(golden("brief"))

        assert wire.of_type("done")[0]["errorClass"] == "harness_error"
        assert "sk-live-1234" not in json.dumps(wire.sent), "a provider message reached the wire"

    run(scenario())


def test_a_send_becomes_a_post_on_the_open_request() -> None:
    async def scenario():
        adapter, wire = build(FakeStore())
        await adapter._on_frame(golden("brief"))
        await adapter._on_frame(golden("opener"))
        result = await adapter.send("conv_01J8ZQ7X4K2M9P0Q", "yes, it is filed")
        assert result.success
        post = wire.of_type("post")[0]
        assert post["requestId"] == "req_01J8ZQ7X4K2M9P0Q"
        assert post["text"] == "yes, it is filed"

    run(scenario())


def test_a_send_with_no_open_request_is_dropped_never_guessed() -> None:
    """🔴 Posting to another request would put this reply in a turn it does not
    belong to, and the platform would attribute it to whoever that turn answered."""

    async def scenario():
        adapter, wire = build(FakeStore())
        result = await adapter.send("conv_unknown", "a reply with nowhere to go")
        assert result.success is False
        assert wire.sent == []

    run(scenario())


# --------------------------------------------------------------------------- the turn's end


@pytest.mark.parametrize(
    ("outcome", "posts", "expected"),
    [
        (ProcessingOutcome.SUCCESS, 1, "answered"),
        (ProcessingOutcome.SUCCESS, 0, "empty"),
        (ProcessingOutcome.FAILURE, 0, "error"),
        (ProcessingOutcome.FAILURE, 2, "error"),
        (ProcessingOutcome.CANCELLED, 1, "cancelled"),
    ],
)
def test_the_turns_end_maps_to_its_done_outcome(outcome, posts: int, expected: str) -> None:
    async def scenario():
        adapter, wire = build(FakeStore())
        await adapter._on_frame(golden("brief"))
        await adapter._on_frame(golden("opener"))
        for _ in range(posts):
            await adapter.send("conv_01J8ZQ7X4K2M9P0Q", "a reply")

        await adapter.on_processing_complete(adapter.handled[0], outcome)
        await settle(adapter)
        done = wire.of_type("done")[0]
        assert done["outcome"] == expected
        if expected == "error":
            assert done["errorClass"] == "harness_error"

    run(scenario())


def test_an_event_that_is_not_ours_closes_nothing() -> None:
    async def scenario():
        from conftest import MessageEvent

        adapter, wire = build(FakeStore())
        await adapter.on_processing_complete(MessageEvent(text="x"), ProcessingOutcome.SUCCESS)
        await settle(adapter)
        assert wire.sent == []

    run(scenario())


def test_the_request_is_forgotten_once_it_closes() -> None:
    async def scenario():
        adapter, wire = build(FakeStore())
        await adapter._on_frame(golden("brief"))
        await adapter._on_frame(golden("opener"))
        await adapter.on_processing_complete(adapter.handled[0], ProcessingOutcome.SUCCESS)
        await settle(adapter)
        # A send after the close has no open request and must be dropped.
        assert (await adapter.send("conv_01J8ZQ7X4K2M9P0Q", "late")).success is False

    run(scenario())


# --------------------------------------------------------------------------- notices


def test_a_hermes_platform_notice_is_never_posted(caplog) -> None:
    """#2102: Hermes's "No home channel is set" onboarding notice reached the person as
    the agent's first post. Notices are routed private and the private path posts
    nothing — while a real reply in the same request still posts."""

    async def scenario():
        adapter, wire = build(FakeStore())
        assert adapter.config.extra.get("notice_delivery") == "private"
        await adapter._on_frame(golden("brief"))
        await adapter._on_frame(golden("opener"))
        chat = golden("brief")["conversationId"]
        with caplog.at_level("INFO"):
            result = await adapter.send_private_notice(
                chat, "u-1", "📬 No home channel is set for Uffici. Type /sethome …"
            )
        assert result.success, "a failed private notice makes Hermes fall back to a public send"
        assert [f for f in wire.sent if f.get("type") == "post"] == [], "the notice was posted"
        assert "sethome" not in caplog.text, "the notice's text reached the log"
        await adapter.send(chat, "the real answer")
        assert [f["text"] for f in wire.sent if f.get("type") == "post"] == ["the real answer"]

    run(scenario())


def test_a_config_with_no_extra_still_routes_notices_private() -> None:
    config = PlatformConfig(enabled=True)
    config.extra = None
    adapter = UfficiAdapter(config)
    assert adapter.config.extra == {"notice_delivery": "private"}


def test_an_operator_choice_of_notice_delivery_is_kept() -> None:
    adapter = UfficiAdapter(PlatformConfig(enabled=True, extra={"notice_delivery": "public"}))
    assert adapter.config.extra["notice_delivery"] == "public"


# --------------------------------------------------------------------------- cancel


def test_a_cancel_becomes_hermess_own_stop() -> None:
    """🔴 ``allow_gateway_control=True`` on this ONE event: it is the companion
    speaking over an authenticated local wire, not transcript text."""

    async def scenario():
        adapter, _ = build(FakeStore())
        await adapter._on_frame(golden("brief"))
        await adapter._on_frame(golden("opener"))
        await adapter._on_frame(golden("cancel-superseded"))

        stop = adapter.handled[-1]
        assert stop.text == "/stop"
        assert stop.message_type is MessageType.COMMAND
        assert stop.allow_gateway_control is True
        assert stop.is_command(), "Hermes would not route this as a command"

    run(scenario())


def test_a_failed_cancel_stops_the_turn_and_logs_only_the_class(caplog) -> None:
    """#2077: the platform's failure reaches the plugin as a ``failed`` cancel.
    The turn stops exactly as for any cancel; the class is logged, and neither
    the instruction's text nor anything else from the frame is."""

    async def scenario():
        adapter, _ = build(FakeStore())
        await adapter._on_frame(golden("brief"))
        await adapter._on_frame(golden("opener"))
        with caplog.at_level("WARNING"):
            await adapter._on_frame(golden("cancel-failed"))

        stop = adapter.handled[-1]
        assert stop.text == "/stop"
        assert "platform_unavailable" in caplog.text
        assert "could not be delivered" not in caplog.text

    run(scenario())


def test_a_cancel_for_an_unknown_request_does_nothing() -> None:
    async def scenario():
        adapter, _ = build(FakeStore())
        await adapter._on_frame(golden("cancel-stopped"))
        assert adapter.handled == []

    run(scenario())


def test_after_a_supersede_the_next_brief_starts_a_new_turn() -> None:
    async def scenario():
        adapter, wire = build(FakeStore())
        await adapter._on_frame(golden("brief"))
        await adapter._on_frame(golden("opener"))
        await adapter._on_frame(golden("cancel-superseded"))

        source = adapter.build_source(chat_id="conv_01J8ZQ7X4K2M9P0Q", chat_type=CHAT_TYPE)
        adapter._session_store._entries[build_session_key(source)] = object()
        await adapter._on_frame({**golden("brief"), "requestId": "req_2"})

        assert wire.of_type("ack")[-1] == {
            "v": 1, "type": "ack", "requestId": "req_2",
            "outcome": "accepted", "session": "resumed",
        }

    run(scenario())


# --------------------------------------------------------------------------- misc


def test_chat_info_names_the_platform() -> None:
    async def scenario():
        adapter, _ = build()
        info = await adapter.get_chat_info("conv_1")
        assert info == {"id": "conv_1", "type": CHAT_TYPE, "platform": PLATFORM_NAME}

    run(scenario())


def test_connect_refuses_without_a_token_and_says_why(caplog) -> None:
    async def scenario():
        adapter, _ = build()
        adapter._client = None
        assert await adapter.connect() is False

    run(scenario())


def test_sending_without_a_connection_is_a_dropped_frame() -> None:
    async def scenario():
        adapter, _ = build()
        adapter._client = None
        assert await adapter._send({"v": 1, "type": "post"}) is False

    run(scenario())


def test_disconnect_is_safe_before_connect() -> None:
    async def scenario():
        adapter, _ = build()
        adapter._client = None
        await adapter.disconnect()

    run(scenario())


def test_a_source_the_plugin_builds_is_always_a_dm() -> None:
    adapter, _ = build()
    source: SessionSource = adapter.build_source(chat_id="conv_1", chat_type=CHAT_TYPE)
    assert source.chat_type == "dm"


def test_an_empty_reply_is_not_posted() -> None:
    """🔴 `post.schema.json` requires at least one character, and the companion's
    decoder closes the connection on a frame it refuses — so an empty reply
    would cost the whole conversation. It closes as `empty` instead."""

    async def scenario():
        adapter, wire = build(FakeStore(), opener=golden("opener"))
        await adapter._on_frame(golden("brief"))
        result = await adapter.send("conv_01J8ZQ7X4K2M9P0Q", "   \n  ")
        assert result.success is False
        assert wire.of_type("post") == []

    run(scenario())


def test_an_over_long_reply_is_clipped_rather_than_refused() -> None:
    """Clipped, because a truncated answer reaches the person and a refused
    frame reaches nobody."""

    async def scenario():
        from uffici_for_hermes.adapter import POST_TEXT_MAX

        adapter, wire = build(FakeStore(), opener=golden("opener"))
        await adapter._on_frame(golden("brief"))
        result = await adapter.send("conv_01J8ZQ7X4K2M9P0Q", "x" * (POST_TEXT_MAX + 500))
        assert result.success is True
        assert len(wire.of_type("post")[0]["text"]) == POST_TEXT_MAX

    run(scenario())


def test_a_superseded_requests_state_does_not_outlive_it() -> None:
    """🔴 The `_forget` guard that protects the NEW request's mapping has no
    other test: mutate it to an unconditional pop and everything else stays
    green."""

    async def scenario():
        adapter, _ = build(FakeStore(), opener=golden("opener"))
        await adapter._on_frame(golden("brief"))
        first = "req_01J8ZQ7X4K2M9P0Q"

        # A second ring on the same conversation, the way a supersede arrives.
        await adapter._on_frame({**golden("brief"), "requestId": "req_2"})
        assert adapter._open_request["conv_01J8ZQ7X4K2M9P0Q"] == "req_2"

        # The old request closing must not take the new one's mapping with it.
        await adapter._on_frame({**golden("cancel-superseded"), "requestId": first})
        assert adapter._open_request.get("conv_01J8ZQ7X4K2M9P0Q") == "req_2", (
            "closing the superseded request unmapped the conversation, so the new "
            "turn's replies would be dropped"
        )

    run(scenario())


# --------------------------------------------------------- the settle window


def test_a_reply_arriving_after_the_harness_said_the_turn_ended_is_still_delivered() -> None:
    """🔴 THE REASON THE SETTLE WINDOW EXISTS (#2014).

    `on_processing_complete` is not a reliable "the turn is over" signal.
    Measured against a real Hermes, the final reply landed about two seconds
    AFTER that hook — Hermes's own log says "Normal final-send NOT suppressed
    despite active stream consumer". Closing on the hook alone threw the answer
    away: the companion recorded `empty`, and the reply arrived for a request
    that no longer existed.
    """

    async def scenario():
        adapter, wire = build(FakeStore(), opener=golden("opener"))
        await adapter._on_frame(golden("brief"))

        # The harness says the turn is over, having posted nothing.
        await adapter.on_processing_complete(adapter.handled[0], ProcessingOutcome.SUCCESS)
        assert wire.of_type("done") == [], "the close fired before the replies had stopped"

        # ...and then the reply arrives, the way a real one did.
        result = await adapter.send("conv_01J8ZQ7X4K2M9P0Q", "yes, it is filed")
        assert result.success is True, "the late reply was dropped"
        assert wire.of_type("post"), "the late reply never reached the wire"

        await settle(adapter)
        done = wire.of_type("done")[0]
        assert done["outcome"] == "answered", (
            f"the turn closed as {done['outcome']} though it answered — the post count "
            "must be read when the close FIRES, not when the hook did"
        )

    run(scenario())


def test_a_silent_turn_still_closes_empty_after_the_window() -> None:
    """The other half: settling must not turn every quiet turn into a hang."""

    async def scenario():
        adapter, wire = build(FakeStore(), opener=golden("opener"))
        await adapter._on_frame(golden("brief"))
        await adapter.on_processing_complete(adapter.handled[0], ProcessingOutcome.SUCCESS)
        await settle(adapter)
        assert wire.of_type("done")[0]["outcome"] == "empty"

    run(scenario())


def test_a_cancel_during_the_settle_wins() -> None:
    """A withdrawn turn must not get a second terminal frame from a timer."""

    async def scenario():
        adapter, wire = build(FakeStore(), opener=golden("opener"))
        await adapter._on_frame(golden("brief"))
        await adapter.on_processing_complete(adapter.handled[0], ProcessingOutcome.SUCCESS)

        await adapter._on_frame(golden("cancel-stopped"))
        await settle(adapter)

        assert wire.of_type("done") == [], (
            "the settling close fired for a request the companion had already cancelled"
        )

    run(scenario())


def test_disconnecting_cancels_a_settling_close() -> None:
    """🔴 CANCELS it, which is not the same as forgetting it.

    The first version of this test asserted `not adapter._settling` after a
    disconnect — and `_forget_everything` clears that dict whether or not the
    tasks were cancelled, so deleting the `cancel()` loop left it green while an
    orphaned timer went on to fire a `done` against a dead client. It now holds
    the task and asks the task.
    """

    async def scenario():
        adapter, _ = build(FakeStore(), opener=golden("opener"))
        await adapter._on_frame(golden("brief"))
        await adapter.on_processing_complete(adapter.handled[0], ProcessingOutcome.SUCCESS)

        settling = list(adapter._settling.values())
        assert settling, "nothing was settling to begin with"
        task = settling[0]

        await adapter.disconnect()
        # Give the event loop a turn to process the cancellation.
        for _ in range(10):
            await asyncio.sleep(0)

        assert task.cancelled(), (
            "the settling close outlived the connection; it will fire a `done` at a "
            "client that is gone"
        )
        assert not adapter._settling

    run(scenario())


def test_the_closed_memory_evicts_the_oldest_not_an_arbitrary_id() -> None:
    """🔴 `set.pop()` removes an ARBITRARY element — as likely the id just added
    as any other. The bound is enforced either way, so nothing fails loudly; what
    breaks is the guarantee the memory exists for. A forgotten id makes the next
    lifecycle hook for that request look live, the plugin sends a second `done`,
    and the companion answers a frame after the plugin's OWN close with
    `unknown_request` — which drops the connection and every other request on it
    (#2024, and the asymmetry it leans on is #2023)."""
    from uffici_for_hermes.adapter import CLOSED_MEMORY

    adapter, _ = build()
    for index in range(CLOSED_MEMORY + 1):
        adapter._forget(f"req_{index}")

    assert len(adapter._closed) == CLOSED_MEMORY
    # The newest is still remembered: it is the one a late hook will name.
    assert f"req_{CLOSED_MEMORY}" in adapter._closed
    # The oldest is the one that left.
    assert "req_0" not in adapter._closed


def test_a_turn_that_has_produced_nothing_waits_longer_before_closing() -> None:
    """🔴 The settle window is asymmetric on purpose, and CI is why.

    `on_processing_complete` is not "the turn is over" — the window exists
    because the reply can still be coming. The risk of closing early is highest
    exactly when NOTHING has arrived yet, and that is the case the short window
    got wrong: on a cold harness the first turn's reply landed ~3s after the
    hook, the plugin closed at 2.5s as `empty`, forgot the request, and Hermes's
    answer was dropped with `no open request for conversation …` — a member's
    answer lost, reported as a turn that said nothing (#2014).

    Once a reply HAS arrived, 2.5s after the last one is plenty: the turn is
    demonstrably producing, and each post restarts the window.
    """
    from uffici_for_hermes.adapter import DONE_SETTLE_SECONDS, EMPTY_SETTLE_SECONDS

    assert EMPTY_SETTLE_SECONDS > DONE_SETTLE_SECONDS, (
        "a turn that has produced nothing must get at least as long as one that has"
    )

    adapter, _ = build()
    # No posts recorded for this request: the empty window applies.
    assert adapter._settle_window("req_silent") == EMPTY_SETTLE_SECONDS
    # One post recorded: the turn is producing, so the short window applies.
    adapter._posts["req_talking"] = 1
    assert adapter._settle_window("req_talking") == DONE_SETTLE_SECONDS


# ── The call tunnel through the adapter (#2028) ───────────────────────────────


def test_a_result_frame_reaches_the_call_it_names() -> None:
    async def scenario():
        adapter, wire = build(FakeStore())
        task = asyncio.create_task(adapter.calls.call("memory.recall", deadline_ms=5000))
        await asyncio.sleep(0)
        call = wire.sent[-1]
        assert call["type"] == "call"
        await adapter._on_frame({"v": 1, "type": "result", "callId": call["callId"],
                                 "payload": {"facts": []}})
        answered = await task
        assert answered.ok and answered.payload == {"facts": []}

    run(scenario())


def test_disconnecting_fails_every_call_in_flight() -> None:
    """The companion answers none of a detached plugin's calls; waiting out each
    deadline would only hang the turns that made them."""

    async def scenario():
        adapter, _ = build(FakeStore())
        tasks = [
            asyncio.create_task(adapter.calls.call("memory.recall", deadline_ms=300_000))
            for _ in range(2)
        ]
        await asyncio.sleep(0)
        assert adapter.calls.in_flight == 2
        await adapter.disconnect()
        for task in tasks:
            assert (await task).error_class == "upstream_failed"
        assert adapter.calls.in_flight == 0

    run(scenario())


# ── An approval escalation in wave 1 (#2164) ──────────────────────────────────


def test_the_runner_asks_this_adapter_for_approvals_and_never_posts_a_text_prompt() -> None:
    """🔴 Without an approval hook of its own, Hermes's runner sends its plain-text
    `/approve` prompt through `send` — which this adapter posts as the AGENT'S MESSAGE to a
    person who cannot type `/approve` (every brief carries `allow_gateway_control=False`).
    Measured on the first real-provider run: a 547-character "Hermes wants to run a command
    that needs your OK" post, and a turn that then sat on the approval until it expired."""
    assert UfficiAdapter.supports_exec_approval_buttons() is True


def test_an_approval_escalation_is_denied_at_once_with_a_reason_the_agent_can_relay(
    caplog,
) -> None:
    """🔴 Never a silent failure where an agent answers a person (#1958 decision 32).

    Hermes's smart-approval guardian escalates to a human whenever it is unsure — and on
    a reasoning model it is unsure every time (its 16-token verdict is spent on hidden
    reasoning, `finish_reason=length`). Wave 1 has no approval surface (stage 3 brings the
    tool call and the approval, #2125/#2128), so nobody can answer: waiting holds the
    turn until the request expires and the person hears nothing. Denying at once hands the
    agent a BLOCKED result carrying this reason, and the agent answers the person inside
    the deadline instead."""
    import logging

    from conftest import APPROVALS

    async def scenario():
        adapter, wire = build(FakeStore(), opener=golden("opener"))
        await adapter._on_frame(golden("brief"))
        session_key = "agent:main:uffici:dm:conv_01J8ZQ7X4K2M9P0Q"
        APPROVALS.pending[session_key] = [{"command": "curl https://example.test"}]

        with caplog.at_level(logging.INFO):
            result = await adapter.send_exec_approval(
                chat_id="conv_01J8ZQ7X4K2M9P0Q",
                command="curl https://example.test/?token=s3cret",
                session_key=session_key,
                description="network access",
            )

        assert result.success is True, (
            "a failed approval hook makes Hermes fall back to the plain-text prompt through send"
        )
        assert APPROVALS.pending[session_key] == [], "the pending approval was left waiting"
        denied = APPROVALS.resolved[-1]
        assert denied["choice"] == "deny"
        assert denied["reason"] == adapter_module.APPROVAL_UNAVAILABLE_REASON
        assert wire.of_type("post") == [], "the approval prompt reached the conversation"
        assert "s3cret" not in caplog.text and "example.test" not in caplog.text, (
            "the command is the model's text and may carry a credential; it is never logged"
        )

    run(scenario())


def test_an_approval_with_nothing_pending_still_answers_success() -> None:
    """A prompt whose wait already ended (withdrawn, timed out) resolves nothing — and must
    still not make the runner re-send it as plain text through `send`."""

    async def scenario():
        adapter, wire = build(FakeStore())
        result = await adapter.send_exec_approval(
            chat_id="conv_x", command="ls", session_key="agent:main:uffici:dm:conv_x"
        )
        assert result.success is True
        assert wire.sent == []

    run(scenario())


# ── A turn Hermes is still running is never closed (#2164) ───────────────────


class FakeRunner:
    """The gateway runner, as much of it as the busy probe reads."""

    def __init__(self, *, restoring: bool = False, running: set[str] | None = None) -> None:
        self._startup_restore_in_progress = restoring
        self.running = running or set()

    def _is_session_running(self, session_key: str) -> bool:
        return session_key in self.running


SESSION = "agent:main:uffici:dm:conv_01J8ZQ7X4K2M9P0Q"


async def spin(times: int = 50) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


def zero_windows() -> None:
    adapter_module.DONE_SETTLE_SECONDS = 0
    adapter_module.EMPTY_SETTLE_SECONDS = 0
    adapter_module.BUSY_RECHECK_SECONDS = 0


@pytest.mark.parametrize("busy", ["active", "pending", "running", "restoring"])
def test_a_turn_hermes_is_still_running_is_not_closed_until_it_is_idle(busy: str) -> None:
    """🔴 THE FIRST REAL-PROVIDER RUN'S `empty` (#2164).

    `on_processing_complete` fires when Hermes's message HANDLER returns, and on several
    paths that is before the agent has run at all: during the startup restore every
    inbound message is queued and the handler returns None at once (reported SUCCESS),
    and on a busy session the message is merged into the running turn. The turn then
    runs for real — measured: the reply came 25-40 s later — and the plugin, trusting
    the hook, had closed `empty` after its 15 s window and dropped the answer.

    Hermes's own idle is four things: no processing guard, no pending message, no
    running-turn slot in the runner, and no startup restore in flight. The close waits
    for all four."""

    async def scenario():
        zero_windows()
        adapter, wire = build(FakeStore(), opener=golden("opener"))
        adapter._active_sessions, adapter._pending_messages = {}, {}
        adapter.gateway_runner = FakeRunner()
        await adapter._on_frame(golden("brief"))
        await adapter.on_processing_complete(adapter.handled[0], ProcessingOutcome.SUCCESS)

        if busy == "active":
            adapter._active_sessions[SESSION] = asyncio.Event()
        elif busy == "pending":
            adapter._pending_messages[SESSION] = object()
        elif busy == "running":
            adapter.gateway_runner.running.add(SESSION)
        else:
            adapter.gateway_runner._startup_restore_in_progress = True

        await spin()
        assert wire.of_type("done") == [], f"closed while Hermes was still {busy}"

        # The turn runs, and its answer arrives on the request that is still open.
        assert (await adapter.send("conv_01J8ZQ7X4K2M9P0Q", "the invoice is paid")).success

        adapter._active_sessions.clear()
        adapter._pending_messages.clear()
        adapter.gateway_runner = FakeRunner()
        await spin()
        assert [f["outcome"] for f in wire.of_type("done")] == ["answered"]

    run(scenario())


def test_an_adapter_without_the_harness_internals_still_closes() -> None:
    """The probe reads Hermes internals that move between versions: absent means idle, so a
    renamed attribute costs the old behaviour, never a request that is held forever."""

    async def scenario():
        zero_windows()
        adapter, wire = build(FakeStore(), opener=golden("opener"))
        await adapter._on_frame(golden("brief"))
        await adapter.on_processing_complete(adapter.handled[0], ProcessingOutcome.SUCCESS)
        await spin()
        assert [f["outcome"] for f in wire.of_type("done")] == ["empty"]

    run(scenario())


def test_ready_waits_for_the_startup_restore() -> None:
    """🔴 A brief delivered while Hermes restores its sessions is QUEUED by the runner and
    reported complete at once, which is how the first ring of every real run closed
    `empty`. `ready` is not announced until the restore is over, so the companion holds
    the ring instead."""
    adapter, _ = build()
    adapter._message_handler = object()
    adapter.gateway_runner = FakeRunner(restoring=True)
    assert adapter.harness_is_ready() is False
    adapter.gateway_runner._startup_restore_in_progress = False
    assert adapter.harness_is_ready() is True


def test_a_superseded_request_stops_waiting_on_the_new_turn() -> None:
    """The busy state after a supersede is the NEW turn's; the old request must not wait
    on it, or it closes only when the whole conversation goes quiet."""

    async def scenario():
        zero_windows()
        adapter, wire = build(FakeStore(), opener=golden("opener"))
        adapter._active_sessions = {SESSION: asyncio.Event()}
        await adapter._on_frame(golden("brief"))
        old = golden("brief")["requestId"]
        await adapter.on_processing_complete(adapter.handled[0], ProcessingOutcome.SUCCESS)
        await spin()
        assert wire.of_type("done") == []

        await adapter._on_frame({**golden("brief"), "requestId": "req_newer"})
        await spin()
        assert [f["requestId"] for f in wire.of_type("done")] == [old]

    run(scenario())


def test_a_session_that_never_goes_idle_is_closed_by_the_backstop(caplog) -> None:
    """A Hermes internal left set must not hold a request for the life of the process."""
    import logging

    async def scenario():
        zero_windows()
        adapter_module.BUSY_MAX_SECONDS = 0
        adapter, wire = build(FakeStore(), opener=golden("opener"))
        adapter._active_sessions = {SESSION: asyncio.Event()}
        await adapter._on_frame(golden("brief"))
        with caplog.at_level(logging.WARNING):
            await adapter.on_processing_complete(adapter.handled[0], ProcessingOutcome.SUCCESS)
            await spin()
        assert [f["outcome"] for f in wire.of_type("done")] == ["empty"]
        assert "still shows its session busy" in caplog.text

    run(scenario())


def test_an_approval_without_a_session_key_uses_the_conversations_request() -> None:
    from conftest import APPROVALS

    async def scenario():
        adapter, _ = build(FakeStore(), opener=golden("opener"))
        await adapter._on_frame(golden("brief"))
        APPROVALS.pending[SESSION] = [{"command": "ls"}]
        await adapter.send_exec_approval(
            chat_id="conv_01J8ZQ7X4K2M9P0Q", command="ls", session_key=""
        )
        assert APPROVALS.pending[SESSION] == [], "an empty key left the agent waiting"

    run(scenario())
