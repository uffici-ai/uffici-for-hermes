"""The adapter, against a fake companion replaying the golden frames."""

from __future__ import annotations

import asyncio
import json

import pytest

from conftest import (
    CLARIFY,
    MessageEvent,
    MessageType,
    PlatformConfig,
    ProcessingOutcome,
    SessionSource,
    _followup_processing_hooks_apply,
    _run_followup_processing_hook,
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


def test_hermess_restart_notice_is_switched_off_for_uffici() -> None:
    """#2247: "⚠️ Hermes is restarting — … send any message …" reached the conversation as
    the agent's message. Hermes gates it per platform on the config the adapter is handed,
    and the adapter turns it off — even over an explicit ``true``, which Hermes's loader
    cannot tell from its own default."""
    config = PlatformConfig(enabled=True)
    assert config.gateway_restart_notification is True
    UfficiAdapter(config)
    assert config.gateway_restart_notification is False
    explicit = PlatformConfig(enabled=True, gateway_restart_notification=True)
    UfficiAdapter(explicit)
    assert explicit.gateway_restart_notification is False


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


def test_a_cancel_releases_a_pending_question_before_the_stop() -> None:
    """🔴 #2233: a turn waiting in ``clarify`` ignores ``/stop``.

    Hermes's clarify wait returns only on a resolve, its timeout (an hour by default) or
    ``clear_session``, and its gateway deliberately leaves a pending clarify in place for
    a slash command. So the cancelled turn stayed blocked, holding Hermes's durable
    session lease, and the next turn on the conversation waited behind it far past its
    request's deadline. Releasing the question FIRST is what lets the stop take.
    """

    async def scenario():
        adapter, _ = build(FakeStore())
        brief = golden("brief")
        await adapter._on_frame(brief)
        await adapter._on_frame(golden("opener"))
        session_key = adapter._session_key_of[brief["requestId"]]
        CLARIFY.register("clarify_1", session_key, "Which board?", ["Uffici", "Trello"])

        stops_while_pending: list[bool] = []
        admit = adapter.handle_message

        async def handle(event):
            stops_while_pending.append(CLARIFY.has_pending(session_key))
            return await admit(event)

        adapter.handle_message = handle
        await adapter._on_frame({**golden("cancel-superseded"), "requestId": brief["requestId"]})

        assert ("clear_session", session_key) in CLARIFY.calls
        assert adapter.handled[-1].text == "/stop"
        assert stops_while_pending == [False], (
            "the stop was sent while the question still held the turn"
        )

    run(scenario())


def test_a_cancel_with_no_pending_question_still_stops() -> None:
    async def scenario():
        adapter, _ = build(FakeStore())
        await adapter._on_frame(golden("brief"))
        await adapter._on_frame(golden("opener"))
        await adapter._on_frame(golden("cancel-superseded"))
        assert adapter.handled[-1].text == "/stop"

    run(scenario())


def test_a_status_line_is_never_a_post() -> None:
    """🔴 #2233: Hermes talking about itself is not the agent speaking.

    Hermes routes its status lines ("⏳ Still waiting for the other Hermes process…")
    to ``send_or_update_status`` when the adapter has one and to plain ``send``
    otherwise. Without it every line became a post on the open request — a wall of
    them, every fifteen seconds — and the posts counted as the agent answering, so the
    platform's "did not answer" note never fired when the request then expired.
    """

    async def scenario():
        adapter, wire = build(FakeStore())
        brief = golden("brief")
        await adapter._on_frame(brief)
        await adapter._on_frame(golden("opener"))

        result = await adapter.send_or_update_status(
            brief["conversationId"], "lease_wait",
            "⏳ Still waiting for the other Hermes process on this session (15s)...",
            metadata=None,
        )

        assert result.success is True, "a failure would send Hermes back to plain send"
        assert wire.of_type("post") == []
        assert adapter._posts.get(brief["requestId"], 0) == 0

    run(scenario())


def test_a_status_line_is_logged_by_key_and_length_only(caplog) -> None:
    async def scenario():
        adapter, _ = build(FakeStore())
        with caplog.at_level("DEBUG"):
            await adapter.send_or_update_status("conv_1", "lease_wait", "waiting on /home/x/secret")

    run(scenario())
    assert "lease_wait" in caplog.text
    assert "secret" not in caplog.text


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


def test_a_brief_hermes_drains_as_a_queued_follow_up_still_closes() -> None:
    """🔴 THE RESTART THAT NEVER CLOSED (#2246).

    After a restart Hermes auto-resumes the interrupted turn itself, so the redelivered
    brief lands on a BUSY session: Hermes parks it and drains it IN-BAND as a queued
    follow-up, never through the base adapter's own processing path. That drain fires the
    lifecycle hooks only for an adapter that overrides ``on_processing_start``
    (``_followup_processing_hooks_apply``). Without one the completion never came, and
    the answered request sat open until the platform's deadline cancelled it."""

    async def scenario():
        zero_windows()
        adapter, wire = build(FakeStore({SESSION}))
        adapter._active_sessions, adapter._pending_messages = {}, {}
        adapter.gateway_runner = FakeRunner(running={SESSION})
        await adapter._on_frame(golden("brief"))
        parked = adapter.handled[0]

        # Hermes's drain, in its order: start, the follow-up's turn, complete.
        await _run_followup_processing_hook(adapter, parked, "on_processing_start")
        assert (await adapter.send("conv_01J8ZQ7X4K2M9P0Q", "the invoice is paid")).success
        await _run_followup_processing_hook(
            adapter, parked, "on_processing_complete", ProcessingOutcome.SUCCESS
        )
        adapter.gateway_runner = FakeRunner()
        await spin()
        assert [f["outcome"] for f in wire.of_type("done")] == ["answered"]

    run(scenario())


def test_every_brief_qualifies_for_hermess_follow_up_hooks() -> None:
    """Both halves of Hermes's rule, stated where they can drift: the adapter overrides the
    start hook, and the event it hands over carries a ``message_id`` (#2246)."""

    async def scenario():
        adapter, _ = build(FakeStore(), opener=golden("opener"))
        await adapter._on_frame(golden("brief"))
        return adapter, adapter.handled[0]

    adapter, event = run(scenario())
    assert event.message_id
    assert _followup_processing_hooks_apply(adapter, event) is True


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


# --------------------------------------------------------------------------- questions (#2232)

QUESTION_HELLO = {"v": 1, "type": "hello", "companionVersion": "0.4.0", "features": ["question"]}


def answer_brief(text: str, request_id: str = "req_answer", **over) -> dict:
    """The next brief on the conversation: a member's reply, as the platform sends it."""
    base = golden("brief")
    return {
        **base,
        "requestId": request_id,
        "delta": [
            {
                "id": "entry_9", "cursor": "c0009", "at": "2026-09-17T14:05:00Z",
                "author": {
                    "type": "member", "id": "member_2", "label": "Ana Marques", "self": False,
                },
                "kind": "message", "text": text,
            }
        ],
        **over,
    }


async def asked(features: bool = True, choices=("Uffici board", "Trello")):
    """A turn on the brief's conversation that has asked a question through Hermes."""
    adapter, wire = build(FakeStore())
    if features:
        await adapter._on_companion_hello(QUESTION_HELLO)
    brief = golden("brief")
    await adapter._on_frame(brief)
    await adapter._on_frame(golden("opener"))
    session_key = adapter._session_key_of[brief["requestId"]]
    CLARIFY.register("clarify_1", session_key, "Which board?", list(choices) if choices else None)
    result = await adapter.send_clarify(
        brief["conversationId"], "Which board?", list(choices) if choices else None,
        "clarify_1", session_key,
    )
    return adapter, wire, brief, session_key, result


def test_a_question_reaches_the_person_as_the_card_and_settles_the_request() -> None:
    """🔴 #2232: Hermes's clarify was a numbered list typed back by hand.

    With a companion that can carry one, the question is a post WITH the question — the
    card — and the request is settled at once: the person sees the card rather than a
    working line, and the deadline no longer runs against a turn that is waiting on them.
    """

    async def scenario():
        adapter, wire, brief, _, result = await asked()
        assert result.success is True, "a failure would send Hermes to its text fallback"
        posts = wire.of_type("post")
        assert posts == [
            {
                "v": 1, "type": "post", "requestId": brief["requestId"], "text": "Which board?",
                "question": {"id": "clarify_1", "choices": ["Uffici board", "Trello"],
                             "multiSelect": False},
            }
        ]
        assert wire.of_type("done") == [
            {"v": 1, "type": "done", "requestId": brief["requestId"], "outcome": "answered"}
        ]
        assert ("mark_awaiting_text", "clarify_1") not in CLARIFY.calls
        pending = adapter._pending_questions[brief["conversationId"]]
        assert CLARIFY.has_pending(pending["sessionKey"])

    run(scenario())


def test_an_older_companion_gets_hermess_own_text_fallback() -> None:
    async def scenario():
        _, wire, brief, _, result = await asked(features=False)
        assert result.success is True
        [post] = wire.of_type("post")
        assert "question" not in post
        assert post["text"].startswith("❓ Which board?")
        assert wire.of_type("done") == [], "the text fallback waits on the open request"
        assert ("mark_awaiting_text", "clarify_1") in CLARIFY.calls

    run(scenario())


def test_the_answer_resolves_the_question_rather_than_starting_a_new_turn() -> None:
    """The person's answer arrives as the next brief. The turn that asked is still running,
    blocked in Hermes's clarify; the answer unblocks it, and a second turn would answer a
    conversation the first had not finished."""

    async def scenario():
        adapter, wire, brief, session_key, _ = await asked()
        handled_before = len(adapter.handled)
        await adapter._on_frame(answer_brief("Trello"))

        assert CLARIFY.entries["clarify_1"].response == "Trello"
        assert len(adapter.handled) == handled_before, "the answer became a new Hermes turn"
        assert wire.of_type("ack")[-1] == {
            "v": 1, "type": "ack", "requestId": "req_answer",
            "outcome": "accepted", "session": "resumed",
        }
        # The continuing turn speaks on the request that carried the answer.
        assert adapter._open_request[brief["conversationId"]] == "req_answer"
        assert adapter._session_key_of["req_answer"] == session_key
        await adapter.send(brief["conversationId"], "Here is the Trello board.")
        assert wire.of_type("post")[-1]["requestId"] == "req_answer"

    run(scenario())


def test_the_turn_that_asked_closes_the_request_that_answered() -> None:
    async def scenario():
        adapter, wire, brief, _, _ = await asked()
        original = adapter.handled[-1]
        await adapter._on_frame(answer_brief("Trello"))
        await adapter.send(brief["conversationId"], "Done.")
        await adapter.on_processing_complete(original, ProcessingOutcome.SUCCESS)
        await settle(adapter)
        assert wire.of_type("done")[-1] == {
            "v": 1, "type": "done", "requestId": "req_answer", "outcome": "answered"
        }

    run(scenario())


def test_a_number_picks_the_choice_and_free_text_is_the_other_answer() -> None:
    async def scenario():
        adapter, _, _, _, _ = await asked()
        await adapter._on_frame(answer_brief("2"))
        assert CLARIFY.entries["clarify_1"].response == "Trello"

    run(scenario())

    async def other():
        adapter, _, _, _, _ = await asked()
        await adapter._on_frame(answer_brief("the one Bruno made yesterday"))
        # "Other" on the card: not a choice, still the answer.
        assert CLARIFY.entries["clarify_1"].response == "the one Bruno made yesterday"

    run(other())


def test_a_question_hermes_already_gave_up_on_leaves_the_answer_an_ordinary_turn() -> None:
    async def scenario():
        adapter, _, _, session_key, _ = await asked()
        CLARIFY.clear_session(session_key)  # Hermes's own timeout, or a /new
        handled_before = len(adapter.handled)
        await adapter._on_frame(answer_brief("Trello"))
        await adapter._on_frame({**golden("opener"), "requestId": "req_answer"})
        assert len(adapter.handled) == handled_before + 1

    run(scenario())


def test_a_question_the_card_cannot_carry_falls_back_to_text() -> None:
    """An id outside the wire's alphabet would close the connection as malformed — so it
    never reaches the wire as a card."""

    async def scenario():
        adapter, wire = build(FakeStore())
        await adapter._on_companion_hello(QUESTION_HELLO)
        brief = golden("brief")
        await adapter._on_frame(brief)
        await adapter._on_frame(golden("opener"))
        key = adapter._session_key_of[brief["requestId"]]
        CLARIFY.register("id with spaces", key, "Which?", ["a", "b"])
        await adapter.send_clarify(
            brief["conversationId"], "Which?", ["a", "b"], "id with spaces", key
        )
        [post] = wire.of_type("post")
        assert "question" not in post

    run(scenario())


def test_a_question_whose_post_failed_upstream_is_released_by_its_cancel() -> None:
    """Review of #2232: the post can fail AFTER the plugin settled the request — the
    platform expired it, a 5xx — and the companion then sends `cancel{failed}` for a
    request the plugin already forgot. The question must not linger: Hermes's turn would
    hold the session, and the person's next "hello?" would be taken as the answer."""

    async def scenario():
        adapter, _, brief, session_key, _ = await asked()
        handled = len(adapter.handled)
        await adapter._on_frame({**golden("cancel-failed"), "requestId": brief["requestId"]})
        assert ("clear_session", session_key) in CLARIFY.calls
        assert adapter._pending_questions == {}
        # 🔴 #2272: and the turn STOPS. Released and left running, it carries on with an
        # empty answer, Hermes merges the person's next message into it, and that
        # message's request never closes — the completion names the withdrawn one.
        stops = [event for event in adapter.handled[handled:] if event.text == "/stop"]
        assert len(stops) == 1, "a withdrawn question request left its turn running"
        assert stops[0].source.chat_id == brief["conversationId"]
        assert stops[0].allow_gateway_control is True

    run(scenario())


def test_a_disconnect_releases_every_question_in_hermes() -> None:
    async def scenario():
        adapter, _, _, session_key, _ = await asked()
        await adapter.disconnect()
        assert ("clear_session", session_key) in CLARIFY.calls

    run(scenario())


def test_an_answer_that_arrives_with_more_brings_all_of_it_to_the_agent() -> None:
    """Nothing in the answering brief is dropped: with more than the answer in it, the
    whole delta, labelled, is the answer the waiting turn reads."""

    async def scenario():
        adapter, _, _, _, _ = await asked()
        brief = answer_brief("Trello")
        earlier = {**brief["delta"][0], "id": "entry_8", "text": "also add the budget"}
        brief["delta"].insert(0, earlier)
        await adapter._on_frame(brief)
        response = CLARIFY.entries["clarify_1"].response
        assert "also add the budget" in response and "Trello" in response

    run(scenario())


def test_while_a_card_question_waits_a_brief_no_person_spoke_in_is_held_not_queued() -> None:
    """A mention by another agent, a card arrival: queued behind the turn blocked in
    clarify, it would expire, and its cancel would end the person's question. It is closed
    `cancelled` at once instead — which leaves the offset, so the answer's ring re-covers it."""

    async def scenario():
        adapter, wire, _, session_key, _ = await asked()
        brief = answer_brief("I can take it", request_id="req_agent")
        brief["delta"][0]["author"] = {
            "type": "agent", "id": "a2", "label": "Archer", "self": False,
        }
        handled_before = len(adapter.handled)
        await adapter._on_frame(brief)
        assert len(adapter.handled) == handled_before, "it was queued behind the question"
        assert wire.of_type("done")[-1] == {
            "v": 1, "type": "done", "requestId": "req_agent", "outcome": "cancelled",
        }
        assert CLARIFY.has_pending(session_key), "the question was ended by an unrelated brief"

    run(scenario())


def test_a_choice_is_measured_as_the_platform_counts_it() -> None:
    """UTF-16 units, as the gateway's zod counts: 150 emoji are 150 code points and 300
    units, and a question the platform would refuse is asked in text instead."""
    from uffici_for_hermes.adapter import _question_card

    assert _question_card("Which?", ["😀" * 150], "q1", False) is None
    assert _question_card("Which?", ["😀" * 100], "q1", False) is not None


def startup_resume(chat_id: str = "conv_1") -> MessageEvent:
    """What Hermes hands the adapter for a restart-interrupted session on boot
    (``gateway/run_startup.py::_schedule_resume_pending_sessions``)."""
    adapter, _ = build(FakeStore())
    return MessageEvent(
        text="", source=adapter.build_source(chat_id=chat_id, chat_type="dm"), internal=True
    )


def test_hermess_startup_auto_resume_is_declined_when_the_companion_redelivers() -> None:
    """🔴 #2273: a restart mid-turn answered the person twice — Hermes's auto-resume
    turn, then the brief the companion redelivered (#2234). The companion's record of
    the work is the one answered; Hermes keeps the session resume-pending, so the
    redelivered brief carries its interruption note."""

    async def scenario():
        adapter, _ = build(FakeStore())
        await adapter._on_companion_hello(
            {"v": 1, "type": "hello", "features": ["question", "redelivery"]}
        )
        handled = len(adapter.handled)
        await adapter.handle_message(startup_resume())
        assert adapter.handled[handled:] == [], "Hermes's resume turn reached the harness"

        # A brief is never the resume, whatever the companion says.
        brief = MessageEvent(
            text="", source=startup_resume().source, internal=True,
            metadata={"uffici_request_id": "req_1"},
        )
        await adapter.handle_message(brief)
        assert adapter.handled[-1] is brief

    run(scenario())


def test_hermess_startup_auto_resume_runs_when_the_companion_does_not_redeliver() -> None:
    """An older companion hands a restarted harness nothing again, so Hermes's own
    resume is the only thing that finishes the interrupted turn."""

    async def scenario():
        adapter, _ = build(FakeStore())
        await adapter._on_companion_hello({"v": 1, "type": "hello", "features": ["question"]})
        resume = startup_resume()
        await adapter.handle_message(resume)
        assert adapter.handled[-1] is resume

    run(scenario())


def test_a_startup_resume_waits_for_the_companions_hello(monkeypatch) -> None:
    """#2273: Hermes resumes on the same boot the plugin connects on, so the resume can
    arrive before the companion's hello — deciding on an empty feature set would admit
    it. It waits; and with no hello at all it reads as an older companion."""

    async def scenario():
        adapter, _ = build(FakeStore())
        resume = startup_resume()
        pending = asyncio.ensure_future(adapter.handle_message(resume))
        await asyncio.sleep(0)
        await adapter._on_companion_hello(
            {"v": 1, "type": "hello", "features": ["question", "redelivery"]}
        )
        assert await pending is None, "a declined resume answers nothing"
        assert resume not in adapter.handled, "the resume was decided before the hello"

        monkeypatch.setattr(adapter_module, "HELLO_WAIT_SECONDS", 0.01)
        silent, _ = build(FakeStore())
        await silent.handle_message(resume)
        assert silent.handled[-1] is resume

    run(scenario())


def test_only_hermess_resume_shape_is_declined_and_only_for_the_companion_that_said_so() -> None:
    """#2273 review: the decline is narrow — an internal empty COMMAND is not Hermes's
    resume — and a companion's features end with its connection."""

    async def scenario():
        adapter, _ = build(FakeStore())
        await adapter._on_companion_hello(
            {"v": 1, "type": "hello", "features": ["question", "redelivery"]}
        )
        command = MessageEvent(
            text="", message_type=MessageType.COMMAND,
            source=startup_resume().source, internal=True,
        )
        await adapter.handle_message(command)
        assert adapter.handled[-1] is command

        await adapter.disconnect()
        assert adapter._companion_features == frozenset()
        assert adapter._hello_seen is None

    run(scenario())
