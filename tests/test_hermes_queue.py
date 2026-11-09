"""Hermes 0.22+'s durable input queue, as the plugin reconciles with it (#3124).

The stand-ins below are TRANSCRIBED from Hermes 0.23.0 (commit 2bd378ed):
``hermes_state_runtime.list_session_admissions`` (rows of ``session_admissions``, pending
= ``queued`` / ``started`` / ``unknown``), ``gateway.session_authorities.active_authority``,
the authority's ``logical_owner``, and ``gateway.slash_commands_session.discard_lost_turns``
(every ``unknown`` row of the session resolved, never re-run). The real ones are run by
``dev/hermes_contract.py``.
"""

from __future__ import annotations

import logging
import sys
import types

import pytest

from conftest import ProcessingOutcome
from test_adapter import SESSION, FakeRunner, FakeStore, build, golden, run, zero_windows
from uffici_for_hermes import adapter as adapter_module
from uffici_for_hermes import hermes_queue
from uffici_for_hermes.adapter import hermes_queue_notice

REQUEST = golden("brief")["requestId"]


async def spin(times: int = 50) -> None:
    """Let the adapter run: its queue reads go to a worker thread (as Hermes reads the table),
    so a fixed number of loop turns is not enough on a slow machine — real time passes too."""
    import asyncio

    for _ in range(times):
        await asyncio.sleep(0)
    await asyncio.sleep(0.05)
    for _ in range(times):
        await asyncio.sleep(0)


async def until(predicate, seconds: float = 5.0) -> None:
    """Wait for ``predicate`` to hold, bounded: the assertion that follows says what failed."""
    import asyncio

    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    while not predicate() and loop.time() < deadline:
        await asyncio.sleep(0.005)


CONVERSATION = golden("brief")["conversationId"]


class FakeAuthority:
    """One profile's session authority: its admissions table, and the discards it made."""

    def __init__(self) -> None:
        self.db = object()
        self.rows: dict[str, list[dict]] = {}
        self.discarded: list[str] = []

    def logical_owner(self, session_id: str) -> str:
        return session_id

    def add(self, session_id: str, request_id: str, status: str) -> dict:
        row = {"request_id": request_id, "status": status, "admission_id": f"adm_{request_id}"}
        self.rows.setdefault(session_id, []).append(row)
        return row


@pytest.fixture
def authority(monkeypatch) -> FakeAuthority:
    held = FakeAuthority()

    def list_session_admissions(db, *, session_id, pending_only=True):
        assert db is held.db
        rows = held.rows.get(session_id, [])
        if pending_only:
            rows = [r for r in rows if r["status"] in ("queued", "started", "unknown")]
        return list(rows)

    async def discard_lost_turns(authority, session_id):
        authority.discarded.append(session_id)
        for row in authority.rows.get(session_id, []):
            if row["status"] == "unknown":
                row["status"] = "terminal"

    runtime = types.ModuleType("hermes_state_runtime")
    runtime.list_session_admissions = list_session_admissions
    authorities = types.ModuleType("gateway.session_authorities")
    authorities.active_authority = lambda runner: getattr(runner, "session_authority", None)
    commands = types.ModuleType("gateway.slash_commands_session")
    commands.discard_lost_turns = discard_lost_turns
    monkeypatch.setitem(sys.modules, "hermes_state_runtime", runtime)
    monkeypatch.setitem(sys.modules, "gateway.session_authorities", authorities)
    monkeypatch.setitem(sys.modules, "gateway.slash_commands_session", commands)
    return held


def with_queue(authority: FakeAuthority):
    """An adapter on a seeded session whose runner carries Hermes's authority."""
    zero_windows()
    # A real recheck interval: the watch on Hermes's queue counts its checks against
    # BUSY_MAX_SECONDS, and a zero interval would exhaust them in a blink.
    adapter_module.BUSY_RECHECK_SECONDS = 0.005
    store = FakeStore({SESSION})
    adapter, wire = build(store, opener=golden("opener"))
    runner = FakeRunner()
    runner.session_authority = authority
    adapter.gateway_runner = runner
    adapter._active_sessions, adapter._pending_messages = {}, {}
    return adapter, wire, store.peek_session_id(SESSION)


def test_a_delivery_names_its_request_and_its_process() -> None:
    message_id = hermes_queue.delivery_id(REQUEST, "1a2b3c4d")
    assert message_id != REQUEST, "a redelivery would be a changed retry of a recorded id"
    assert hermes_queue.request_of(message_id) == REQUEST
    assert hermes_queue.request_of(REQUEST) == REQUEST
    assert hermes_queue.new_delivery_token() != hermes_queue.new_delivery_token()
    # A request id that itself carries the separator still names itself (#3124 review).
    assert hermes_queue.request_of("req~odd") == "req~odd"
    assert hermes_queue.request_of(hermes_queue.delivery_id("req~odd", "0a1b2c3d")) == "req~odd"


def test_without_hermess_queue_nothing_is_held() -> None:
    """An older Hermes has no queue: the read is None and the brief goes in as before."""
    adapter, _ = build(FakeStore({SESSION}))
    assert run(hermes_queue.read_queue(adapter, SESSION)) is None


def test_a_turn_hermes_lost_at_a_restart_is_discarded_before_the_brief(authority) -> None:
    """🔴 THE PAUSED CONVERSATION (#3124): a turn that had started when Hermes restarted is
    ``unknown`` and pauses its conversation until ``/discard``. The companion redelivers the
    request, so the lost turn is discarded through Hermes's own path, never re-run, and the
    redelivered brief is handed over."""

    async def scenario():
        adapter, _, session_id = with_queue(authority)
        authority.add(session_id, f"{REQUEST}~0a1d0a1d", "unknown")
        await adapter._on_frame(golden("brief"))
        await spin()
        return adapter, session_id

    adapter, session_id = run(scenario())
    assert authority.discarded == [session_id]
    assert len(adapter.handled) == 1, "the redelivered brief was not handed to Hermes"


def test_a_brief_hermes_still_holds_is_not_handed_over_twice(authority) -> None:
    """🔴 THE DOUBLE ANSWER (#3124): a brief still QUEUED in Hermes when it restarted is run
    by the new process itself. The companion's redelivery is not handed over again, and the
    request closes once Hermes's own run has answered it."""

    async def scenario():
        adapter, wire, session_id = with_queue(authority)
        row = authority.add(session_id, f"{REQUEST}~0a1d0a1d", "queued")
        await adapter._on_frame(golden("brief"))
        await spin()
        assert adapter.handled == [], "the brief Hermes still holds was handed over again"
        assert wire.of_type("done") == [], "closed before Hermes ran it"

        # Hermes's replay answers on the conversation, then settles the row.
        assert (await adapter.send(CONVERSATION, "the invoice is paid")).success
        row["status"] = "terminal"
        await until(lambda: wire.of_type("done"))
        return wire

    wire = run(scenario())
    assert [(f["requestId"], f["outcome"]) for f in wire.of_type("done")] == [(REQUEST, "answered")]


def test_a_brief_hermes_runs_while_busy_closes_without_its_hooks(authority) -> None:
    """🔴 #2246 AGAIN, ON HERMES 0.22+: a brief admitted while its session is busy is run by
    Hermes's queue (``dispatch_shared_busy``) with no processing hooks. The close scheduled
    at admission waits for Hermes's queue to run it, then closes with what it posted."""

    async def scenario():
        adapter, wire, session_id = with_queue(authority)
        await adapter._on_frame(golden("brief"))
        brief = adapter.handled[0]
        row = authority.add(session_id, brief.message_id, "queued")
        await spin()
        assert wire.of_type("done") == [], "closed while Hermes still owed the turn"

        assert (await adapter.send(CONVERSATION, "done, the card is moved")).success
        row["status"] = "terminal"
        await until(lambda: wire.of_type("done"))
        return wire

    wire = run(scenario())
    assert [(f["requestId"], f["outcome"]) for f in wire.of_type("done")] == [(REQUEST, "answered")]


def test_a_completion_hook_replaces_the_close_scheduled_at_admission(authority) -> None:
    async def scenario():
        adapter, wire, _ = with_queue(authority)
        await adapter._on_frame(golden("brief"))
        assert REQUEST in adapter._provisional
        await adapter.on_processing_complete(adapter.handled[0], ProcessingOutcome.SUCCESS)
        assert REQUEST not in adapter._provisional
        await until(lambda: wire.of_type("done"))
        return wire

    wire = run(scenario())
    assert [f["outcome"] for f in wire.of_type("done")] == ["empty"]


def test_a_disconnect_never_closes_a_turn_nothing_says_has_ended(authority) -> None:
    """A close scheduled at admission is no evidence the turn finished: a restart leaves it
    for the companion's redelivery, as it does a turn the restart cut (#2306)."""

    async def scenario():
        adapter, wire, session_id = with_queue(authority)
        await adapter._on_frame(golden("brief"))
        authority.add(session_id, adapter.handled[0].message_id, "started")
        assert (await adapter.send(CONVERSATION, "working on it")).success
        await adapter._close_finished_turns()
        return wire

    wire = run(scenario())
    assert wire.of_type("done") == []


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        (
            "⏳ Hermes is restarting — your message is saved and will be answered when it is back.",
            "saved",
        ),
        (
            "⏳ Your message is saved and will be answered when the work running in this "
            "conversation finishes.",
            "saved",
        ),
        (
            "⚠️ Hermes refused this message (invalid_params), so it was not saved or answered. "
            "Ask the operator to check gateway.log.",
            "unanswered",
        ),
        (
            "⏳ Too many messages are already waiting in this conversation; send this again after "
            "the current reply.",
            "unanswered",
        ),
        ("the invoice is paid", None),
    ],
)
def test_hermess_queue_notices_are_told_apart(text: str, kind: str | None) -> None:
    assert hermes_queue_notice(text) == kind


def test_a_saved_notice_is_not_posted_and_the_turn_is_still_owed(authority) -> None:
    async def scenario():
        adapter, wire, _ = with_queue(authority)
        await adapter._on_frame(golden("brief"))
        await adapter.send(
            CONVERSATION,
            "⏳ Your message is saved and will be answered when the work running in this "
            "conversation finishes.",
        )
        await adapter.send(CONVERSATION, "the answer")
        await adapter.on_processing_complete(adapter.handled[0], ProcessingOutcome.SUCCESS)
        await until(lambda: wire.of_type("done"))
        return wire

    wire = run(scenario())
    assert [f["text"] for f in wire.of_type("post")] == ["the answer"]
    assert [f["outcome"] for f in wire.of_type("done")] == ["answered"]


def test_a_turn_that_produced_only_a_refusal_closes_as_an_error(authority) -> None:
    """Hermes refused the brief: nothing was saved or answered, so the platform must not
    read the message as handled."""

    async def scenario():
        adapter, wire, _ = with_queue(authority)
        await adapter._on_frame(golden("brief"))
        await adapter.send(
            CONVERSATION,
            "⚠️ Hermes refused this message (invalid_params), so it was not saved or "
            "answered. Ask the operator to check gateway.log.",
        )
        await adapter.on_processing_complete(adapter.handled[0], ProcessingOutcome.SUCCESS)
        await until(lambda: wire.of_type("done"))
        return wire

    wire = run(scenario())
    assert wire.of_type("post") == []
    assert [f["outcome"] for f in wire.of_type("done")] == ["error"]


def test_a_queue_that_cannot_be_read_reads_as_nothing_held(authority, caplog) -> None:
    """Every read is of a Hermes internal: a store that raises costs today's behaviour, and
    says so, never the brief."""
    adapter, _, _ = with_queue(authority)
    sys.modules["hermes_state_runtime"].list_session_admissions = lambda *a, **k: 1 / 0
    assert run(hermes_queue.read_queue(adapter, SESSION)) is None
    assert "input queue could not be read (ZeroDivisionError)" in caplog.text


def test_no_authority_no_session_and_no_key_read_as_nothing_held(authority) -> None:
    adapter, _, _ = with_queue(authority)
    assert run(hermes_queue.read_queue(adapter, "")) is None
    assert run(hermes_queue.read_queue(adapter, "agent:main:uffici:dm:never_seen")) is None
    adapter.gateway_runner.session_authority = None
    assert run(hermes_queue.read_queue(adapter, SESSION)) is None


def test_a_lost_turn_that_cannot_be_discarded_still_lets_the_brief_go(authority, caplog) -> None:
    caplog.set_level(logging.INFO)

    async def failing(authority, session_id):
        raise RuntimeError("refused")

    sys.modules["gateway.slash_commands_session"].discard_lost_turns = failing

    async def scenario():
        adapter, _, session_id = with_queue(authority)
        authority.add(session_id, f"{REQUEST}~0a1d0a1d", "unknown")
        await adapter._on_frame(golden("brief"))
        await spin()
        return adapter

    adapter = run(scenario())
    assert len(adapter.handled) == 1
    assert "lost turn could not be discarded (RuntimeError)" in caplog.text
    assert "could not discard 1 turn(s)" in caplog.text


# ------------------------------------------------------------- review guards (#3124)


def test_a_close_waiting_on_hermess_queue_never_closes_without_evidence(authority) -> None:
    """🔴 H1: the close for a hookless turn waits for Hermes's queue to hold the delivery and
    let go of it. A queue that never shows it, or cannot be read, leaves the request owed —
    never `empty` 15 s into a turn whose answer would then be dropped."""

    async def scenario(readable: bool):
        adapter, wire, _ = with_queue(authority)
        if not readable:
            adapter.gateway_runner.session_authority = None
        await adapter._on_frame(golden("brief"))
        await spin(300)
        return adapter, wire

    for readable in (True, False):
        adapter, wire = run(scenario(readable))
        assert wire.of_type("done") == [], f"closed without evidence (readable={readable})"
        assert REQUEST in adapter._conversation_of, "the request was given up"


def test_a_completion_for_a_request_this_process_never_opened_closes_nothing(authority) -> None:
    """🔴 H2: after a restart Hermes may run a brief from its own queue before the companion
    hands the request over again. Its end must not answer the companion for that request."""

    async def scenario():
        adapter, wire, _ = with_queue(authority)
        replayed = types.SimpleNamespace(message_id=hermes_queue.delivery_id(REQUEST, "0a1d0a1d"))
        await adapter.on_processing_complete(replayed, ProcessingOutcome.SUCCESS)
        await spin(100)
        return wire

    assert run(scenario()).of_type("done") == []


def test_a_hookless_turn_hermes_finished_is_closed_by_a_restart(authority) -> None:
    """🔴 M1: once Hermes's queue has run a hookless brief, a draining restart inside its
    settle closes it like any finished turn (#2300) — or the companion redelivers it and the
    person is answered twice."""

    async def scenario():
        adapter, wire, session_id = with_queue(authority)
        adapter_module.DONE_SETTLE_SECONDS = 3600  # the restart lands inside the settle
        await adapter._on_frame(golden("brief"))
        row = authority.add(session_id, adapter.handled[0].message_id, "queued")
        await spin()
        assert (await adapter.send(CONVERSATION, "the card is moved")).success
        row["status"] = "terminal"
        await until(lambda: REQUEST not in adapter._provisional)
        assert REQUEST not in adapter._provisional, "Hermes ran it, so the turn has ended"
        await adapter._close_finished_turns()
        return wire

    try:
        wire = run(scenario())
    finally:
        zero_windows()
    assert [f["outcome"] for f in wire.of_type("done")] == ["answered"]


def test_a_redelivery_in_the_same_process_is_a_new_delivery(authority) -> None:
    """🔴 M3: the same request handed over twice by one process carries two names, so Hermes
    never refuses the second as a changed retry of the first."""

    async def scenario():
        adapter, _, _ = with_queue(authority)
        await adapter._on_frame(golden("brief"))
        adapter._forget_everything()
        await adapter._on_frame(golden("brief"))
        return [event.message_id for event in adapter.handled]

    first, second = run(scenario())
    assert first != second
    assert hermes_queue.request_of(first) == hermes_queue.request_of(second) == REQUEST


def test_a_cancel_while_the_queue_is_read_hands_nothing_over(authority) -> None:
    """🔴 M5: reading Hermes's queue (and discarding a lost turn) is a suspension point. A
    request the companion withdrew meanwhile is not handed to Hermes."""
    import asyncio

    release = asyncio.Event()

    async def slow_discard(authority, session_id):
        await release.wait()

    sys.modules["gateway.slash_commands_session"].discard_lost_turns = slow_discard

    async def scenario():
        adapter, _, session_id = with_queue(authority)
        authority.add(session_id, f"{REQUEST}~0a1d0a1d", "unknown")
        admitting = asyncio.ensure_future(adapter._on_frame(golden("brief")))
        await spin()
        await adapter._on_frame(golden("cancel-superseded"))
        release.set()
        await admitting
        await spin()
        return adapter

    handed = [event for event in run(scenario()).handled if event.text != "/stop"]
    assert handed == [], "the withdrawn brief was handed to Hermes"


def test_a_never_processed_reply_is_one_of_hermess_unanswered_notices() -> None:
    """One classifier for every reply of Hermes's that is not the agent's answer."""
    for sentence in adapter_module.HERMES_UNPROCESSED_REPLIES:
        assert hermes_queue_notice(sentence) == "unanswered"


def test_a_refusal_on_a_hookless_turn_closes_it_as_an_error_at_once(authority) -> None:
    """Hermes said it will not answer: the close stops waiting on its queue."""

    async def scenario():
        adapter, wire, session_id = with_queue(authority)
        await adapter._on_frame(golden("brief"))
        authority.add(session_id, adapter.handled[0].message_id, "queued")
        await spin()
        await adapter.send(
            CONVERSATION,
            "⏳ Too many messages are already waiting in this conversation; send this again "
            "after the current reply.",
        )
        await until(lambda: wire.of_type("done"))
        return wire

    assert [f["outcome"] for f in run(scenario()).of_type("done")] == ["error"]
