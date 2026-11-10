"""Hermes 0.22+'s durable input queue, as the plugin reconciles with it (#3124).

The stand-ins below are TRANSCRIBED from Hermes 0.23.0 (commit 2bd378ed):
``hermes_state_runtime.list_session_admissions`` (rows of ``session_admissions``, pending
= ``queued`` / ``started`` / ``unknown``), ``gateway.session_authorities.active_authority``,
the authority's ``logical_owner``, and ``gateway.slash_commands_session.discard_lost_turns``
(every ``unknown`` row of the session resolved, never re-run), the authority's
``cancel_queued(actor, ref, admission_id)`` (a ``queued`` row becomes terminal, a ``started``
one is refused ``stale_generation``) with ``gateway.session_contract``'s ``SessionActor`` and
``SessionRef``, and each row's ``owner_epoch`` (the process that admitted it). The real ones
are run by ``dev/hermes_contract.py``.
"""

from __future__ import annotations

import dataclasses
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


@dataclasses.dataclass(frozen=True)
class SessionActor:
    subject: str
    profile_id: str
    capabilities: frozenset
    transport_id: str


@dataclasses.dataclass(frozen=True)
class SessionRef:
    profile_id: str
    session_id: str


class RuntimeStoreError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class FakeAuthority:
    """One profile's session authority: its admissions table, and the discards and cancels it
    made. ``epoch`` is this process's; a row admitted by an earlier one carries another."""

    profile_id = "/opt/data"
    epoch = 7

    def __init__(self) -> None:
        self.db = object()
        self.rows: dict[str, list[dict]] = {}
        self.discarded: list[str] = []
        self.cancelled: list[tuple[SessionActor, SessionRef, str]] = []

    def logical_owner(self, session_id: str) -> str:
        return session_id

    def add(self, session_id: str, request_id: str, status: str, *, epoch: int = 7) -> dict:
        row = {
            "request_id": request_id,
            "status": status,
            "admission_id": f"adm_{request_id}",
            "owner_epoch": epoch,
        }
        self.rows.setdefault(session_id, []).append(row)
        return row

    async def cancel_queued(self, actor, ref, admission_id):
        rows = self.rows.get(ref.session_id, [])
        row = next(r for r in rows if r["admission_id"] == admission_id)
        if row["status"] != "queued":
            raise RuntimeStoreError("stale_generation")
        row["status"] = "terminal"
        self.cancelled.append((actor, ref, admission_id))


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
    contract = types.ModuleType("gateway.session_contract")
    contract.SessionActor, contract.SessionRef = SessionActor, SessionRef
    monkeypatch.setitem(sys.modules, "gateway.session_contract", contract)
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


# ------------------------------------------------------------- the three leftovers (#3125)


def _uffici_sessions(adapter, *session_ids: str) -> None:
    """Hermes's session store lists its entries with their origin, as its recovery reads them."""
    origin = types.SimpleNamespace(platform=types.SimpleNamespace(value="uffici"))
    other = types.SimpleNamespace(platform=types.SimpleNamespace(value="telegram"))
    adapter._session_store.list_sessions = lambda: [
        *(types.SimpleNamespace(session_id=sid, origin=origin) for sid in session_ids),
        types.SimpleNamespace(session_id="session_telegram", origin=other),
    ]


def test_a_withdrawn_brief_hermes_still_queues_is_cancelled_in_its_queue(authority) -> None:
    """🔴 #3125 (1): a cancel stopped only a RUNNING turn; a brief Hermes had queued and not
    started ran later and answered whatever request was open then. It is cancelled through
    Hermes's own ``cancel_queued``, as the actor Hermes's messaging ``/stop`` uses."""

    async def scenario():
        adapter, _, session_id = with_queue(authority)
        await adapter._on_frame(golden("brief"))
        row = authority.add(session_id, adapter.handled[0].message_id, "queued")
        await spin()
        await adapter._on_frame(golden("cancel-superseded"))
        await spin()
        return row, session_id

    row, session_id = run(scenario())
    assert row["status"] == "terminal", "the queued brief was left to run"
    [(actor, ref, admission_id)] = authority.cancelled
    assert admission_id == row["admission_id"] and ref.session_id == session_id
    assert actor.subject == hermes_queue.CONTROL_SUBJECT == "messaging-control"
    assert actor.capabilities == frozenset({"session:submit", "session:control"})


def test_a_reply_from_a_withdrawn_brief_never_reaches_a_later_request(authority) -> None:
    """🔴 #3125 (1): whatever a withdrawn brief still produces — Hermes kept the row, a lost
    turn paused the conversation — is dropped, never posted to the request open now, and its
    end closes nothing."""

    async def scenario():
        adapter, wire, _ = with_queue(authority)
        await adapter._on_frame(golden("brief"))
        withdrawn = adapter.handled[0]
        await adapter._on_frame(golden("cancel-superseded"))
        await spin()
        later = {**golden("brief"), "requestId": "req_later"}
        await adapter._on_frame(later)
        await spin()
        late = await adapter.send(CONVERSATION, "the old answer", reply_to=withdrawn.message_id)
        await adapter.on_processing_complete(withdrawn, ProcessingOutcome.SUCCESS)
        own = await adapter.send(
            CONVERSATION, "the new answer", reply_to=adapter.handled[-1].message_id
        )
        await spin(100)
        return wire, late, own

    wire, late, own = run(scenario())
    assert late.success is False and own.success is True
    assert [(f["requestId"], f["text"]) for f in wire.of_type("post")] == [
        ("req_later", "the new answer")
    ], "a withdrawn brief's reply was posted to a later request"
    assert [f for f in wire.of_type("done") if f["requestId"] == "req_later"] == []


@pytest.mark.parametrize("redelivers", [True, False])
def test_a_brief_queued_before_the_restart_is_withdrawn_before_hermes_replays_it(
    authority, redelivers: bool
) -> None:
    """🔴 #3125 (2): Hermes's startup recovery replays a brief still ``queued`` from the old
    process, and a replay that beat the companion's redelivery answered nobody — then the
    redelivered brief ran the model again. Once the companion says it redelivers, every such
    row is cancelled through Hermes's own path; this process's own rows are kept, and so is
    every row while the companion does not redeliver (Hermes's replay is all that answers)."""

    async def scenario():
        adapter, _, session_id = with_queue(authority)
        _uffici_sessions(adapter, session_id)
        replay = authority.add(session_id, f"{REQUEST}~0a1d0a1d", "queued", epoch=6)
        mine = authority.add(session_id, "req_mine~1b2c3d4e", "queued")
        foreign = authority.add(session_id, "a-hermes-wake", "queued", epoch=6)
        features = ["redelivery"] if redelivers else []
        await adapter._on_companion_hello({"v": 1, "type": "hello", "features": features})
        await adapter._on_frame(golden("brief"))
        await spin()
        return adapter, replay, mine, foreign

    adapter, replay, mine, foreign = run(scenario())
    assert mine["status"] == "queued" and foreign["status"] == "queued"
    if redelivers:
        assert replay["status"] == "terminal", "Hermes was left to replay the old brief"
        assert len(adapter.handled) == 1, "the redelivered brief must run, exactly once"
    else:
        assert replay["status"] == "queued" and authority.cancelled == []
        assert adapter.handled == [], "a brief Hermes still holds was handed over twice"


def test_a_reset_brief_sees_what_hermes_holds_on_the_session_it_would_leave(authority) -> None:
    """🔴 #3125 (3): a ``resetSession`` brief reset Hermes's session BEFORE reading its queue,
    so a delivery of the same request held on the old session was invisible and the brief
    was handed over beside it — two answers. The queue is read first, and the session kept."""

    async def scenario():
        adapter, wire, session_id = with_queue(authority)
        row = authority.add(session_id, f"{REQUEST}~0a1d0a1d", "started")
        await adapter._on_frame({**golden("brief"), "resetSession": True})
        await spin()
        assert adapter.handled == [], "handed over beside the delivery Hermes holds"
        assert adapter._session_store.resets == [], "the session Hermes holds it on was left"
        assert (await adapter.send(CONVERSATION, "the card is moved")).success
        row["status"] = "terminal"
        await until(lambda: wire.of_type("done"))
        return wire

    wire = run(scenario())
    assert [(f["requestId"], f["outcome"]) for f in wire.of_type("done")] == [(REQUEST, "answered")]


def test_a_reset_brief_whose_held_delivery_ran_meanwhile_is_not_handed_over(authority) -> None:
    """The delivery held when the reset brief arrived finished before the reconcile read the
    queue again: Hermes ran it, so the request closes and nothing is handed over."""

    async def scenario():
        adapter, wire, session_id = with_queue(authority)
        row = authority.add(session_id, f"{REQUEST}~0a1d0a1d", "started")
        adapter_module.DONE_SETTLE_SECONDS = 0
        original = adapter._reconcile_with_hermes_queue

        async def finished_first(event, request_id):
            row["status"] = "terminal"
            return await original(event, request_id)

        adapter._reconcile_with_hermes_queue = finished_first
        await adapter._on_frame({**golden("brief"), "resetSession": True})
        await until(lambda: wire.of_type("done"))
        return adapter, wire

    adapter, wire = run(scenario())
    assert adapter.handled == []
    assert [f["requestId"] for f in wire.of_type("done")] == [REQUEST]


def test_the_queue_is_never_cancelled_on_without_hermess_session_contract(authority) -> None:
    """A Hermes without ``gateway.session_contract`` cancels nothing, and says so."""
    sys.modules.pop("gateway.session_contract")
    queue = hermes_queue.HeldQueue(authority=authority, session_id="s", rows=[])
    queue.rows.append(authority.add("s", "req~0a1d0a1d", "queued"))
    assert run(queue.cancel({"req~0a1d0a1d"})) == set()
    assert queue.rows[0]["status"] == "queued"


def test_a_row_hermes_claimed_meanwhile_is_kept_and_reported_by_its_reason(
    authority, caplog
) -> None:
    caplog.set_level(logging.INFO)
    queue = hermes_queue.HeldQueue(authority=authority, session_id="s", rows=[])
    row = authority.add("s", "req~0a1d0a1d", "queued")
    queue.rows.append(dict(row))
    row["status"] = "started"  # Hermes's drain claimed it between the read and the cancel
    assert run(queue.cancel({"req~0a1d0a1d"})) == set()
    assert "Hermes kept a queued brief (stale_generation)" in caplog.text


def test_without_an_epoch_no_row_is_called_a_replay(authority) -> None:
    queue = hermes_queue.HeldQueue(authority=types.SimpleNamespace(), session_id="s", rows=[])
    queue.rows.append({"request_id": "req~0a1d0a1d", "status": "queued", "owner_epoch": 1})
    assert queue.replayed() == set()


def test_platform_queues_that_cannot_be_read_are_not_known(authority, caplog) -> None:
    """None, never []: "nothing queued" lets the process skip the withdrawal for good."""
    adapter, _, _ = with_queue(authority)
    assert run(hermes_queue.read_platform_queues(adapter, "uffici")) is None
    assert "input queues could not be read (AttributeError)" in caplog.text
    adapter.gateway_runner.session_authority = None
    assert run(hermes_queue.read_platform_queues(adapter, "uffici")) is None
    # A Hermes with no durable queue has nothing queued: known, and empty.
    sys.modules.pop("hermes_state_runtime")
    assert run(hermes_queue.read_platform_queues(adapter, "uffici")) == []


def test_withdrawn_deliveries_are_remembered_bounded_oldest_first() -> None:
    adapter, _ = build(FakeStore({SESSION}))
    for index in range(adapter_module.CLOSED_MEMORY + 3):
        adapter._withdraw({hermes_queue.delivery_id(f"req_{index}", "0a1d0a1d")})
    assert len(adapter._withdrawn) == adapter_module.CLOSED_MEMORY
    assert hermes_queue.delivery_id("req_0", "0a1d0a1d") not in adapter._withdrawn
    assert hermes_queue.delivery_id(f"req_{adapter_module.CLOSED_MEMORY + 2}", "0a1d0a1d") in (
        adapter._withdrawn
    )


# ------------------------------------------------------------- review guards (#3125)


class HelloOnCue:
    """The companion's end of one wire connection: its hello arrives when ``cue`` is set."""

    def __init__(self, features: list[str]) -> None:
        import asyncio

        self.cue = asyncio.Event()
        self.features = features
        self.sent: list[str] = []

    async def send(self, raw: str) -> None:
        self.sent.append(raw)

    async def close(self) -> None:
        return None

    def __aiter__(self):
        import asyncio
        import json

        async def frames():
            await self.cue.wait()
            yield json.dumps({"v": 1, "type": "hello", "features": self.features})
            await asyncio.Event().wait()  # the connection stays up

        return frames()


def connecting_adapter(authority, features: list[str]):
    """An adapter whose ``connect`` dials a companion that answers on cue."""
    adapter, _, session_id = with_queue(authority)
    adapter._get_env = {
        "UFFICI_WIRE_TOKEN": "t" * 40,
        "UFFICI_COMPANION_URL": "http://127.0.0.1:7411",
        "UFFICI_CONTEXT_WINDOW_TOKENS": "100000",
    }.get
    connection = HelloOnCue(features)

    async def connect_fn(_address):
        return connection

    adapter._connect_fn = connect_fn
    adapter._log_handler = None  # no log push for the disconnect to finish
    _uffici_sessions(adapter, session_id)
    return adapter, connection, session_id


def test_connect_returns_only_once_the_old_briefs_are_withdrawn(authority) -> None:
    """🔴 #3125 review: Hermes's startup recovery replays queued rows once every adapter's
    connect has returned. Withdrawn on the hello's own task alone, the replay could claim a
    row first — ``cancel_queued`` refused ``stale_generation`` — and the model ran twice. The
    connect now waits for the hello and the withdrawal, so recovery finds nothing to run."""
    import asyncio

    async def scenario():
        adapter, connection, session_id = connecting_adapter(authority, ["redelivery"])
        row = authority.add(session_id, f"{REQUEST}~0a1d0a1d", "queued", epoch=6)
        connecting = asyncio.ensure_future(adapter.connect())
        await spin()
        assert not connecting.done(), "connected (so recovery may run) before the hello"
        connection.cue.set()
        assert await asyncio.wait_for(connecting, 5) is True
        # Hermes's recovery, now: it claims the head of the queue if anything is queued.
        claimable = [r for r in authority.rows[session_id] if r["status"] == "queued"]
        await adapter.disconnect()
        return row, claimable

    row, claimable = run(scenario())
    assert row["status"] == "terminal" and claimable == [], "recovery would replay the old brief"


def test_a_hello_that_never_comes_holds_the_connect_only_as_long_as_its_wait(authority) -> None:
    """Bounded: no hello leaves Hermes's replay to answer, and the connect still returns."""
    import asyncio

    adapter_module.HELLO_WAIT_SECONDS = 0.05

    async def scenario():
        adapter, _, session_id = connecting_adapter(authority, ["redelivery"])
        row = authority.add(session_id, f"{REQUEST}~0a1d0a1d", "queued", epoch=6)
        connected = await asyncio.wait_for(adapter.connect(), 5)
        await adapter.disconnect()
        return connected, row

    connected, row = run(scenario())
    assert connected is True and row["status"] == "queued"


def test_nothing_to_withdraw_costs_the_connect_no_wait(authority) -> None:
    import asyncio

    adapter_module.HELLO_WAIT_SECONDS = 3600

    async def scenario():
        adapter, _, session_id = connecting_adapter(authority, ["redelivery"])
        authority.add(session_id, "req_mine~1b2c3d4e", "queued")  # this process's own
        connected = await asyncio.wait_for(adapter.connect(), 5)
        await adapter.disconnect()
        return connected

    assert run(scenario()) is True
    assert adapter_module._PROCESS["replays_withdrawn"] is True


def test_the_withdrawal_runs_once_per_process(authority, monkeypatch) -> None:
    """🔴 #3125 review: a previous process's rows exist only until the first withdrawal, so
    a reconnect's hello neither reads every session's queue again nor makes every brief's
    reconcile wait on it."""
    reads: list[str] = []
    real = hermes_queue.read_platform_queues

    async def counted(adapter, platform):
        reads.append(platform)
        return await real(adapter, platform)

    monkeypatch.setattr(hermes_queue, "read_platform_queues", counted)

    async def scenario():
        adapter, _, session_id = with_queue(authority)
        _uffici_sessions(adapter, session_id)
        hello = {"v": 1, "type": "hello", "features": ["redelivery"]}
        await adapter._on_companion_hello(hello)
        await adapter._replays_withdrawn()
        late = authority.add(session_id, f"{REQUEST}~0a1d0a1d", "queued", epoch=6)
        await adapter._on_companion_hello(hello)  # a reconnect
        await adapter._replays_withdrawn()
        return late

    late = run(scenario())
    assert reads == ["uffici"], "the withdrawal ran again on a reconnect"
    assert late["status"] == "queued"


def test_a_withdrawn_brief_that_asks_puts_no_card_on_a_later_request(authority) -> None:
    """🔴 #3125 review: Hermes's clarify names a session, not a brief. A withdrawn brief that
    runs anyway and asks would put its card on the request open now and close it
    ``answered``; the running row of Hermes's queue says whose question it is."""

    async def scenario():
        adapter, wire, session_id = with_queue(authority)
        adapter._companion_features = frozenset({"question"})
        await adapter._on_frame(golden("brief"))
        withdrawn = adapter.handled[0]
        authority.add(session_id, withdrawn.message_id, "started")  # Hermes runs it regardless
        await adapter._on_frame(golden("cancel-superseded"))
        await adapter._on_frame({**golden("brief"), "requestId": "req_later"})
        await spin()
        result = await adapter.send_clarify(
            CONVERSATION, "Which board?", ["Uffici board", "Trello"], "clarify_1", SESSION
        )
        await spin()
        return wire, result

    wire, result = run(scenario())
    assert result.success is True, "a failure would send Hermes to its text fallback"
    assert [f for f in wire.of_type("post") if f.get("question")] == []
    assert [f for f in wire.of_type("done") if f["requestId"] == "req_later"] == []


def test_withdrawing_an_answer_withdraws_the_turn_that_asked(authority) -> None:
    """🔴 #3125 review: a turn that asked carries on under the request that brought the
    answer, still anchoring its replies to its FIRST brief. Cancelling that request must
    withdraw the first brief too, or the turn's replies reach whatever request opens next."""
    from test_adapter import answer_brief, asked

    async def scenario():
        adapter, wire, brief, _, _ = await asked()
        first = adapter.handled[0]
        await adapter._on_frame(answer_brief("Trello"))
        await adapter._on_frame(
            {"v": 1, "type": "cancel", "requestId": "req_answer", "reason": "superseded"}
        )
        await adapter._on_frame({**golden("brief"), "requestId": "req_later"})
        await adapter._on_frame({**golden("opener"), "requestId": "req_later"})
        late = await adapter.send(
            brief["conversationId"], "the old turn", reply_to=first.message_id
        )
        await spin()
        return wire, late

    wire, late = run(scenario())
    assert late.success is False
    assert [f["text"] for f in wire.of_type("post") if f["requestId"] == "req_later"] == []


def test_an_unconfigured_platform_says_why_once(monkeypatch, caplog) -> None:
    """#3119 review: Hermes skips a platform that is not connected; a silent False left no
    line saying why. Said once, naming the variables and no value."""
    from conftest import PlatformConfig

    for name in ("UFFICI_WIRE_SOCKET", "UFFICI_COMPANION_URL", "UFFICI_WIRE_TOKEN_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("UFFICI_WIRE_TOKEN", "t" * 40)
    assert adapter_module.is_connected(PlatformConfig(enabled=True)) is False
    assert adapter_module.is_connected(PlatformConfig(enabled=True)) is False
    lines = [r.getMessage() for r in caplog.records if "not connecting" in r.getMessage()]
    assert len(lines) == 1 and "UFFICI_WIRE_SOCKET" in lines[0]
    assert "t" * 40 not in caplog.text


def test_an_unreadable_queue_at_connect_still_waits_for_the_withdrawal(authority) -> None:
    """🔴 #3125 review: a queue that cannot be read during the cold connect is "not known",
    never "nothing": the connect still waits for the hello and its withdrawal — which reads
    the queue again, wired by then — or Hermes's recovery races it."""
    import asyncio

    async def scenario():
        adapter, connection, session_id = connecting_adapter(authority, ["redelivery"])
        row = authority.add(session_id, f"{REQUEST}~0a1d0a1d", "queued", epoch=6)
        runner = adapter.gateway_runner
        runner.session_authority = None  # not wired yet when the connect starts
        connecting = asyncio.ensure_future(adapter.connect())
        await spin()
        assert not connecting.done(), "connected (so recovery may run) without knowing"
        runner.session_authority = authority
        connection.cue.set()
        assert await asyncio.wait_for(connecting, 5) is True
        await adapter.disconnect()
        return row

    assert run(scenario())["status"] == "terminal"


def test_the_queue_enumeration_runs_off_the_event_loop(authority) -> None:
    """🔴 #3125 review: the session store's listing and the owner lookups run in a worker
    thread with the reads, never on the gateway's loop."""
    import threading

    seen: list[bool] = []

    async def scenario():
        adapter, _, session_id = with_queue(authority)
        _uffici_sessions(adapter, session_id)
        listing = adapter._session_store.list_sessions

        def recorded():
            seen.append(threading.current_thread() is threading.main_thread())
            return listing()

        adapter._session_store.list_sessions = recorded
        return await hermes_queue.read_platform_queues(adapter, "uffici")

    queues = run(scenario())
    assert queues is not None and seen == [False], "the listing ran on the event loop"
