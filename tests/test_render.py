"""The one rendering rule, over the goldens the wire actually carries.

🔴 Driven from ``wire/v1/golden/`` wherever it can be, not from hand-written
dicts: a hand-written entry is this test's opinion of the wire, and the two
drift the moment the contract changes. The goldens are the contract.
"""

import json

import pytest

from uffici_for_hermes.render import (
    TOMBSTONE_LINE,
    cause_author,
    render_entries,
    render_entry,
    render_reason,
)
from uffici_for_hermes.wire import WIRE_V1


def golden(name: str) -> dict:
    return json.loads((WIRE_V1 / "golden" / "valid" / f"{name}.json").read_text())


def test_a_person_is_labelled_without_a_handle() -> None:
    entry = golden("brief")["delta"][0]
    assert render_entry(entry) == ["Ana Marques: Can you check the Lisbon invoice?"]


def test_an_agent_carries_its_handle_so_the_model_can_address_it() -> None:
    # 🔴 The handle is how the platform's wake rule is triggered. Without it in
    # the text, a model cannot hand work to another agent even when it should.
    agent_entry = next(e for e in golden("opener")["entries"] if e["author"]["type"] == "agent")
    assert render_entry(agent_entry) == ["Smith (@smith): I will take a look."]


def test_a_tool_line_is_verbatim_and_unattributed() -> None:
    entry = {
        "id": "e9",
        "cursor": "c9",
        "at": "2026-09-17T14:01:00Z",
        "kind": "tool_line",
        "text": "queried a table (3 rows)",
        "author": {"type": "agent", "id": "a1", "label": "Smith", "self": True},
    }
    assert render_entry(entry) == ["queried a table (3 rows)"], (
        "a tool line is the platform's own rendering of what an agent did; "
        "prefixing it with an author attributes the tool's output to someone"
    )


def test_a_tombstone_says_something_was_withdrawn() -> None:
    entry = {
        "id": "e9",
        "cursor": "c9",
        "at": "2026-09-17T14:01:00Z",
        "kind": "tombstone",
        "text": "",
        "author": {"type": "member", "id": "m1", "label": "Ana Marques", "self": False},
    }
    assert render_entry(entry) == [f"Ana Marques: {TOMBSTONE_LINE}"]


def test_an_unknown_kind_is_labelled_rather_than_dropped() -> None:
    """🔴 Decision 44 and the wire's forward-compatibility requirement.

    Wave 4's ``summary`` kind must arrive with no version bump. A reader that
    dropped an unknown kind would lose it silently; one that raised would make
    that arrival a breaking change.
    """
    entry = {
        "id": "e9",
        "cursor": "c9",
        "at": "2026-09-17T14:01:00Z",
        "kind": "summary",
        "text": "Earlier: the invoice was approved.",
        "author": {"type": "system", "id": "sys", "label": "Uffici", "self": False},
    }
    assert render_entry(entry) == ["Uffici [summary]: Earlier: the invoice was approved."]


def test_an_attachment_is_a_reference_line_never_its_content() -> None:
    entry = {
        "id": "e9",
        "cursor": "c9",
        "at": "2026-09-17T14:01:00Z",
        "kind": "message",
        "text": "Here it is.",
        "author": {"type": "member", "id": "m1", "label": "Ana Marques", "self": False},
        "attachments": [{"id": "f1", "name": "invoice.pdf", "ref": "files/invoice.pdf"}],
    }
    assert render_entry(entry) == [
        "Ana Marques: Here it is.",
        "  [attachment] invoice.pdf — files/invoice.pdf",
    ]


def test_a_timestamp_appears_only_on_a_gap() -> None:
    def at(minute: int) -> dict:
        return {
            "id": f"e{minute}",
            "cursor": f"c{minute}",
            "at": f"2026-09-17T14:{minute:02d}:00Z",
            "kind": "message",
            "text": f"line {minute}",
            "author": {"type": "member", "id": "m1", "label": "Ana", "self": False},
        }

    close = render_entries([at(1), at(2)])
    assert "[" not in close, f"a two-minute gap earned a timestamp line: {close!r}"

    far = render_entries([at(1), at(31)])
    assert "2026-09-17T14:31:00+00:00" in far, f"a thirty-minute gap earned none: {far!r}"


def test_an_unreadable_timestamp_costs_the_gap_line_never_the_entry() -> None:
    entries = [
        {
            "id": "e1",
            "cursor": "c1",
            "at": "not a date",
            "kind": "message",
            "text": "still here",
            "author": {"type": "member", "id": "m1", "label": "Ana", "self": False},
        },
    ]
    assert render_entries(entries) == "Ana: still here"


@pytest.mark.parametrize("payload", [None, "not a dict", 7])
def test_a_non_entry_in_the_list_is_skipped(payload: object) -> None:
    assert render_entries([payload]) == ""


def test_the_reason_line_names_who_caused_the_turn() -> None:
    brief = golden("brief")
    line = render_reason(brief["reason"], brief["cause"], brief["delta"])
    assert line == "[Uffici] You were woken because: mentioned (by Ana Marques)."


def test_the_reason_line_survives_a_cause_outside_the_delta() -> None:
    line = render_reason("mentioned", {"entryId": "gone", "authorId": "m1"}, [])
    assert line == "[Uffici] You were woken because: mentioned."


@pytest.mark.parametrize(
    ("reason", "sentence"),
    [
        ("card_arrived", "the card arrived in your lane"),
        ("mention", "you were mentioned"),
        ("default_agent", "you are this conversation's default agent"),
        ("handback", "a person handed the conversation back to the agents"),
        ("reopen", "a person answered your question or decided an approval"),
        ("wake_flag", "the platform asked you to answer"),
    ],
)
def test_a_known_reason_reads_as_a_sentence(reason: str, sentence: str) -> None:
    assert render_reason(reason, None, []) == f"[Uffici] You were woken because: {sentence}."


def test_an_unknown_reason_is_still_rendered_verbatim() -> None:
    line = render_reason("a_newer_reason", None, [])
    assert line == "[Uffici] You were woken because: a_newer_reason."


def test_a_platform_cause_names_nobody() -> None:
    """A card's arrival is a note the platform wrote: never "(by system)"."""
    note = {"id": "7", "kind": "note", "text": "Ana moved the card.", "author": {"type": "system"}}
    line = render_reason("card_arrived", {"entryId": "7"}, [note])
    assert line == "[Uffici] You were woken because: the card arrived in your lane."


def test_a_missing_reason_still_produces_a_line() -> None:
    assert render_reason("", None, []) == "[Uffici] You were woken because: unknown."


def test_the_sender_is_the_causes_author_not_the_last_human() -> None:
    """🔴 The acceptance criterion: a brief caused by ANOTHER AGENT names that
    agent as sender.

    Hermes keys authorization and per-message prompt building off the sender. A
    plugin that reported "the last human" would attribute an agent's hand-off to
    a person who said nothing this turn.
    """
    entries = [
        {
            "id": "e1",
            "cursor": "c1",
            "at": "2026-09-17T14:01:00Z",
            "kind": "message",
            "text": "earlier",
            "author": {"type": "member", "id": "m1", "label": "Ana", "self": False},
        },
        {
            "id": "e2",
            "cursor": "c2",
            "at": "2026-09-17T14:02:00Z",
            "kind": "message",
            "text": "@smith take this",
            "author": {
                "type": "agent",
                "id": "a7",
                "label": "Archer",
                "handle": "@archer",
                "self": False,
            },
        },
    ]
    author = cause_author({"entryId": "e2", "authorId": "a7"}, entries)
    assert author["id"] == "a7" and author["label"] == "Archer"


def test_a_cause_outside_the_delta_still_yields_its_author_id() -> None:
    author = cause_author({"entryId": "gone", "authorId": "m9"}, [])
    assert author["id"] == "m9"
