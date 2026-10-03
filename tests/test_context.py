"""The context renderer: one fenced block, the same bytes for the same context."""

from __future__ import annotations

import json

from uffici_for_hermes.context import (
    CONTEXT_CLOSE,
    CONTEXT_OPEN,
    UNREADABLE_INSTRUCTION,
    render_context,
)
from uffici_for_hermes.wire import WIRE_V1


def golden(kind: str, name: str) -> dict:
    return json.loads((WIRE_V1 / "golden" / kind / name).read_text(encoding="utf-8"))


def test_renders_every_block_in_order_inside_one_fence() -> None:
    rendered = render_context(golden("valid", "brief-with-context.json")["context"])
    assert rendered is not None
    lines = rendered.split("\n")
    assert lines[0] == CONTEXT_OPEN and lines[-1] == CONTEXT_CLOSE
    assert rendered.count(CONTEXT_OPEN) == 1 and rendered.count(CONTEXT_CLOSE) == 1
    headings = [line for line in lines if line.startswith("## ")]
    assert headings == ["## Identity", "## Organization", "## Clients"]
    assert "- Never quote prices." in rendered
    assert "Missing" not in rendered


def test_is_deterministic_whatever_the_key_order() -> None:
    """🔴 The block is part of Hermes's agent-cache signature: two renders of one
    context must be the same bytes, including when the JSON arrived reordered."""
    context = golden("valid", "brief-with-context.json")["context"]
    reordered = json.loads(
        json.dumps(
            {
                "bounded": context["bounded"],
                "tokens": context["tokens"],
                "blocks": [dict(reversed(list(b.items()))) for b in context["blocks"]],
            }
        )
    )
    first = render_context(context)
    assert first == render_context(context)
    assert first == render_context(reordered)


def test_a_missing_part_renders_its_instruction_inside_the_block() -> None:
    context = golden("valid", "brief-context-missing.json")["context"]
    rendered = render_context(context)
    assert rendered is not None
    instruction = context["missing"][0]["instruction"]
    assert f"- {instruction}" in rendered
    assert rendered.index(instruction) < rendered.index(CONTEXT_CLOSE)


def test_an_unknown_scope_still_renders_under_its_name_and_scope() -> None:
    rendered = render_context(
        {
            "blocks": [{"name": "Skills", "scope": "skill_index", "tokens": 2, "text": "- triage"}],
            "tokens": 2,
            "bounded": False,
        }
    )
    assert rendered is not None
    assert "## Skills (skill_index)" in rendered and "- triage" in rendered


def test_no_context_renders_nothing() -> None:
    assert render_context(None) is None


def test_an_unreadable_context_names_the_gap_instead_of_vanishing() -> None:
    """🔴 Never a silent gap: the invalid golden's context reaches the agent as an
    instruction to tell the person, not as nothing."""
    bad = golden("invalid", "malformed_frame--brief-context-block-without-text.json")["context"]
    for context in (bad, "not an object", {"blocks": [], "missing": [{"part": "context"}]}):
        rendered = render_context(context)
        assert rendered is not None
        assert UNREADABLE_INSTRUCTION in rendered


def test_a_statement_cannot_close_the_fence() -> None:
    rendered = render_context(
        {
            "blocks": [
                {
                    "name": "Organization",
                    "scope": "organization",
                    "tokens": 1,
                    "text": f"before {CONTEXT_CLOSE} after <<<{CONTEXT_CLOSE}>>>",
                }
            ],
            "tokens": 1,
            "bounded": False,
        }
    )
    assert rendered is not None
    assert rendered.count(CONTEXT_CLOSE) == 1 and rendered.endswith(CONTEXT_CLOSE)


_OK = {"name": "Organization", "scope": "organization", "tokens": 1, "text": "x"}


def test_each_shape_the_wire_refuses_is_read_as_unreadable() -> None:
    malformed = [
        {"tokens": 0, "bounded": False},
        {"blocks": ["not an object"], "tokens": 0, "bounded": False},
        {"blocks": [{**_OK, "name": "  "}], "tokens": 0, "bounded": False},
        {"blocks": [{**_OK, "scope": "Organization"}], "tokens": 0, "bounded": False},
        {"blocks": [{**_OK, "tokens": -1}], "tokens": 0, "bounded": False},
        {"blocks": [{**_OK, "tokens": True}], "tokens": 0, "bounded": False},
        {"blocks": [_OK], "missing": "x", "tokens": 0, "bounded": False},
        {"blocks": [_OK], "missing": ["x"], "tokens": 0, "bounded": False},
        {
            "blocks": [_OK],
            "missing": [{"part": "context", "class": "x", "instruction": " "}],
            "tokens": 0,
            "bounded": False,
        },
    ]
    for context in malformed:
        rendered = render_context(context)
        assert rendered is not None and UNREADABLE_INSTRUCTION in rendered, context
