"""The deterministic model, tested directly.

🔴 It speaks a protocol that took three misleading runs to get right — SSE with
a role chunk, a content chunk, a `finish_reason` and `[DONE]` — and until now
nothing but the end-to-end job exercised it. A mock that cannot speak the
protocol it stands in for produces failures that look like defects in the code
under test: the first cut answered plain JSON, Hermes reported
`EmptyStreamError`, retried three times, and a turn completed before its own
reply existed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

MOCK = Path(__file__).resolve().parents[1] / "dev" / "model_mock" / "server.py"
PORT = 18131
BASE = f"http://127.0.0.1:{PORT}"


@pytest.fixture(scope="module")
def mock_server() -> Iterator[None]:
    process = subprocess.Popen(  # noqa: S603 - our own file, fixed argv
        [sys.executable, str(MOCK)],
        env={
            "MODEL_MOCK_PORT": str(PORT),
            "PATH": "/usr/bin:/bin",
            "MODEL_MOCK_SLOW_SECONDS": "1.5",
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(100):
        try:
            urllib.request.urlopen(f"{BASE}/healthz", timeout=0.5).read()
            break
        except (urllib.error.URLError, OSError):
            time.sleep(0.1)
    else:  # pragma: no cover - only if the mock cannot start at all
        process.kill()
        pytest.fail("the model mock never came up")
    yield
    process.terminate()
    process.wait(timeout=5)


#: What Hermes offers on a turn: its tools, clarify among them.
CLARIFY_TOOL = [{"type": "function", "function": {"name": "clarify", "parameters": {}}}]


def ask(messages: list[dict], *, stream: bool, tools: list[dict] | None = None) -> str:
    body = json.dumps(
        {
            "model": "mock-model",
            "stream": stream,
            "messages": messages,
            "tools": CLARIFY_TOOL if tools is None else tools,
        }
    ).encode()
    request = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(request, timeout=5).read().decode()


def test_a_streamed_request_gets_the_three_chunks_a_consumer_needs(mock_server: None) -> None:
    """🔴 Role, content, finish_reason, then [DONE].

    Without the finish_reason Hermes reports "an empty stream with no
    finish_reason" and retries — which is what made a turn complete before its
    reply existed.
    """
    raw = ask([{"role": "user", "content": "the Lisbon invoice"}], stream=True)
    assert "the Lisbon invoice" in raw, "the answer is not derived from the input"
    assert '"finish_reason": "stop"' in raw, "no finish_reason — this is the EmptyStreamError"
    assert raw.rstrip().endswith("[DONE]"), "the stream never terminates"
    chunks = [line for line in raw.split("\n") if line.startswith("data: ")]
    assert len(chunks) >= 3, f"only {len(chunks)} chunk(s); a consumer needs role, content and stop"


def test_a_plain_request_still_gets_a_completion(mock_server: None) -> None:
    payload = json.loads(ask([{"role": "user", "content": "hello"}], stream=False))
    assert payload["choices"][0]["message"]["content"].endswith("hello")
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_the_marker_makes_one_turn_answer_with_nothing(mock_server: None) -> None:
    """🔴 The `empty` outcome's only reachable path.

    A per-container switch cannot give one ring an empty turn while the others
    answer, so the marker is looked for in the request. This is the guard the
    end-to-end job cannot be: delete `marker_present` from the server and the
    conversation still passes, because a turn that answers is also a valid run.
    """
    from uffici_for_hermes import __file__ as _  # noqa: F401 - import sanity

    raw = ask([{"role": "user", "content": "ignore this one [say-nothing]"}], stream=True)
    assert '"content"' not in raw, f"the marker did not silence the answer: {raw[:200]}"
    assert '"finish_reason": "stop"' in raw, "an empty answer must still terminate cleanly"


def test_the_marker_is_found_anywhere_in_the_request(mock_server: None) -> None:
    """How a harness arranges a prompt is its business — a system block, an
    earlier turn, a restructured transcript. The fixture should not have to
    model that arrangement."""
    raw = ask(
        [
            {"role": "system", "content": "context: ignore this one [say-nothing]"},
            {"role": "user", "content": "what should I do?"},
        ],
        stream=True,
    )
    assert '"content"' not in raw, "the marker was missed outside the last user message"


def test_the_slow_marker_slows_only_the_turn_that_carries_it(mock_server: None) -> None:
    """#2079: the ``/stop`` journey needs one turn still running when it is stopped.
    The marker is read from the LAST user message, so a later turn whose history
    still carries it answers at full speed."""
    started = time.monotonic()
    raw = ask([{"role": "user", "content": "take your time [answer-slowly]"}], stream=False)
    assert time.monotonic() - started >= 1.4, "the marked turn was not slowed"
    assert "take your time" in raw

    started = time.monotonic()
    ask(
        [
            {"role": "user", "content": "take your time [answer-slowly]"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "and now quickly"},
        ],
        stream=False,
    )
    assert time.monotonic() - started < 1.0, "a later turn was slowed by the marker in its history"


def test_the_ask_marker_makes_the_model_call_hermess_clarify(mock_server: None) -> None:
    """🔴 #2232: the only way a scripted run reaches the question card.

    Hermes asks a person through its ``clarify`` tool, so the model must CALL it — a
    tool call in the stream, with a question and its choices, and ``tool_calls`` as the
    finish reason. Without this the e2e cannot exercise the card at all.
    """
    raw = ask(
        [{"role": "user", "content": "Ana: do we have a board? [ask-a-question]"}], stream=True
    )
    chunks = [
        json.loads(line[len("data: "):])
        for line in raw.split("\n")
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    calls = [
        call
        for chunk in chunks
        for call in (chunk["choices"][0]["delta"].get("tool_calls") or [])
    ]
    assert len(calls) == 1 and calls[0]["function"]["name"] == "clarify"
    arguments = json.loads(calls[0]["function"]["arguments"])
    assert arguments["question"] and len(arguments["choices"]) == 2
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_once_the_person_answered_the_model_says_what_they_chose(mock_server: None) -> None:
    """The turn resumes with the clarify tool's result; the reply must come from it — a
    fixed string would pass even if the answer never reached the model."""
    messages = [
        {"role": "user", "content": "Ana: do we have a board? [ask-a-question]"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "clarify", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_1",
         "content": json.dumps({"question": "Which?", "user_response": "Uffici task board"})},
    ]
    payload = json.loads(ask(messages, stream=False))
    message = payload["choices"][0]["message"]
    assert "tool_calls" not in message, "the model asked again instead of answering"
    assert "Uffici task board" in message["content"]


def test_a_marker_the_person_was_already_asked_with_does_not_ask_again(mock_server: None) -> None:
    """#2235: a turn cancelled while it waited on its question leaves its message
    unanswered, so the next brief carries the marker again beside the person's words.
    The person was already asked; the model answers."""
    messages = [
        {"role": "user", "content": "Inês: which office? [ask-a-question]"},
        {"role": "user", "content": "Inês: which office? [ask-a-question]\nInês: Lisbon, please"},
    ]
    message = json.loads(ask(messages, stream=False))["choices"][0]["message"]
    assert "tool_calls" not in message, "the model asked a person who was already asked"
    assert message["content"].startswith("[mock] ")


def test_a_request_that_offers_no_clarify_is_never_asked_through_it(mock_server: None) -> None:
    """#2235: Hermes makes tool-less calls of its own around a turn, carrying the first
    message alone; a model cannot call a tool it was not given."""
    messages = [{"role": "user", "content": "Inês: which office? [ask-a-question]"}]
    message = json.loads(ask(messages, stream=False, tools=[]))["choices"][0]["message"]
    assert "tool_calls" not in message, "the model called clarify, which was not offered"
