"""The deterministic model's tool marker: the only way a scripted run calls a Uffici tool.

A marked turn calls ``list_boards`` — by name when Hermes offers it directly, through the
``tool_call`` bridge when Hermes's tool search deferred it (the pinned Hermes's default for
every plugin tool) — and the turn it resumes answers from the tool's result, so the answer
is derived from what the platform returned and never a fixed string.
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
PORT = 18139
BASE = f"http://127.0.0.1:{PORT}"

#: The fixture's catalog result, as the plugin hands it back to the model.
RESULT = json.dumps(
    {"callUrn": "urn:uffici:tool_call:1", "result": {"boards": [{"id": "b-1", "name": "Support"}]}}
)

DIRECT = [
    {"type": "function", "function": {"name": "clarify", "parameters": {}}},
    {"type": "function", "function": {"name": "list_boards", "parameters": {}}},
]
#: What the pinned Hermes sends instead: the bridge, its listing naming the deferred tool.
BRIDGED = [
    {
        "type": "function",
        "function": {
            "name": "tool_search",
            "description": "Search deferred tools. Available: list_boards — List the boards.",
            "parameters": {},
        },
    },
    {"type": "function", "function": {"name": "tool_describe", "parameters": {}}},
    {"type": "function", "function": {"name": "tool_call", "parameters": {}}},
]


@pytest.fixture(scope="module")
def mock_server() -> Iterator[subprocess.Popen]:
    process = subprocess.Popen(  # noqa: S603 - our own file, fixed argv
        [sys.executable, str(MOCK)],
        env={"MODEL_MOCK_PORT": str(PORT), "PATH": "/usr/bin:/bin"},
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
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
    yield process
    if process.poll() is None:
        process.terminate()
        process.communicate(timeout=5)


def ask(messages: list[dict], *, stream: bool, tools: list[dict]) -> str:
    body = json.dumps(
        {"model": "mock-model", "stream": stream, "messages": messages, "tools": tools}
    ).encode()
    request = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(request, timeout=5).read().decode()


def streamed_calls(raw: str) -> list[dict]:
    chunks = [
        json.loads(line[len("data: ") :])
        for line in raw.split("\n")
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    return [
        call for chunk in chunks for call in (chunk["choices"][0]["delta"].get("tool_calls") or [])
    ]


MARKED = [{"role": "user", "content": "Ana: which boards do we have? [use-a-tool]"}]


@pytest.mark.parametrize("stream", [True, False])
def test_a_marked_turn_calls_the_tool_it_is_offered_by_name(
    mock_server: subprocess.Popen, stream: bool
) -> None:
    raw = ask(MARKED, stream=stream, tools=DIRECT)
    if stream:
        calls = streamed_calls(raw)
    else:
        choice = json.loads(raw)["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        calls = choice["message"]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "list_boards"
    assert json.loads(calls[0]["function"]["arguments"]) == {}


@pytest.mark.parametrize("stream", [True, False])
def test_a_marked_turn_reaches_a_deferred_tool_through_the_bridge(
    mock_server: subprocess.Popen, stream: bool
) -> None:
    raw = ask(MARKED, stream=stream, tools=BRIDGED)
    calls = (
        streamed_calls(raw) if stream else json.loads(raw)["choices"][0]["message"]["tool_calls"]
    )
    assert [c["function"]["name"] for c in calls] == ["tool_call"]
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "calls": [{"name": "list_boards", "arguments": {}}]
    }


def test_a_turn_that_cannot_reach_the_tool_answers_instead(mock_server: subprocess.Popen) -> None:
    """A model calls only what it was given: Hermes's tool-less calls around a turn, or a
    catalog without the tool, get an answer."""
    for tools in ([], [DIRECT[0]], [BRIDGED[2]]):
        message = json.loads(ask(MARKED, stream=False, tools=tools))["choices"][0]["message"]
        assert "tool_calls" not in message
        assert message["content"].startswith("[mock] ")


def resumed(content: object, call_id: str = "call_mock_tool") -> list[dict]:
    return [
        *MARKED,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "list_boards", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": content},
    ]


@pytest.mark.parametrize("stream", [True, False])
def test_the_resumed_turn_answers_from_the_tools_result(
    mock_server: subprocess.Popen, stream: bool
) -> None:
    raw = ask(resumed(RESULT), stream=stream, tools=DIRECT)
    if stream:
        assert '"tool_calls"' not in raw, "the model called the tool again"
        assert "Support" in raw
    else:
        message = json.loads(raw)["choices"][0]["message"]
        assert "tool_calls" not in message
        assert "Support" in message["content"]
        assert "you answered" not in message["content"], "read as a clarify answer"


def test_a_result_in_parts_is_read_too(mock_server: subprocess.Popen) -> None:
    raw = ask(resumed([{"type": "text", "text": RESULT}]), stream=False, tools=DIRECT)
    assert "Support" in json.loads(raw)["choices"][0]["message"]["content"]


#: A refused call, as the plugin hands it back: no call was made, and the platform's result
#: carries the sentence the agent relays.
REFUSED = json.dumps(
    {
        "callUrn": None,
        "result": {
            "ok": False,
            "status": 403,
            "error": "scope_missing",
            "tool": "list_boards",
            "scope": "boards.read",
            "message": (
                "This agent has no access to do this anywhere in this organization, so nothing "
                "ran; tell the person, and an admin of this organization can grant this agent "
                "access."
            ),
        },
    },
    ensure_ascii=False,
)


@pytest.mark.parametrize("stream", [True, False])
def test_a_refused_result_is_relayed_with_its_sentence(
    mock_server: subprocess.Popen, stream: bool
) -> None:
    """A failed result is answered from like any other: the refusal and its sentence reach
    the reply, so a scripted run can tell a refused call from one that ran."""
    raw = ask(resumed(REFUSED), stream=stream, tools=DIRECT)
    if stream:
        assert '"tool_calls"' not in raw, "the model called the tool again"
        content = "".join(
            json.loads(line[len("data: ") :])["choices"][0]["delta"].get("content") or ""
            for line in raw.split("\n")
            if line.startswith("data: ") and line != "data: [DONE]"
        )
    else:
        content = json.loads(raw)["choices"][0]["message"]["content"]
    assert content.startswith("[mock] the tool said: ")
    assert "scope_missing" in content
    assert "This agent has no access to do this anywhere in this organization" in content


def test_a_narrowed_result_keeps_its_flag_in_the_reply(mock_server: subprocess.Popen) -> None:
    """A list answered under a narrower place says so, and the reply quotes it."""
    narrowed = json.dumps(
        {"callUrn": "urn:uffici:tool_call:2", "result": {"boards": [], "narrowed": True}}
    )
    content = json.loads(ask(resumed(narrowed), stream=False, tools=DIRECT))["choices"][0][
        "message"
    ]["content"]
    assert '"narrowed": true' in content


def test_a_clarify_answer_is_still_a_clarify_answer(mock_server: subprocess.Popen) -> None:
    messages = resumed(json.dumps({"user_response": "Uffici task board"}), call_id="call_1")
    content = json.loads(ask(messages, stream=False, tools=DIRECT))["choices"][0]["message"][
        "content"
    ]
    assert "you answered: Uffici task board" in content


def test_the_log_says_what_the_mock_did_and_nothing_it_was_told(
    mock_server: subprocess.Popen,
) -> None:
    """Booleans and counts only — never the tool's result or the person's words."""
    ask(MARKED, stream=False, tools=BRIDGED)
    ask(resumed(RESULT), stream=False, tools=DIRECT)
    mock_server.terminate()
    out, _ = mock_server.communicate(timeout=5)
    lines = [line for line in out.splitlines() if line.startswith("[mock] request ")]
    assert any("uses_tool=True tool_bridged=True tool_result=False" in line for line in lines)
    assert any("uses_tool=False tool_bridged=False tool_result=True" in line for line in lines)
    for secret in ("Support", "which boards", "b-1", "tool_call:1"):
        assert secret not in out, f"the mock logged content: {secret!r}"
