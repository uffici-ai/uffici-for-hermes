"""The deterministic model's memory marker, and its account of where a recall landed.

A marked turn calls ``uffici_remember`` — by name, or through Hermes's ``tool_call``
bridge — with a fact carrying the memory canary. The end-to-end runs then judge, from the
request line alone, that a later turn found the canary inside Hermes's
``<memory-context>`` on the USER side (and, with the platform's memory down, the outage's
sentence there instead). Each is a boolean on the line — never the text.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

MOCK = Path(__file__).resolve().parents[1] / "dev" / "model_mock" / "server.py"
PORT = 18141
BASE = f"http://127.0.0.1:{PORT}"
CANARY = "[memory-canary]"
UNAVAILABLE = (
    "Memory was unavailable because of a system problem; tell the person you could not "
    "check what was remembered."
)

DIRECT = [{"type": "function", "function": {"name": "uffici_remember", "parameters": {}}}]
BRIDGED = [
    {
        "type": "function",
        "function": {
            "name": "tool_search",
            "description": "Search deferred tools. Available: uffici_remember — Remember.",
            "parameters": {},
        },
    },
    {"type": "function", "function": {"name": "tool_call", "parameters": {}}},
]
MARKED = [{"role": "user", "content": "Ana: remember the invoice day. [remember-this]"}]


@pytest.fixture
def mock_log(tmp_path: Path):
    log = tmp_path / "mock.log"
    with log.open("w") as out:
        process = subprocess.Popen(  # noqa: S603 - our own file, fixed argv
            [sys.executable, str(MOCK)],
            env={"MODEL_MOCK_PORT": str(PORT), "PATH": "/usr/bin:/bin"},
            stdout=out,
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
        yield log
        process.terminate()
        process.wait(timeout=5)


def ask(messages: list[dict], tools: list[dict] | None = None) -> dict:
    body = json.dumps(
        {"model": "mock-model", "stream": False, "messages": messages, "tools": tools or []}
    ).encode()
    request = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(request, timeout=5).read())["choices"][0]["message"]


def lines(log: Path, count: int) -> list[str]:
    for _ in range(50):
        found = [line for line in log.read_text().splitlines() if line.startswith("[mock] request")]
        if len(found) >= count:
            return found
        time.sleep(0.05)
    return []


@pytest.mark.parametrize(("tools", "name"), [(DIRECT, "uffici_remember"), (BRIDGED, "tool_call")])
def test_a_marked_turn_remembers_the_canarys_fact(mock_log: Path, tools: list, name: str) -> None:
    calls = ask(MARKED, tools)["tool_calls"]
    assert [c["function"]["name"] for c in calls] == [name]
    arguments = json.loads(calls[0]["function"]["arguments"])
    fact = (
        arguments["text"]
        if name == "uffici_remember"
        else arguments["calls"][0]["arguments"]["text"]
    )
    assert CANARY in fact and "Lisbon" in fact


def test_the_resumed_turn_answers_and_does_not_remember_again(mock_log: Path) -> None:
    messages = [
        *MARKED,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_mock_remember",
                    "type": "function",
                    "function": {"name": "uffici_remember", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_mock_remember", "content": '{"urn": "u"}'},
    ]
    message = ask(messages, DIRECT)
    assert "tool_calls" not in message and message["content"].startswith("[mock] remembered:")


def test_a_turn_without_the_tool_answers_instead(mock_log: Path) -> None:
    assert "tool_calls" not in ask(MARKED, [])


def fenced(text: str) -> str:
    return f"<memory-context>\n[System note: recalled]\n\n{text}\n</memory-context>"


def test_reports_a_recall_inside_the_fence_on_the_user_side_and_never_its_text(
    mock_log: Path,
) -> None:
    ask([{"role": "user", "content": "When is Lisbon invoiced?\n\n" + fenced(f"- x {CANARY}")}])
    (line,) = lines(mock_log, 1)
    assert "memory_recalled=True" in line and "memory_system=False" in line
    assert CANARY not in mock_log.read_text() and "Lisbon" not in mock_log.read_text()


def test_the_canary_outside_the_fence_is_not_a_recall(mock_log: Path) -> None:
    ask([{"role": "user", "content": f"Remember {CANARY}"}])
    (line,) = lines(mock_log, 1)
    assert "memory_recalled=False" in line


def test_reports_the_canary_on_the_system_side(mock_log: Path) -> None:
    ask([{"role": "system", "content": fenced(CANARY)}, {"role": "user", "content": "hi"}])
    (line,) = lines(mock_log, 1)
    assert "memory_system=True" in line and "memory_recalled=False" in line


def test_reports_the_outage_sentence_inside_the_fence(mock_log: Path) -> None:
    ask([{"role": "user", "content": "q\n\n" + fenced(f"Uffici memory: {UNAVAILABLE}")}])
    ask([{"role": "user", "content": UNAVAILABLE}])
    first, second = lines(mock_log, 2)
    assert "memory_unavailable=True" in first and "memory_unavailable=False" in second
