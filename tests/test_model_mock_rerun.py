"""The deterministic model's re-run marker (#2454): a marked turn calls ``uffici_rerun`` —
by name, or through Hermes's ``tool_call`` bridge — naming the latest ``tool_call`` URN a
tool message in the request carries, and the turn it resumes answers from the re-run's
result, so the answer is derived from what the platform returned and never a fixed string.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from uffici_for_hermes.wire import WIRE_V1

MOCK = Path(__file__).resolve().parents[1] / "dev" / "model_mock" / "server.py"
PORT = 18143
BASE = f"http://127.0.0.1:{PORT}"

DIRECT = [{"type": "function", "function": {"name": "uffici_rerun", "parameters": {}}}]
BRIDGED = [
    {
        "type": "function",
        "function": {
            "name": "tool_search",
            "description": "Search deferred tools. Available: uffici_rerun — Re-run a call.",
            "parameters": {},
        },
    },
    {"type": "function", "function": {"name": "tool_call", "parameters": {}}},
]

#: An earlier turn's tool call and its result, as the resumed session carries them.
EARLIER = [
    {"role": "user", "content": "Ana: is there a board? [use-a-tool]"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_mock_tool",
                "type": "function",
                "function": {"name": "list_boards", "arguments": "{}"},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_mock_tool",
        "content": json.dumps(
            {"callUrn": "urn:uffici:tool_call:7", "result": {"boards": [{"name": "Support"}]}}
        ),
    },
    {"role": "assistant", "content": "[mock] the tool said: Support"},
]
MARKED = [*EARLIER, {"role": "user", "content": "Ana: check again. [rerun-a-tool]"}]


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


@pytest.mark.parametrize(("tools", "name"), [(DIRECT, "uffici_rerun"), (BRIDGED, "tool_call")])
def test_a_marked_turn_reruns_the_latest_call_it_can_see(
    mock_log: Path, tools: list, name: str
) -> None:
    newer = {**EARLIER[2], "content": '{"callUrn": "urn:uffici:tool_call:9", "result": {}}'}
    calls = ask([*EARLIER, newer, MARKED[-1]], tools)["tool_calls"]
    assert [c["function"]["name"] for c in calls] == [name]
    arguments = json.loads(calls[0]["function"]["arguments"])
    sent = arguments if name == "uffici_rerun" else arguments["calls"][0]["arguments"]
    assert sent == {"call": "urn:uffici:tool_call:9"}
    # The mock's own grammar is the wire's: what it sends, the companion would admit.
    schema = json.loads((WIRE_V1 / "ops" / "tools.rerun.request.schema.json").read_text())
    assert re.fullmatch(schema["properties"]["call"]["pattern"], sent["call"])
    assert "rerun=True" in lines(mock_log, 1)[0]


def test_the_resumed_turn_answers_from_the_reruns_result(mock_log: Path) -> None:
    messages = [
        *MARKED,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_mock_rerun",
                    "type": "function",
                    "function": {"name": "uffici_rerun", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_mock_rerun",
            "content": '{"callUrn": "urn:uffici:tool_call:8", "result": {"boards": ["Support"]}}',
        },
    ]
    message = ask(messages, DIRECT)
    assert "tool_calls" not in message and message["content"].startswith("[mock] rerun:")
    assert "Support" in message["content"]
    line = lines(mock_log, 1)[0]
    assert "rerun=False rerun_result=True" in line and "Support" not in line


@pytest.mark.parametrize(
    "messages,tools",
    [
        (MARKED, []),  # the tool is not offered
        ([MARKED[-1]], DIRECT),  # no call to re-run
        (EARLIER, DIRECT),  # no marker
    ],
)
def test_a_turn_that_cannot_rerun_answers_instead(
    mock_log: Path, messages: list, tools: list
) -> None:
    assert "tool_calls" not in ask(messages, tools)
    assert "rerun=False" in lines(mock_log, 1)[0]
