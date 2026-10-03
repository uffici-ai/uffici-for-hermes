"""The model mock's account of where the brief's context landed.

The end-to-end runs judge the context's placement from this line alone: the canary in
a SYSTEM message is a pass, the canary anywhere else is the placement the plugin must
never make, and the missing-context sentence must reach the system side too. Each is a
boolean on the request line — never the text itself.
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
PORT = 18137
BASE = f"http://127.0.0.1:{PORT}"
CANARY = "[context-canary]"
MISSING = (
    "The organization's and this place's context were unavailable because of a system "
    "problem; tell the person you are working without them."
)


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


def _ask(messages: list[dict]) -> str:
    body = json.dumps({"model": "mock-model", "stream": False, "messages": messages}).encode()
    request = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    answer = json.loads(urllib.request.urlopen(request, timeout=5).read())
    return answer["choices"][0]["message"]["content"]


def _lines(log: Path) -> list[str]:
    for _ in range(50):
        lines = [line for line in log.read_text().splitlines() if line.startswith("[mock] request")]
        if lines:
            return lines
        time.sleep(0.05)
    return []


def test_reports_the_canary_on_the_system_side_and_never_its_text(mock_log: Path) -> None:
    _ask(
        [
            {"role": "system", "content": f"Org context: {CANARY} be brief."},
            {"role": "user", "content": "hello"},
        ]
    )
    line = _lines(mock_log)[-1]
    assert "system_canary=True" in line
    assert "user_canary=False" in line
    assert "system_missing=False" in line
    assert CANARY not in line and "be brief" not in line


def test_reports_a_canary_placed_on_the_user_side(mock_log: Path) -> None:
    _ask([{"role": "system", "content": "x"}, {"role": "user", "content": f"{CANARY} hi"}])
    line = _lines(mock_log)[-1]
    assert "system_canary=False" in line
    assert "user_canary=True" in line


def test_reports_the_missing_context_sentence_on_the_system_side(mock_log: Path) -> None:
    _ask(
        [
            {"role": "system", "content": [{"type": "text", "text": MISSING}]},
            {"role": "user", "content": "hello"},
        ]
    )
    assert "system_missing=True" in _lines(mock_log)[-1]


def test_a_reply_says_the_context_was_seen_only_when_it_was_on_the_system_side(
    mock_log: Path,
) -> None:
    seen = _ask([{"role": "system", "content": CANARY}, {"role": "user", "content": "hi"}])
    assert seen.endswith(" [context-seen]")
    unseen = _ask([{"role": "system", "content": "x"}, {"role": "user", "content": f"{CANARY}"}])
    assert "[context-seen]" not in unseen
