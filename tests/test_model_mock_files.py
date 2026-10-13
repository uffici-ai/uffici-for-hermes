"""The deterministic model's files (#2679): what the change gate's files conversation needs.

A person's PDF and PNG reach the model as media — the PNG as an image part Hermes inlined,
the PDF as a document Hermes names by its local path — and the mock names each canary only
when it found it in the BYTES, so a reply naming both proves both files arrived. A marked
turn makes a file inside the shared workspace and answers Hermes's ``MEDIA:`` tag; another
saves it into Files through ``uffici_upload`` with ``to``.
"""

from __future__ import annotations

import base64
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
PORT = 18145
BASE = f"http://127.0.0.1:{PORT}"

PNG = b"\x89PNG\r\n\x1a\n...tEXtComment\x00Meter photo. [png-canary]..."
PDF = b"%PDF-1.4\n... (Invoice 2026-118. [pdf-canary]) Tj ..."

UPLOAD_DIRECT = [{"type": "function", "function": {"name": "uffici_upload", "parameters": {}}}]
UPLOAD_BRIDGED = [
    {
        "type": "function",
        "function": {
            "name": "tool_search",
            "description": "Search deferred tools. Available: uffici_upload — Upload a file.",
            "parameters": {},
        },
    },
    {"type": "function", "function": {"name": "tool_call", "parameters": {}}},
]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("uffici-workspace")


@pytest.fixture(scope="module")
def mock_server(workspace: Path) -> Iterator[subprocess.Popen]:
    process = subprocess.Popen(  # noqa: S603 - our own file, fixed argv
        [sys.executable, str(MOCK)],
        env={
            "MODEL_MOCK_PORT": str(PORT),
            "MODEL_MOCK_WORKSPACE": str(workspace),
            "PATH": "/usr/bin:/bin",
        },
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


def ask(messages: list[dict], *, tools: list[dict] | None = None) -> dict:
    body = json.dumps(
        {"model": "mock-model", "stream": False, "messages": messages, "tools": tools or []}
    ).encode()
    request = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(request, timeout=5).read())["choices"][0]["message"]


def media_turn(pdf_path: Path, image: bytes) -> list[dict]:
    """A user turn as Hermes builds it: the document note naming the PDF's local path, and
    the image inlined as a data URL part."""
    note = (
        f"[The user sent a document: 'invoice.pdf'. It is saved at: {pdf_path}. Its text is "
        "not inlined here.]\nAna: here are the invoice and the meter photo."
    )
    data = "data:image/png;base64," + base64.b64encode(image).decode()
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": note},
                {"type": "image_url", "image_url": {"url": data}},
            ],
        }
    ]


def test_both_canaries_are_named_only_from_the_bytes(
    mock_server: subprocess.Popen, workspace: Path
) -> None:
    pdf = workspace / "conversations" / "c1" / "f1.0-invoice.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(PDF)
    content = ask(media_turn(pdf, PNG))["content"]
    assert "[pdf-canary]" in content and "[png-canary]" in content
    # 🔴 Never the file's path: Hermes delivers a local path a reply names, so an echo
    # sent the person's own PDF back as an attachment.
    assert str(workspace) not in content
    # A PDF that does not carry it, and an image without it, name nothing.
    other = workspace / "other.pdf"
    other.write_bytes(b"%PDF-1.4 nothing here")
    content = ask(media_turn(other, b"\x89PNG plain"))["content"]
    assert "[pdf-canary]" not in content and "[png-canary]" not in content
    # A path that does not exist is not media either.
    content = ask(media_turn(workspace / "absent.pdf", b"not base64 at all"))["content"]
    assert "[pdf-canary]" not in content


def test_a_marked_turn_makes_the_report_inside_the_workspace(
    mock_server: subprocess.Popen, workspace: Path
) -> None:
    content = ask([{"role": "user", "content": "Ana: make me the report [make-a-file]"}])["content"]
    report = workspace / "out" / "q3-report.pdf"
    assert f"MEDIA:{report}" in content
    data = report.read_bytes()
    assert data.startswith(b"%PDF-") and b"[report-canary]" in data
    assert len(data) == 256 * 1024, "the interrupted upload needs a file larger than 64 KiB"


@pytest.mark.parametrize(
    ("tools", "name"), [(UPLOAD_DIRECT, "uffici_upload"), (UPLOAD_BRIDGED, "tool_call")]
)
def test_a_marked_turn_saves_the_report_into_files(
    mock_server: subprocess.Popen, workspace: Path, tools: list[dict], name: str
) -> None:
    message = ask([{"role": "user", "content": "Ana: save it [save-a-file]"}], tools=tools)
    call = message["tool_calls"][0]
    assert call["id"] == "call_mock_upload" and call["function"]["name"] == name
    arguments = json.loads(call["function"]["arguments"])
    expected = {"local_path": "out/q3-report.pdf", "to": "reports/q3-report.pdf"}
    assert arguments == expected or arguments == {
        "calls": [{"name": "uffici_upload", "arguments": expected}]
    }
    assert (workspace / "out" / "q3-report.pdf").is_file()


def resumed(result: str) -> list[dict]:
    return [
        {"role": "user", "content": "Ana: save it [save-a-file]"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_mock_upload",
                    "type": "function",
                    "function": {"name": "uffici_upload", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_mock_upload", "content": result},
    ]


def test_the_resumed_save_says_saved_only_when_it_was(mock_server: subprocess.Popen) -> None:
    saved = ask(
        resumed(json.dumps({"savedAt": "/reports/q3-report.pdf", "size": 1})), tools=UPLOAD_DIRECT
    )
    assert "tool_calls" not in saved and saved["content"].startswith("[mock] saved: ")
    refused = ask(resumed(json.dumps({"error": "transfer_denied"})), tools=UPLOAD_DIRECT)
    assert "saved:" not in refused["content"] and "the upload failed" in refused["content"]
    assert "you answered" not in refused["content"], "read as a clarify answer"


def test_a_save_the_mock_cannot_reach_answers_instead(mock_server: subprocess.Popen) -> None:
    message = ask([{"role": "user", "content": "Ana: save it [save-a-file]"}], tools=[])
    assert "tool_calls" not in message


def test_the_log_carries_the_files_booleans_and_no_content(
    mock_server: subprocess.Popen, workspace: Path
) -> None:
    mock_server.terminate()
    out, _ = mock_server.communicate(timeout=5)
    lines = [line for line in out.splitlines() if line.startswith("[mock] request ")]
    for flag in (
        "pdf_media=True",
        "image_media=True",
        "media_written=True",
        "upload_call=True",
        "upload_result=True",
    ):
        assert any(flag in line for line in lines), flag
    for secret in ("[pdf-canary]", "[png-canary]", "invoice", str(workspace), "q3-report"):
        assert secret not in out, f"the mock logged content: {secret!r}"


def test_a_mock_with_no_workspace_makes_nothing(tmp_path: Path) -> None:
    env = {"MODEL_MOCK_PORT": str(PORT + 1), "PATH": "/usr/bin:/bin"}
    process = subprocess.Popen(  # noqa: S603
        [sys.executable, str(MOCK)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        base = f"http://127.0.0.1:{PORT + 1}"
        for _ in range(100):
            try:
                urllib.request.urlopen(f"{base}/healthz", timeout=0.5).read()
                break
            except (urllib.error.URLError, OSError):
                time.sleep(0.1)
        body = json.dumps(
            {"model": "m", "messages": [{"role": "user", "content": "[make-a-file]"}]}
        ).encode()
        request = urllib.request.Request(
            f"{base}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
        )
        content = json.loads(urllib.request.urlopen(request, timeout=5).read())["choices"][0][
            "message"
        ]["content"]
        assert "MEDIA:" not in content and "no workspace" in content
    finally:
        process.terminate()
        process.wait(timeout=5)
