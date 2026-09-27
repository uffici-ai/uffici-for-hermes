"""A deterministic OpenAI-compatible model, for the local loop.

🔴 It exists so the whole loop — platform, companion, plugin, harness — can run
on a laptop and in CI with **no secret and no flaky answer**. A real provider
makes the one thing the loop is testing (did the turn arrive, did the reply come
back) depend on a network, a key and a model's mood.

Standard library only: it is mounted into the Hermes image, which ships its own
sealed environment, and a dependency here would be one more thing to install
into a container whose whole point is that it is already complete.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: Prefixed to every answer, so a reply that reached the platform is obviously
#: the mock's and never mistaken for a model's.
REPLY_PREFIX = "[mock] "

#: Seconds to wait before answering, for exercising the platform's deadlines.
DELAY_SECONDS = float(os.environ.get("MODEL_MOCK_DELAY_SECONDS", "0"))

#: When set, the model answers with no content at all — the `empty` outcome.
ANSWER_EMPTY = os.environ.get("MODEL_MOCK_EMPTY", "").strip() == "1"

#: 🔴 A marker the SCRIPT can put in one ring's text to make that one turn
#: answer with nothing. The env switch above is per-container, so a fixture
#: needing an `empty` close AND answered closes in the same run could not use
#: it — and `empty` is a real outcome the platform treats differently from an
#: error, so it has to be reachable in a run that also answers.
EMPTY_MARKER = os.environ.get("MODEL_MOCK_EMPTY_MARKER", "[say-nothing]")

#: 🔴 A marker a smoke step puts in ONE message to make that turn slow (#2079): the
#: `/stop` journey needs a turn that is still running when the person stops it,
#: and the container-wide delay above would slow every other step of the run.
#: Looked for in the LAST user message only, so the turns after it in the same
#: conversation — whose history still carries it — answer at full speed.
SLOW_MARKER = os.environ.get("MODEL_MOCK_SLOW_MARKER", "[answer-slowly]")

#: How long a turn carrying the slow marker takes.
SLOW_SECONDS = float(os.environ.get("MODEL_MOCK_SLOW_SECONDS", "30"))

MODEL_ID = os.environ.get("MODEL_MOCK_MODEL", "mock-model")

#: The largest request body this will read into memory.
MAX_BODY_BYTES = 4 << 20

#: How many completions this process has served, for the log line below.
_served = 0


def _last_user_message(payload: dict) -> str:
    for message in reversed(payload.get("messages") or []):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
    return ""


def _completion(text: str) -> dict:
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _chunk(text: str, finish: str | None) -> dict:
    """One `chat.completion.chunk`, the shape an SSE consumer expects."""
    delta = {"content": text} if text else {}
    return {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def handle_one_request(self) -> None:
        """Serve one request, tolerating a client that hangs up.

        🔴 The suppression belongs HERE, not only around the body writes. A
        cancelled turn makes Hermes close the socket mid-response, and the write
        that fails is as likely to be the HEADER write — inside `send_response`
        and `end_headers`, which this class does not own — as the body. Twelve
        tracebacks per run came from exactly that, and a traceback from the
        stand-in model is indistinguishable, in a CI log, from the harness
        having produced nothing: both are "no content, plus noise".

        Suppressing it is correct rather than convenient: a client hanging up is
        that client's business, and the only thing this server can usefully do
        is stop writing.
        """
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            super().handle_one_request()
            return
        # The connection is gone; do not try to read another request on it.
        self.close_connection = True

    def _stream(self, text: str) -> None:
        """Answer as Server-Sent Events.

        🔴 This is the mode that matters. Hermes asks the provider for a STREAM
        — `stream: true` — and a mock that answered only plain JSON produced
        `EmptyStreamError: Provider returned an empty stream with no
        finish_reason`, three retries, and a turn that completed before its own
        reply existed. The companion then recorded `empty` for a turn that went
        on to answer, which read for two runs like an adapter defect and was a
        mock that could not speak the protocol it was standing in for.
        """
        # 🔴 The connection is CLOSED at the end, and the handler is told so
        # before the headers go out. This speaks HTTP/1.1, where a body with no
        # `Content-Length` and no chunked encoding can only be framed by the
        # connection ending — without this the client waits for more, the
        # handler keeps the socket alive, and the write eventually fails with a
        # broken pipe that looks like the mock crashing.
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(payload: object) -> None:
            # A client that hangs up mid-stream is ordinary (a cancelled turn);
            # it is not this server's problem and must not print a traceback.
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()

        # A role chunk, the content, then a finish reason — the three a consumer
        # needs to see a complete message.
        emit(_chunk("", None))
        if text:
            emit(_chunk(text, None))
        emit(_chunk("", "stop"))
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    def _reply(self, status: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        # Same reason as `handle_one_request`: the client may already be gone.
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._reply(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]})
            return
        if self.path.rstrip("/") in ("/healthz", "/health"):
            self._reply(200, {"status": "ok"})
            return
        self._reply(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._reply(404, {"error": {"message": "not found"}})
            return

        # Bounded: dev-only and non-routable, but an unbounded read is an
        # unbounded allocation and the cap costs nothing.
        length = min(int(self.headers.get("Content-Length") or 0), MAX_BODY_BYTES)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._reply(400, {"error": {"message": "body is not JSON"}})
            return

        if DELAY_SECONDS:
            time.sleep(DELAY_SECONDS)

        wants_stream = bool(payload.get("stream"))

        # 🔴 Deterministic, and derived from the INPUT: a fixed string would
        # pass even if the brief never reached the model, which is exactly the
        # failure the loop exists to catch.
        asked = _last_user_message(payload).strip()
        slow = bool(SLOW_MARKER) and SLOW_MARKER in asked
        if slow:
            time.sleep(SLOW_SECONDS)
        # 🔴 The marker is looked for in the WHOLE request, not only in the last
        # user message. How a harness arranges a prompt is its business — it may
        # put the transcript in a system block, fold it into an earlier turn, or
        # restructure it entirely — and a fixture that wants one turn to answer
        # with nothing should not have to model that arrangement.
        marker_present = bool(EMPTY_MARKER) and EMPTY_MARKER in json.dumps(payload)
        text = "" if (ANSWER_EMPTY or marker_present) else REPLY_PREFIX + asked[:500]

        # 🔴 One line per request, and it carries NO transcript content — a
        # count, two booleans and two lengths. The default handler's request
        # line is silenced below precisely because it would carry content; this
        # says what the mock DID without saying what it was told.
        #
        # It exists because a turn that closed `empty` in CI and `answered` on a
        # laptop could not be told apart from the companion's side: the wire
        # shows no post either way, and "the harness produced nothing" and "the
        # harness never called the model" look identical. Now they do not.
        global _served
        _served += 1
        print(
            f"[mock] request {_served}: stream={bool(wants_stream)} "
            f"marker={marker_present} slow={slow} asked_chars={len(asked)} reply_chars={len(text)}",
            flush=True,
        )

        if wants_stream:
            self._stream(text)
            return
        self._reply(200, _completion(text))

    def log_message(self, fmt: str, *args: object) -> None:
        """🔴 Silent. The default handler logs the request line, and a request
        line is this mock's only sight of transcript content.
        """
        return


def main() -> None:
    port = int(os.environ.get("MODEL_MOCK_PORT", "8080"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()  # noqa: S104


if __name__ == "__main__":
    main()
