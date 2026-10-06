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

#: 🔴 A marker a script puts in ONE message to make that turn ASK (#2232): the model
#: calls Hermes's ``clarify`` tool, which the plugin turns into the question card. Read
#: from the LAST user message, like the slow marker, so a later turn carrying it in its
#: history answers instead of asking again.
ASK_MARKER = os.environ.get("MODEL_MOCK_ASK_MARKER", "[ask-a-question]")

#: What a marked turn asks, and offers.
ASK_QUESTION = "Which board do you mean?"
ASK_CHOICES = ["Uffici task board", "Another app"]

#: 🔴 A marker a script puts in ONE message to make that turn USE A UFFICI TOOL (#2351): the
#: model calls ``list_boards`` — offered directly, or behind Hermes's tool search, whose
#: ``tool_call`` bridge unwraps to it — and answers from the tool's result. Read from the
#: LAST user message, like the ask marker; the request that carries the result answers.
TOOL_MARKER = os.environ.get("MODEL_MOCK_TOOL_MARKER", "[use-a-tool]")

#: The tool a marked turn calls, and the id its call carries — what tells the tool's
#: result apart from a ``clarify`` answer when the turn resumes.
TOOL_NAME = "list_boards"
TOOL_CALL_ID = "call_mock_tool"

#: Hermes's tool-search bridge (``tools/tool_search.py``): a plugin tool is deferred behind
#: it by default, and ``tool_call {calls: [{name, arguments}]}`` invokes one.
BRIDGE_CALL = "tool_call"

#: 🔴 A marker a script puts in ONE message to make that turn REMEMBER (#2386): the model
#: calls ``uffici_remember`` — offered directly, or behind Hermes's tool search — with a
#: fact carrying the memory canary below. A later turn whose question shares a word with
#: the fact must then find the canary in Hermes's ``<memory-context>`` on the USER side,
#: recalled by the plugin's prefetch before the call.
MEMORY_MARKER = os.environ.get("MODEL_MOCK_MEMORY_MARKER", "[remember-this]")
MEMORY_TOOL = "uffici_remember"
MEMORY_CALL_ID = "call_mock_remember"
MEMORY_CANARY = os.environ.get("MODEL_MOCK_MEMORY_CANARY", "[memory-canary]")
MEMORY_FACT = f"Invoices for the Lisbon client go out on the 10th. {MEMORY_CANARY}"

#: 🔴 A marker a script puts in ONE message to make that turn RE-RUN A UFFICI TOOL CALL
#: (#2454): the model calls ``uffici_rerun`` — offered directly, or behind Hermes's tool
#: search — naming the LATEST ``tool_call`` URN any tool message in the request carries (an
#: earlier turn's ``list_boards`` result does), and answers from the re-run's result.
RERUN_MARKER = os.environ.get("MODEL_MOCK_RERUN_MARKER", "[rerun-a-tool]")
RERUN_TOOL = "uffici_rerun"
RERUN_CALL_ID = "call_mock_rerun"
#: The key a Uffici tool's result names its call by — read from the result, never matched
#: by the URN's own spelling (one codec, on the platform: no second spelling here).
CALL_URN_KEY = "callUrn"

#: The sentence a turn's recall carries when the platform's memory is down — matched by a
#: fixed fragment of the platform's instruction, inside the same fence.
MEMORY_UNAVAILABLE_FRAGMENT = "Memory was unavailable because of a system problem"

#: Hermes's fence around a prefetch (``agent/memory_manager.py``).
MEMORY_FENCE_OPEN, MEMORY_FENCE_CLOSE = "<memory-context>", "</memory-context>"

#: 🔴 The brief's context must reach the model on the SYSTEM side (the context fixture
#: carries this canary in its organization block). Reported per request as a boolean,
#: never as text: ``system_canary`` (in a system message) and ``user_canary`` (in any
#: other message — the placement the plugin must never make).
CONTEXT_CANARY = os.environ.get("MODEL_MOCK_CONTEXT_CANARY", "[context-canary]")

#: Appended to a reply whose request carried the canary on the SYSTEM side — the one
#: place a transcript-only observer (the local smoke) can see that the context reached
#: the model. Deterministic, and derived from the input like everything else here.
CONTEXT_SEEN_MARKER = " [context-seen]"

#: The sentence a brief carries when its context could not be read — it too must reach
#: the system side. Matched by a fixed fragment of the platform's instruction.
CONTEXT_MISSING_FRAGMENT = "context were unavailable because of a system problem"

MODEL_ID = os.environ.get("MODEL_MOCK_MODEL", "mock-model")

#: The largest request body this will read into memory.
MAX_BODY_BYTES = 4 << 20

#: How many completions this process has served, for the log line below.
_served = 0


def _user_messages(payload: dict) -> list[str]:
    """Every user message's text, oldest first."""
    texts = []
    for message in payload.get("messages") or []:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.append(
                " ".join(part.get("text", "") for part in content if isinstance(part, dict))
            )
    return texts


def _role_text(payload: dict, system: bool) -> str:
    """Every system message's text (``system=True``), or every other message's."""
    parts = []
    for message in payload.get("messages") or []:
        if (message.get("role") == "system") != system:
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(p.get("text", "") for p in content if isinstance(p, dict))
    return "\n".join(parts)


def _last_user_message(payload: dict) -> str:
    texts = _user_messages(payload)
    return texts[-1] if texts else ""


def _asked_before(payload: dict) -> bool:
    """Whether an EARLIER user message already carried the ask marker (#2235).

    A turn cancelled while it waited on its question leaves its message unanswered, so
    the next brief carries it again beside the person's new words. The person has
    already been asked; asking again would turn the answer into a second card.
    """
    return any(ASK_MARKER in text for text in _user_messages(payload)[:-1])


def _offers_clarify(payload: dict) -> bool:
    """Whether this request offers the ``clarify`` tool at all.

    A model cannot call a tool it was not given. Hermes also makes tool-less calls of
    its own around a turn (the first message alone, no tools); answering one of those
    with a clarify call is a thing no real model does.
    """
    for tool in payload.get("tools") or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name") == "clarify":
            return True
    return False


def _offered_names(payload: dict) -> list[str]:
    """The function names this request offers."""
    names = []
    for tool in payload.get("tools") or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.append(function["name"])
    return names


def _tool_route(payload: dict, tool: str = TOOL_NAME) -> str | None:
    """How a marked turn can reach ``tool``: ``direct`` when it is offered by name,
    ``bridge`` when Hermes deferred it behind its tool search and the bridge's listing names
    it, None when this request cannot reach it at all — a model calls only what it is given.
    """
    names = _offered_names(payload)
    if tool in names:
        return "direct"
    if BRIDGE_CALL in names and tool in json.dumps(payload.get("tools") or []):
        return "bridge"
    return None


def _tool_call(
    route: str, tool: str = TOOL_NAME, call_id: str = TOOL_CALL_ID, arguments: dict | None = None
) -> dict:
    """The one call a marked turn makes, by the route the request offers."""
    arguments = arguments or {}
    if route == "direct":
        name, wrapped = tool, arguments
    else:
        name, wrapped = BRIDGE_CALL, {"calls": [{"name": tool, "arguments": arguments}]}
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(wrapped)},
    }


def _last_tool_message(payload: dict) -> dict | None:
    messages = payload.get("messages") or []
    last = messages[-1] if messages else {}
    return last if isinstance(last, dict) and last.get("role") == "tool" else None


def _tool_text(content: object) -> str:
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content or "")


def _uffici_tool_result(payload: dict, call_id: str = TOOL_CALL_ID) -> str | None:
    """The tool's result, when this request resumes a turn after the marked call."""
    last = _last_tool_message(payload)
    if last is None or last.get("tool_call_id") != call_id:
        return None
    return _tool_text(last.get("content"))


def _latest_call_urn(payload: dict) -> str | None:
    """The newest ``tool_call`` URN any tool message in this request carries, or None."""
    latest = None
    for message in payload.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "tool":
            found = _call_urns(_tool_text(message.get("content")))
            if found:
                latest = found[-1]
    return latest


def _call_urns(text: str) -> list[str]:
    """Every non-empty ``callUrn`` a tool result carries, in order — the result as JSON,
    searched through nested objects, lists and JSON-encoded strings (a bridged call's
    result arrives wrapped)."""
    found: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            urn = value.get(CALL_URN_KEY)
            if isinstance(urn, str) and urn:
                found.append(urn)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str) and value[:1] in "{[":
            try:
                walk(json.loads(value))
            except ValueError:
                return

    walk(text)
    return found


def _fenced(text: str) -> list[str]:
    """Every ``<memory-context>`` block in ``text``, without its fence."""
    blocks, rest = [], text
    while MEMORY_FENCE_OPEN in rest:
        rest = rest.split(MEMORY_FENCE_OPEN, 1)[1]
        block, _, rest = rest.partition(MEMORY_FENCE_CLOSE)
        blocks.append(block)
    return blocks


def _clarify_result(payload: dict) -> str | None:
    """The person's answer, when this request resumes a turn after its ``clarify``.

    The last message is the tool's result; its JSON carries ``user_response`` (the
    pinned ``tools/clarify_tool.py``). A result that is not that shape is still the
    answer, as text.
    """
    last = _last_tool_message(payload)
    if last is None or last.get("tool_call_id") in (TOOL_CALL_ID, MEMORY_CALL_ID):
        # Not a clarify answer: no tool spoke last, or it was a marked turn's own tool.
        return None
    content = last.get("content")
    try:
        parsed = json.loads(content) if isinstance(content, str) else None
    except ValueError:
        parsed = None
    if isinstance(parsed, dict) and "user_response" in parsed:
        return str(parsed.get("user_response") or "")
    return str(content or "")


def _clarify_call() -> dict:
    """The one tool call a marked turn makes."""
    return {
        "id": "call_mock_clarify",
        "type": "function",
        "function": {
            "name": "clarify",
            "arguments": json.dumps({"question": ASK_QUESTION, "choices": ASK_CHOICES}),
        },
    }


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

    def _stream_tool_call(self, call: dict) -> None:
        """A tool call as Server-Sent Events: the role, the call, ``tool_calls``, done."""
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            for payload in (
                _chunk("", None),
                {**_chunk("", None), "choices": [
                    {"index": 0, "delta": {"tool_calls": [{"index": 0, **call}]},
                     "finish_reason": None},
                ]},
                _chunk("", "tool_calls"),
            ):
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

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
        # A question (#2232): the marked turn calls clarify, and the turn it resumes
        # after the person answered says what they chose.
        answered = _clarify_result(payload)
        asks = (
            answered is None
            and bool(ASK_MARKER)
            and ASK_MARKER in asked
            and not _asked_before(payload)
            and _offers_clarify(payload)
        )
        if answered is not None:
            text = f"{REPLY_PREFIX}you answered: {answered[:200]}"
        # A Uffici tool (#2351): the marked turn calls it, and the turn it resumes quotes
        # what the tool returned — derived from the result, so a run where the call never
        # reached the platform cannot pass.
        tool_result = _uffici_tool_result(payload)
        route = _tool_route(payload)
        uses_tool = (
            answered is None
            and tool_result is None
            and bool(TOOL_MARKER)
            and TOOL_MARKER in asked
            and route is not None
        )
        if tool_result is not None:
            text = f"{REPLY_PREFIX}the tool said: {tool_result[:500]}"
        # Memory (#2386): the marked turn remembers the canary's fact, and the turn it
        # resumes says the platform answered.
        remembered = _uffici_tool_result(payload, MEMORY_CALL_ID)
        memory_route = _tool_route(payload, MEMORY_TOOL)
        remembers = (
            answered is None
            and tool_result is None
            and remembered is None
            and not uses_tool
            and bool(MEMORY_MARKER)
            and MEMORY_MARKER in asked
            and memory_route is not None
        )
        if remembered is not None:
            text = f"{REPLY_PREFIX}remembered: {remembered[:200]}"
        # A re-run (#2454): the marked turn re-runs the latest call it can see, and the turn
        # it resumes quotes the re-run's result — derived from it, like the tool marker's.
        reran = _uffici_tool_result(payload, RERUN_CALL_ID)
        rerun_route = _tool_route(payload, RERUN_TOOL)
        rerun_urn = _latest_call_urn(payload)
        reruns = (
            answered is None
            and tool_result is None
            and remembered is None
            and reran is None
            and not uses_tool
            and not remembers
            and bool(RERUN_MARKER)
            and RERUN_MARKER in asked
            and rerun_route is not None
            and rerun_urn is not None
        )
        if reran is not None:
            text = f"{REPLY_PREFIX}rerun: {reran[:500]}"

        # 🔴 One line per request, and it carries NO transcript content — counts,
        # booleans, lengths, and the ROLES of the last four messages (the shape of
        # the call: a tool-less aux call on the first message alone read, in #2272's
        # diagnosis, exactly like a turn until the roles showed it). The default handler's request
        # line is silenced below precisely because it would carry content; this
        # says what the mock DID without saying what it was told.
        #
        # It exists because a turn that closed `empty` in CI and `answered` on a
        # laptop could not be told apart from the companion's side: the wire
        # shows no post either way, and "the harness produced nothing" and "the
        # harness never called the model" look identical. Now they do not.
        system_text = _role_text(payload, system=True)
        other_text = _role_text(payload, system=False)
        system_canary = bool(CONTEXT_CANARY) and CONTEXT_CANARY in system_text
        user_canary = bool(CONTEXT_CANARY) and CONTEXT_CANARY in other_text
        system_missing = CONTEXT_MISSING_FRAGMENT in system_text
        # 🔴 Where a recall landed: inside Hermes's fence, on the USER side. Never text.
        recalled_blocks = _fenced(other_text)
        memory_recalled = any(MEMORY_CANARY in block for block in recalled_blocks)
        memory_unavailable = any(MEMORY_UNAVAILABLE_FRAGMENT in b for b in recalled_blocks)
        memory_system = bool(MEMORY_CANARY) and MEMORY_CANARY in system_text
        if system_canary and text:
            text += CONTEXT_SEEN_MARKER

        global _served
        _served += 1
        print(
            f"[mock] request {_served}: stream={bool(wants_stream)} "
            f"marker={marker_present} slow={slow} asks={asks} asked_chars={len(asked)} "
            f"uses_tool={uses_tool} tool_bridged={uses_tool and route == 'bridge'} "
            f"tool_result={tool_result is not None} "
            f"reply_chars={len(text)} messages={len(payload.get('messages') or [])} "
            f"tools={len(payload.get('tools') or [])} "
            f"system_canary={system_canary} user_canary={user_canary} "
            f"system_missing={system_missing} "
            f"remembers={remembers} remembered={remembered is not None} "
            f"memory_recalled={memory_recalled} memory_unavailable={memory_unavailable} "
            f"memory_system={memory_system} "
            f"rerun={reruns} rerun_result={reran is not None} "
            f"roles={','.join(str(m.get('role')) for m in (payload.get('messages') or [])[-4:])}",
            flush=True,
        )

        call = (
            _clarify_call() if asks
            else _tool_call(route) if uses_tool
            else _tool_call(memory_route, MEMORY_TOOL, MEMORY_CALL_ID, {"text": MEMORY_FACT})
            if remembers
            else _tool_call(rerun_route, RERUN_TOOL, RERUN_CALL_ID, {"call": rerun_urn})
            if reruns
            else None
        )
        if call is not None:
            if wants_stream:
                self._stream_tool_call(call)
                return
            self._reply(200, {
                **_completion(""),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": None, "tool_calls": [call]},
                    "finish_reason": "tool_calls",
                }],
            })
            return
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
