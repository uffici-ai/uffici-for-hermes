"""The agent's Uffici tools, offered to Hermes as Hermes tools (#2350, epic #2128).

The owner's decision (2026-10-03): Hermes's MCP client is process-wide and carries no
session, so the platform's tools cannot reach Hermes as an MCP pointer — one Hermes
serves many conversations, and a tool call must act on the request of the conversation
it was made in. So this plugin registers the agent's catalog (``tools.list``) as PLUGIN
tools, and each call is forwarded over the call tunnel as ``tools.call``. The plugin
stays a format translator: no network code, no credential, no retry — the companion
holds the connection and never retries a tool call either (a tool is not idempotent).

What the pinned Hermes (v2026.9.24) does with them, read from its source:

- ``PluginContext.register_tool`` (``hermes_cli/plugins.py``) returns a
  ``PluginRegistration`` whose ``dispose()`` removes the tool again
  (``hermes_cli/plugins_ledger.py``) — how a tool that left the catalog goes.
- A plugin's toolset is ON for every platform unless an operator saved a selection that
  knows it and leaves it out (``hermes_cli/tools_config.py``, ``_enabled_plugin_toolsets``),
  and naming it ``uffici`` — the platform's own name — also puts it in Hermes's implicit
  ``hermes-uffici`` bundle (``toolsets.py``, ``_plugin_platform_bundle``). So no
  ``platform_toolsets`` entry is written: that key REPLACES a platform's whole selection,
  and ``[hermes-uffici, uffici]`` there leaves the agent with no core tools at all.
- Hermes's tool search (on by default) defers every plugin tool behind
  ``tool_search`` / ``tool_describe`` / ``tool_call``; a ``tool_call`` is unwrapped
  before dispatch, so the handler below runs either way (``model_tools.py``).
- A handler runs in the agent's worker thread, which carries the turn's session
  ContextVars (``gateway/session_context.py``): ``HERMES_SESSION_CHAT_ID`` is the
  source's ``chat_id`` — the Uffici conversation id — and ``HERMES_SESSION_PLATFORM`` is
  ``uffici``. That is how a call finds its conversation, and through the adapter's
  ledger, its request.

🔴 Logged: tool NAMES, counts, request ids and error classes. Never a tool's arguments or
its result — they are tenant content.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
import weakref
from collections.abc import Awaitable, Callable
from typing import Any

from .call_contract import INSTRUCTIONS
from .calls import CallResult

logger = logging.getLogger(__name__)

#: The companion's word for "the plugin may call tools.list and tools.call, and a post may
#: carry citations and activity", in its hello.
FEATURE_TOOLS = "tools"

#: The Hermes toolset the catalog is registered in — the platform's own name, so Hermes's
#: implicit ``hermes-uffici`` bundle carries it (see the module docstring).
TOOLSET = "uffici"

#: The session ContextVars Hermes binds for a turn, by their legacy names.
SESSION_PLATFORM = "HERMES_SESSION_PLATFORM"
SESSION_CHAT_ID = "HERMES_SESSION_CHAT_ID"
PLATFORM_NAME = "uffici"

#: How often the catalog is asked for, at most.
REFRESH_SECONDS = 60.0

#: How long a ``tools.list`` may take. The turn's admission waits on it, so it is short.
LIST_DEADLINE_MS = 15_000

#: How long a ``tools.call`` may take: a platform tool can be slow, and the wire's
#: ceiling is five minutes.
CALL_DEADLINE_MS = 120_000

#: How long past the call's own deadline the waiting worker thread gives up.
THREAD_GRACE_SECONDS = 10.0

#: The wire's bounds (``tools.list.result.schema.json``, ``post.schema.json``).
TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
MAX_DESCRIPTION_CHARS = 16_000
MAX_CITATIONS = 20
MAX_ACTIVITY = 20
MAX_ACTIVITY_TOOL_CHARS = 64

#: Hermes's tool-search bridge (``tools/tool_search_catalog.py``): its own plumbing, not
#: something the agent did — a ``tool_call`` is reported under the tool it unwrapped.
BRIDGE_TOOL_NAMES = frozenset({"tool_search", "tool_describe", "tool_call"})

#: Appended to an ``ask`` tool's description, so the model knows before it calls.
ASK_SENTENCE = (
    "A person approves each call of this tool before it runs: the first call asks them, "
    "and you then tell the person and end your turn."
)

#: 🔴 What the model reads when a call cannot be made, each written TO the agent and each
#: asking it to tell the person (#1958 decision 32) — never a silent failure.
OUTSIDE_A_REQUEST = (
    "This Uffici tool cannot act outside a request: it works only while you are answering "
    "a Uffici conversation, and this call is not part of one, so it did not run. Tell the "
    "person it could not be done here."
)
NO_LONGER_AVAILABLE = (
    "This Uffici tool is no longer available to this agent, so it did not run. Tell the "
    "person it could not be done."
)
CANNOT_WAIT_HERE = (
    "This Uffici tool could not be run from here, so it did not run. Tell the person it "
    "could not be done, and try again shortly."
)

#: What a Hermes tool call is reported as on the post — generic by design: the tool's
#: name is the whole of what leaves, never its arguments or its result.
ACTIVITY_SUMMARY_OK = "Used this tool on the agent's own machine."
ACTIVITY_SUMMARY_FAILED = "Tried this tool on the agent's own machine; it did not succeed."

#: Every live instance, for the one process-wide post-tool hook.
_INSTANCES: weakref.WeakSet[PlatformTools] = weakref.WeakSet()


def session_conversation() -> str | None:
    """The Uffici conversation the CALLING thread's turn belongs to, or None.

    🔴 Read in the thread the tool runs in, where Hermes bound the turn's ContextVars —
    never from anything shared between turns (a remembered conversation, the newest open
    request): one Hermes runs several conversations' turns at once.
    """
    try:
        from gateway.session_context import get_session_env  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - the module exists inside Hermes
        return None
    if str(get_session_env(SESSION_PLATFORM, "") or "") != PLATFORM_NAME:
        return None
    return str(get_session_env(SESSION_CHAT_ID, "") or "") or None


def _refusal(message: str, error_class: str | None = None) -> str:
    """A result the model reads as an error: Hermes's own ``{"error": …}`` shape."""
    body: dict[str, Any] = {"error": message}
    if error_class:
        body["errorClass"] = error_class
    return json.dumps(body, ensure_ascii=False)


def catalog_entries(payload: Any) -> dict[str, dict[str, Any]] | None:
    """The ``tools.list`` result as ``{name: entry}``, or None when it is not one.

    An entry outside the wire's shape is skipped, never repaired: a schema this plugin
    guessed at would be one the platform never offered.
    """
    tools = payload.get("tools") if isinstance(payload, dict) else None
    if not isinstance(tools, list):
        return None
    entries: dict[str, dict[str, Any]] = {}
    skipped = 0
    for tool in tools:
        if not isinstance(tool, dict):
            skipped += 1
            continue
        name, description = tool.get("name"), tool.get("description")
        parameters, permission = tool.get("parameters"), tool.get("permission")
        if (
            not isinstance(name, str)
            or not TOOL_NAME_RE.match(name)
            or not isinstance(description, str)
            or len(description) > MAX_DESCRIPTION_CHARS
            or not isinstance(parameters, dict)
            or permission not in ("allow", "ask")
            or name in entries
        ):
            skipped += 1
            continue
        entries[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "permission": permission,
        }
    if skipped:
        logger.warning(
            "uffici: %d catalog entr%s outside the wire's shape were not offered",
            skipped,
            "y" if skipped == 1 else "ies",
            extra={"skipped": skipped},
        )
    return entries


def hermes_schema(entry: dict[str, Any]) -> dict[str, Any]:
    """The OpenAI function shape Hermes registers: the platform's own name, description
    and JSON Schema, with the approval sentence on an ``ask`` tool."""
    description = entry["description"].strip()
    if entry["permission"] == "ask":
        description = f"{description}\n\n{ASK_SENTENCE}" if description else ASK_SENTENCE
    return {"name": entry["name"], "description": description, "parameters": entry["parameters"]}


class PlatformTools:
    """The catalog as Hermes tools, the calls they make, and what a post cites.

    ``register_tool`` is Hermes's ``PluginContext.register_tool`` (None outside Hermes:
    nothing is registered). ``call`` is the call tunnel's ``CallClient.call``.
    ``request_for`` is the adapter's ledger: the request open on a conversation, read on
    the event loop. ``enabled`` says whether the attached companion carries the feature.
    """

    def __init__(
        self,
        register_tool: Callable[..., Any] | None,
        *,
        call: Callable[..., Awaitable[CallResult]],
        request_for: Callable[[str], str | None],
        enabled: Callable[[], bool],
        session_reader: Callable[[], str | None] = session_conversation,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._register_tool = register_tool
        self._call = call
        self._request_for = request_for
        self._enabled = enabled
        self._session_reader = session_reader
        self._clock = clock
        self._loop: asyncio.AbstractEventLoop | None = None
        #: name → the entry as registered, and its Hermes registration.
        self._catalog: dict[str, dict[str, Any]] = {}
        self._handles: dict[str, Any] = {}
        self._last_refresh: float | None = None
        self._refreshing: asyncio.Task[None] | None = None
        #: request id → what its next post carries.
        self._citations: dict[str, list[str]] = {}
        self._activity: dict[str, list[dict[str, str]]] = {}
        _INSTANCES.add(self)

    # ------------------------------------------------------------------ state

    @property
    def offered(self) -> frozenset[str]:
        """The tool names registered with Hermes right now."""
        return frozenset(self._catalog)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """The event loop the call tunnel runs on, where every call is made."""
        self._loop = loop

    def enabled(self) -> bool:
        return self._register_tool is not None and bool(self._enabled())

    # ---------------------------------------------------------------- refresh

    def refresh_due(self) -> bool:
        return self.enabled() and (
            self._last_refresh is None or self._clock() - self._last_refresh >= REFRESH_SECONDS
        )

    def refreshing(self) -> bool:
        return self._refreshing is not None and not self._refreshing.done()

    def kick(self, conversation_id: str | None) -> None:
        """Start a refresh when one is due, without waiting for it."""
        if not conversation_id or self.refreshing() or not self.refresh_due():
            return
        self._last_refresh = self._clock()
        self._refreshing = asyncio.ensure_future(self._refresh(conversation_id))

    async def wait(self) -> None:
        """Wait for the refresh in flight, if any. Never raises."""
        task = self._refreshing
        if task is None or task.done():
            return
        # The refresh logs its own failure; a turn never fails on it.
        with contextlib.suppress(Exception):
            await asyncio.shield(task)

    async def _refresh(self, conversation_id: str) -> None:
        try:
            answered = await self._call(
                "tools.list", {"conversationId": conversation_id}, deadline_ms=LIST_DEADLINE_MS
            )
        except Exception as exc:  # noqa: BLE001 — a refresh never fails a turn
            logger.warning(
                "uffici: the tool catalog could not be asked for (%s)", type(exc).__name__
            )
            return
        if not answered.ok:
            # 🔴 The tools already registered STAY: a failed read says nothing about the
            # catalog, and dropping every tool on one timeout would take them from a turn
            # that was about to use them. The next refresh asks again.
            logger.warning(
                "uffici: the tool catalog could not be read (%s); keeping the %d tool(s) "
                "already offered",
                answered.error_class,
                len(self._catalog),
                extra={"class": answered.error_class},
            )
            return
        entries = catalog_entries(answered.payload)
        if entries is None:
            logger.warning("uffici: the tool catalog was not a list; keeping what is offered")
            return
        self.apply(entries)

    def apply(self, entries: dict[str, dict[str, Any]]) -> None:
        """Make Hermes's registrations match the catalog: register, replace, dispose."""
        if self._register_tool is None:
            return
        added = removed = changed = 0
        for name in [n for n in self._catalog if n not in entries]:
            self._dispose(name)
            removed += 1
        for name, entry in entries.items():
            if self._catalog.get(name) == entry:
                continue
            if name in self._catalog:
                self._dispose(name)
                changed += 1
            else:
                added += 1
            schema = hermes_schema(entry)
            try:
                handle = self._register_tool(
                    name=name,
                    toolset=TOOLSET,
                    schema=schema,
                    handler=self._handler_for(name),
                    description=schema["description"],
                    emoji="🏢",
                )
            except Exception as exc:  # noqa: BLE001 — one refused tool never costs the rest
                logger.warning(
                    "uffici: Hermes refused the tool %s (%s)",
                    name,
                    type(exc).__name__,
                    extra={"tool": name},
                )
                continue
            if handle is None:
                # Hermes returns None when the name is already another toolset's.
                logger.warning(
                    "uffici: the tool %s was not offered — Hermes already has a tool of that name",
                    name,
                    extra={"tool": name},
                )
                continue
            self._catalog[name] = entry
            self._handles[name] = handle
        if added or removed or changed:
            logger.info(
                "uffici: offering %d Uffici tool(s) to Hermes (%d added, %d removed, %d changed)",
                len(self._catalog),
                added,
                removed,
                changed,
                extra={"tools": len(self._catalog)},
            )

    def _dispose(self, name: str) -> None:
        self._catalog.pop(name, None)
        handle = self._handles.pop(name, None)
        dispose = getattr(handle, "dispose", None)
        if not callable(dispose):
            return
        try:
            dispose()
        except Exception as exc:  # noqa: BLE001 — the handler refuses a tool that left
            logger.warning(
                "uffici: Hermes would not remove the tool %s (%s)",
                name,
                type(exc).__name__,
                extra={"tool": name},
            )

    # ------------------------------------------------------------------ calls

    def _handler_for(self, name: str) -> Callable[[dict[str, Any]], str]:
        def handle(args: dict[str, Any]) -> str:
            return self.run_from_thread(name, args)

        handle.__name__ = f"uffici_tool_{name}"
        return handle

    def run_from_thread(self, name: str, args: Any) -> str:
        """The handler Hermes dispatches, in the agent's worker thread."""
        # 🔴 HERE, in the calling thread, per call: the turn's session is this thread's
        # ContextVars, and two conversations' turns call at the same time.
        conversation_id = self._session_reader()
        loop = self._loop
        if loop is None or loop.is_closed():
            return _refusal(INSTRUCTIONS["unknown_op"], "unknown_op")
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            # Blocking the loop the answer has to arrive on would hang the gateway.
            logger.error(
                "uffici: a tool was dispatched on the event loop itself; refused",
                extra={"tool": name},
            )
            return _refusal(CANNOT_WAIT_HERE)
        future = asyncio.run_coroutine_threadsafe(self.invoke(conversation_id, name, args), loop)
        try:
            return future.result(timeout=CALL_DEADLINE_MS / 1000 + THREAD_GRACE_SECONDS)
        except TimeoutError:
            future.cancel()
            return _refusal(INSTRUCTIONS["timed_out"], "timed_out")

    async def invoke(self, conversation_id: str | None, name: str, args: Any) -> str:
        """One tool call on the request open on ``conversation_id``, on the event loop."""
        if name not in self._catalog:
            return _refusal(NO_LONGER_AVAILABLE)
        request_id = self._request_for(conversation_id) if conversation_id else None
        if not request_id:
            # 🔴 Refused HERE, without the tunnel: a call with no request would have to
            # guess one, and the platform would attribute it to whoever that turn served.
            logger.warning(
                "uffici: the tool %s was called outside a Uffici request; refused",
                name,
                extra={"tool": name},
            )
            return _refusal(OUTSIDE_A_REQUEST)
        if not self._enabled():
            return _refusal(INSTRUCTIONS["unknown_op"], "unknown_op")
        answered = await self._call(
            "tools.call",
            {
                "requestId": request_id,
                "name": name,
                "arguments": args if isinstance(args, dict) else {},
            },
            deadline_ms=CALL_DEADLINE_MS,
        )
        if not answered.ok:
            return _refusal(
                answered.instruction or INSTRUCTIONS["upstream_failed"], answered.error_class
            )
        payload = answered.payload or {}
        urn = payload.get("callUrn")
        # 🔴 The call's name is OPAQUE here: any non-empty string is cited as it came.
        # The companion validated it against the wire, and the platform's own codec is the
        # one place a name like it is ever parsed.
        if "result" not in payload or not (urn is None or (isinstance(urn, str) and urn)):
            return _refusal(INSTRUCTIONS["malformed_payload"], "malformed_payload")
        if urn is not None:
            cited = self._citations.setdefault(request_id, [])
            if urn not in cited and len(cited) < MAX_CITATIONS:
                cited.append(urn)
        logger.info(
            "uffici: the tool %s ran on request %s (%s)",
            name,
            request_id,
            "cited" if urn else "not run",
            extra={"tool": name, "requestId": request_id},
        )
        return json.dumps({"callUrn": urn, "result": payload["result"]}, ensure_ascii=False)

    # ------------------------------------------------------------------ the post

    def on_tool_called(self, tool_name: str, status: str | None) -> None:
        """Hermes's post-tool hook, in the agent's thread: one of HERMES'S tools ran."""
        name = str(tool_name or "")
        if not name or name in self._catalog or name in BRIDGE_TOOL_NAMES:
            return
        conversation_id = self._session_reader()
        loop = self._loop
        if not conversation_id or loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._record_activity, conversation_id, name, status)
        except RuntimeError:  # the loop closed between the check and the call
            return

    def _record_activity(self, conversation_id: str, name: str, status: str | None) -> None:
        request_id = self._request_for(conversation_id)
        if not request_id:
            return
        done = self._activity.setdefault(request_id, [])
        if len(done) >= MAX_ACTIVITY:
            return
        done.append(
            {
                "tool": name[:MAX_ACTIVITY_TOOL_CHARS],
                "summary": ACTIVITY_SUMMARY_FAILED if status == "error" else ACTIVITY_SUMMARY_OK,
            }
        )

    def take_extras(self, request_id: str) -> dict[str, Any]:
        """What the request's next post carries — its citations and activity — handed over
        once. Empty without the companion's ``tools`` feature: an older companion would
        refuse a field it does not know."""
        if not self._enabled():
            return {}
        extras: dict[str, Any] = {}
        if cited := self._citations.pop(request_id, None):
            extras["citations"] = cited
        if done := self._activity.pop(request_id, None):
            extras["activity"] = done
        return extras

    def give_back(self, request_id: str, extras: dict[str, Any]) -> None:
        """A post that did not go keeps what it carried for the next one."""
        if cited := extras.get("citations"):
            self._citations[request_id] = cited + self._citations.get(request_id, [])
        if done := extras.get("activity"):
            self._activity[request_id] = (done + self._activity.get(request_id, []))[:MAX_ACTIVITY]

    def forget(self, request_id: str) -> None:
        self._citations.pop(request_id, None)
        self._activity.pop(request_id, None)

    def forget_requests(self) -> None:
        self._citations.clear()
        self._activity.clear()


def on_post_tool_call(**kwargs: Any) -> None:
    """Hermes's ``post_tool_call`` hook (``model_tools._emit_post_tool_call_hook``).

    🔴 Only the tool's NAME and Hermes's own ok/error status are read. ``args`` and
    ``result`` arrive here too and are never looked at.
    """
    for tools in list(_INSTANCES):
        try:
            tools.on_tool_called(kwargs.get("tool_name"), kwargs.get("status"))
        except Exception:  # noqa: BLE001, S112 — an observer never fails a tool call
            continue
