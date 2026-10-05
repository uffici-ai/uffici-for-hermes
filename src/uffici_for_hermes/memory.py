"""The agent's Uffici memory inside Hermes: recall before each turn, and four tools.

An agent has no memory of its own here: it has the memory of WHERE it is working — the
organization, the folders the conversation sits in, and the conversation itself — held by
the platform and reached over the call tunnel as the ``memory.*`` ops. Two halves:

- **Prefetch.** Hermes's one ``memory.provider`` slot runs :class:`UfficiMemoryProvider`
  before every model call of a turn; it recalls with the turn's own text and Hermes fences
  the answer into ``<memory-context>`` on the user side. The provider does nothing else:
  no tools of its own, no system-prompt block (the brief's context already rides
  ``channel_prompt``), no turn sync and nothing before compression — the platform holds the
  transcript.
- **Tools.** ``uffici_recall``, ``uffici_remember``, ``uffici_forget`` and ``uffici_unfold``
  are PLUGIN tools in the ``uffici`` toolset, registered exactly like the platform's own
  tools (``platform_tools.py``) when the companion's hello announces ``memory``, and each
  call made on the request open on the conversation it was made in.

Hermes loads a memory provider through its OWN loader, which imports this package a second
time under another module name — so the provider shares no module state with the adapter
the gateway built. The two meet at ONE process-global rendezvous (:func:`rendezvous`): the
connected adapter publishes its prefetch there, and every provider copy reads it.

🔴 Every failure is an instruction the agent relays, never ``""`` — Hermes reads an empty
prefetch as "nothing remembered". A turn on another platform this Hermes serves gets
``""``: it is not a Uffici conversation.

🔴 Logged: op names, request ids, counts and error classes. Never a query, a fact or a hit.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import types
from collections.abc import Awaitable, Callable
from typing import Any

from .call_contract import INSTRUCTIONS
from .calls import CallResult
from .platform_tools import (
    OUTSIDE_A_REQUEST,
    TOOLSET,
    refusal,
    run_on_loop,
    session_conversation,
)

logger = logging.getLogger(__name__)

#: The companion's word for "the plugin may call the four memory ops", in its hello.
FEATURE_MEMORY = "memory"

#: The provider's name in Hermes's memory manager, and the label of its recall count.
PROVIDER_NAME = "uffici"
PROVIDER_LABEL = "Uffici"

#: A prefetch's whole budget. 🔴 Inside Hermes's own 8 s bound on an external provider
#: (``agent/memory_manager.py``, ``_EXTERNAL_PREFETCH_TIMEOUT_S``): the companion and the
#: platform each answer inside THIS, so the bound that fires is always one that renders an
#: instruction, never Hermes's, which renders nothing. Pinned against the real Hermes by
#: ``dev/hermes_contract.py``.
PREFETCH_DEADLINE_MS = 5_000
HERMES_PREFETCH_BOUND_SECONDS = 8.0
#: How long the prefetch thread waits for the call: the deadline, the call client's own
#: one-second grace for the companion's ``timed_out``, and a little for the trip.
PREFETCH_WAIT_SECONDS = PREFETCH_DEADLINE_MS / 1000 + 1.5

#: The tools' budgets. A recall the agent asks for is not under the prefetch bound; the
#: wire's ceiling on a recall's deadline is ten seconds.
TOOL_RECALL_DEADLINE_MS = 10_000
TOOL_DEADLINE_MS = 15_000
TOOL_WAIT_SECONDS = TOOL_DEADLINE_MS / 1000 + 5.0

#: The wire's bounds (``memory.recall.request.schema.json``): a query past this is cut to
#: its END, where a turn's newest text is.
MAX_QUERY_CHARS = 8_000
MAX_RECALL_LIMIT = 20

#: What a turn reads in place of recall when recall could not run — each written TO the
#: agent and asking it to tell the person (#1958 decision 32).
NOT_CONNECTED = (
    "Uffici memory could not be checked for this turn because this agent's companion is not "
    "connected; tell the person you could not check what was remembered."
)
COMPANION_WITHOUT_MEMORY = (
    "Uffici memory could not be checked for this turn because this agent's companion cannot "
    "reach memory; tell the person you could not check what was remembered, and that the "
    "companion needs updating."
)
NO_REQUEST = (
    "Uffici memory could not be checked for this turn because it is not answering a Uffici "
    "request; tell the person you could not check what was remembered."
)
BRIDGE_FAILED = (
    "Uffici memory could not be checked for this turn because of a system problem; tell the "
    "person you could not check what was remembered, and try again shortly."
)
UNREADABLE = (
    "Uffici memory answered in a shape this plugin does not understand; tell the person "
    "something went wrong, and that the plugin needs updating."
)

#: The heading the recalled lines sit under. Hermes adds the fence and its own note.
RECALL_HEADING = (
    "This organization's Uffici memory, recalled for this turn (closest level first; each "
    "line names its level and the fact's URN):"
)
DEGRADED_NOTE = (
    "Some of this was found by matching words rather than meaning; something relevant may "
    "be missing."
)
DEGRADED_EMPTY = (
    "This organization's Uffici memory could only be searched by matching words for this "
    "turn, and nothing matched; something relevant may have been missed."
)


_ART5 = (
    " Record facts about the work — decisions, preferences about how work is done, "
    "deadlines, where things are — never an appraisal, rating or judgement of a person."
)

#: The four tools, in the OpenAI function shape Hermes registers. 🔴 Each description says
#: whose memory this is and which URN types it accepts (#2048).
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "uffici_recall": {
        "name": "uffici_recall",
        "description": (
            "Search this organization's Uffici memory for facts relevant to a query: the "
            "organization, the folders this conversation sits in, and this conversation. "
            "Each hit names its level and its fact URN; the closest "
            "level wins when two say the same thing. Relevant memory is already recalled "
            "before each turn — use this to look for something else."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for."},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_RECALL_LIMIT,
                    "description": "At most this many facts (default 8).",
                },
            },
            "required": ["query"],
        },
    },
    "uffici_remember": {
        "name": "uffici_remember",
        "description": (
            "Remember ONE fact in this organization's Uffici memory, on this conversation. "
            "People see it in the conversation's Memory, and may keep, edit, promote or "
            "delete it." + _ART5 + " Answers the new fact's URN."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The fact, in one or two sentences."}
            },
            "required": ["text"],
        },
    },
    "uffici_forget": {
        "name": "uffici_forget",
        "description": (
            "Mark a fact you remembered in this conversation as no longer true, in this "
            "organization's Uffici memory. It is never deleted: it stays visible to people "
            "as superseded. Takes a fact URN, as recall and remember return it; give the URN "
            "of a newer fact you remembered in this conversation as replacementUrn when "
            "there is one. Facts from other levels, or written by people, cannot be "
            "forgotten by you."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                # 🔴 No pattern here: the companion checks the wire's own grammar, and the
                # platform's codec is the one spelling of a URN.
                "urn": {"type": "string", "description": "A fact URN."},
                "replacementUrn": {"type": "string", "description": "A newer fact URN."},
            },
            "required": ["urn"],
        },
    },
    "uffici_unfold": {
        "name": "uffici_unfold",
        "description": (
            "Show what a fact or a context statement in this organization's Uffici memory "
            "replaced, up to two steps back, and what replaced it. Takes a fact URN "
            "as recall returns it, or a context statement URN."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "urn": {"type": "string", "description": "A fact URN or a statement URN."},
                "depth": {"type": "integer", "minimum": 1, "maximum": 2},
            },
            "required": ["urn"],
        },
    },
}

#: Tool → the wire op it is forwarded as.
TOOL_OPS = {
    "uffici_recall": "memory.recall",
    "uffici_remember": "memory.remember",
    "uffici_forget": "memory.forget",
    "uffici_unfold": "memory.unfold",
}


# ------------------------------------------------------------------ drift

#: The provider's name in ``memory.provider``: the plugin's DIRECTORY name, which is how
#: Hermes's memory loader finds a provider under ``$HERMES_HOME/plugins/``.
PROVIDER_DIRECTORY = "uffici-for-hermes"

_FALSE_WORDS = frozenset({"false", "off", "0", "no", "n", ""})


def _is_on(value: Any) -> bool:
    """Hermes's reading of a switch: absent is ON (``is_truthy_value(…, default=True)``)."""
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in _FALSE_WORDS


def memory_drift(config: Any) -> list[str]:
    """What in Hermes's configuration keeps this agent from Uffici's memory, each as a
    sentence for the machine's log — empty when the setup's keys still hold."""
    memory = config.get("memory") if isinstance(config, dict) else None
    memory = memory if isinstance(memory, dict) else {}
    drift: list[str] = []
    provider = str(memory.get("provider") or "").strip()
    if provider != PROVIDER_DIRECTORY:
        drift.append(
            f"memory.provider is {provider or 'unset'}, not {PROVIDER_DIRECTORY}, so this "
            "agent's turns recall nothing from Uffici memory; run `uffici setup` again"
        )
    on = [key for key in ("memory_enabled", "user_profile_enabled") if _is_on(memory.get(key))]
    if on:
        drift.append(
            f"Hermes's own memory is on ({', '.join('memory.' + k for k in on)}), so this "
            "Hermes keeps notes of its own beside Uffici memory that no person in the "
            "organization can see or delete; run `uffici setup` again"
        )
    return drift


def report_memory_drift(load_config: Callable[[], Any] | None = None) -> list[str]:
    """Read Hermes's configuration and log every drift at WARNING. Never raises."""
    try:
        if load_config is None:
            from hermes_cli.config import load_config  # type: ignore[import-not-found]
        config = load_config()
    except Exception as exc:  # noqa: BLE001 — a probe that cannot read says so, once
        logger.info(
            "uffici: Hermes's memory configuration could not be read (%s)", type(exc).__name__
        )
        return []
    drift = memory_drift(config)
    for sentence in drift:
        logger.warning("uffici: %s", sentence)
    return drift


# ------------------------------------------------------------------ the rendezvous

#: 🔴 A fixed name in ``sys.modules``, NOT this package's: Hermes's memory loader imports
#: the package under a module name of its own, and that copy must find the adapter the
#: gateway built from the other one.
RENDEZVOUS_MODULE = "uffici_for_hermes_memory_rendezvous"


def rendezvous() -> types.ModuleType:
    """The process-global meeting point between the adapter and every provider copy."""
    module = sys.modules.get(RENDEZVOUS_MODULE)
    if module is None:
        module = sys.modules.setdefault(RENDEZVOUS_MODULE, types.ModuleType(RENDEZVOUS_MODULE))
    if not hasattr(module, "prefetch"):
        module.prefetch = None
    return module


def publish(prefetch: Callable[[str, str], tuple[str, int]]) -> None:
    """The connected adapter's prefetch, for whichever provider copy Hermes runs."""
    rendezvous().prefetch = prefetch


def withdraw(prefetch: Callable[[str, str], tuple[str, int]]) -> None:
    """Withdraw a prefetch — only if it is still the one published."""
    meeting = rendezvous()
    if meeting.prefetch == prefetch:
        meeting.prefetch = None


# ------------------------------------------------------------------ rendering


def _clip_query(query: str) -> str:
    text = str(query or "").strip()
    return text[-MAX_QUERY_CHARS:] if len(text) > MAX_QUERY_CHARS else text


def render_recall(payload: Any) -> tuple[str, int] | None:
    """A ``memory.recall`` result as the lines a turn reads, and how many facts — or None
    when it is not the wire's shape. ``("", 0)`` when nothing matched."""
    if not isinstance(payload, dict):
        return None
    hits, degraded = payload.get("hits"), payload.get("degraded")
    if not isinstance(hits, list) or not isinstance(degraded, bool):
        return None
    lines: list[str] = []
    for hit in hits:
        if not isinstance(hit, dict):
            return None
        level, urn, text = hit.get("level"), hit.get("urn"), hit.get("text")
        if not (isinstance(level, str) and isinstance(urn, str) and isinstance(text, str)):
            return None
        first, *rest = text.strip().splitlines() or [""]
        lines.append(f"- [{level}] {first} ({urn})")
        # A fact's later lines stay under its bullet, so Hermes's de-duplication of
        # repeated recall bullets never splits one fact.
        lines.extend(f"  {line}" for line in rest)
    flagged = degraded or bool(payload.get("missing"))
    if not lines:
        return (DEGRADED_EMPTY, 0) if flagged else ("", 0)
    body = "\n".join([RECALL_HEADING, *lines])
    if flagged:
        body = f"{body}\n{DEGRADED_NOTE}"
    return body, len(hits)


def _failure_text(answered: CallResult) -> str:
    """A failed recall as the sentence the turn reads, always naming Uffici memory."""
    instruction = answered.instruction or INSTRUCTIONS.get(
        answered.error_class or "", INSTRUCTIONS["upstream_failed"]
    )
    return f"Uffici memory could not be checked for this turn. {instruction}"


# ------------------------------------------------------------------ the tools and prefetch


class MemoryTools:
    """The four memory tools as Hermes tools, and the prefetch the provider calls.

    ``register_tool`` is Hermes's ``PluginContext.register_tool`` (None outside Hermes).
    ``call`` is the call tunnel's ``CallClient.call``; ``request_for`` the adapter's ledger;
    ``available`` whether the attached companion announced ``memory``; ``connected`` whether
    a companion is attached at all.
    """

    def __init__(
        self,
        register_tool: Callable[..., Any] | None,
        *,
        call: Callable[..., Awaitable[CallResult]],
        request_for: Callable[[str], str | None],
        available: Callable[[], bool],
        connected: Callable[[], bool],
        session_reader: Callable[[], str | None] = session_conversation,
    ) -> None:
        self._register_tool = register_tool
        self._call = call
        self._request_for = request_for
        self._available = available
        self._connected = connected
        self._session_reader = session_reader
        self._loop: asyncio.AbstractEventLoop | None = None
        self._handles: dict[str, Any] = {}

    @property
    def offered(self) -> frozenset[str]:
        return frozenset(self._handles)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # -------------------------------------------------------------- registration

    def sync(self) -> None:
        """Offer the four tools while the companion carries memory; withdraw them when not."""
        if self._register_tool is None:
            return
        if self._available():
            self._register_all()
        else:
            self._dispose_all()

    def _register_all(self) -> None:
        added = 0
        for name, schema in TOOL_SCHEMAS.items():
            if name in self._handles:
                continue
            try:
                handle = self._register_tool(
                    name=name,
                    toolset=TOOLSET,
                    schema=schema,
                    handler=self._handler_for(name),
                    description=schema["description"],
                    emoji="🧠",
                )
            except Exception as exc:  # noqa: BLE001 — one refused tool never costs the rest
                logger.warning(
                    "uffici: Hermes refused the memory tool %s (%s)",
                    name,
                    type(exc).__name__,
                    extra={"tool": name},
                )
                continue
            if handle is None:
                logger.warning(
                    "uffici: the memory tool %s was not offered — Hermes already has a tool "
                    "of that name",
                    name,
                    extra={"tool": name},
                )
                continue
            self._handles[name] = handle
            added += 1
        if added:
            logger.info("uffici: offering %d Uffici memory tool(s) to Hermes", len(self._handles))

    def _dispose_all(self) -> None:
        if not self._handles:
            return
        for name in list(self._handles):
            handle = self._handles.pop(name)
            dispose = getattr(handle, "dispose", None)
            if not callable(dispose):
                continue
            try:
                dispose()
            except Exception as exc:  # noqa: BLE001 — the handler refuses a tool that left
                logger.warning(
                    "uffici: Hermes would not remove the memory tool %s (%s)",
                    name,
                    type(exc).__name__,
                    extra={"tool": name},
                )
        logger.info("uffici: the companion carries no memory; the memory tools were withdrawn")

    # ------------------------------------------------------------------ calls

    def _handler_for(self, name: str) -> Callable[[dict[str, Any]], str]:
        def handle(args: dict[str, Any]) -> str:
            # 🔴 Read HERE, in the calling thread: the turn's session is its ContextVars.
            conversation_id = self._session_reader()
            return run_on_loop(
                self._loop,
                lambda: self.invoke(conversation_id, name, args),
                wait_seconds=TOOL_WAIT_SECONDS,
                tool=name,
            )

        handle.__name__ = f"uffici_memory_{name}"
        return handle

    async def invoke(self, conversation_id: str | None, name: str, args: Any) -> str:
        """One memory tool call on the request open on ``conversation_id``."""
        request_id = self._request_for(conversation_id) if conversation_id else None
        if not request_id:
            logger.warning(
                "uffici: the memory tool %s was called outside a Uffici request; refused",
                name,
                extra={"tool": name},
            )
            return refusal(OUTSIDE_A_REQUEST)
        if not self._available():
            return refusal(INSTRUCTIONS["unknown_op"], "unknown_op")
        payload = _op_payload(name, request_id, args if isinstance(args, dict) else {})
        if payload is None:
            return refusal(INSTRUCTIONS["malformed_payload"], "malformed_payload")
        deadline = TOOL_RECALL_DEADLINE_MS + 2_000 if name == "uffici_recall" else TOOL_DEADLINE_MS
        answered = await self._call(TOOL_OPS[name], payload, deadline_ms=deadline)
        if not answered.ok:
            logger.warning(
                "uffici: the memory tool %s failed on request %s (%s)",
                name,
                request_id,
                answered.error_class,
                extra={"tool": name, "requestId": request_id, "class": answered.error_class},
            )
            return refusal(
                answered.instruction or INSTRUCTIONS["upstream_failed"], answered.error_class
            )
        logger.info(
            "uffici: the memory tool %s ran on request %s",
            name,
            request_id,
            extra={"tool": name, "requestId": request_id},
        )
        return json.dumps(answered.payload or {}, ensure_ascii=False)

    # --------------------------------------------------------------- prefetch

    def prefetch(self, conversation_id: str, query: str) -> tuple[str, int]:
        """Recall for a turn, from Hermes's prefetch thread: the text the turn reads in
        ``<memory-context>`` and how many facts it carries. 🔴 Never ``""`` on a failure."""
        clipped = _clip_query(query)
        if not clipped:
            return "", 0
        if not self._connected():
            return NOT_CONNECTED, 0
        if not self._available():
            return COMPANION_WITHOUT_MEMORY, 0
        request_id = self._request_for(conversation_id)
        if not request_id:
            return NO_REQUEST, 0
        box: dict[str, tuple[str, int]] = {}

        async def recall() -> str:
            box["answer"] = await self._recall(request_id, clipped)
            return ""

        refused = run_on_loop(
            self._loop, recall, wait_seconds=PREFETCH_WAIT_SECONDS, tool="memory.recall"
        )
        if "answer" in box:
            return box["answer"]
        # The loop was gone, or the wait ran out: the refusal names the class.
        try:
            error_class = json.loads(refused).get("errorClass") or "timed_out"
        except ValueError:
            error_class = "timed_out"
        logger.warning(
            "uffici: recall did not answer for request %s (%s)",
            request_id,
            error_class,
            extra={"requestId": request_id, "class": error_class},
        )
        return _failure_text(
            CallResult(
                call_id="", error_class=error_class, instruction=INSTRUCTIONS.get(error_class)
            )
        ), 0

    async def _recall(self, request_id: str, query: str) -> tuple[str, int]:
        answered = await self._call(
            "memory.recall",
            {"requestId": request_id, "query": query, "deadlineMs": PREFETCH_DEADLINE_MS},
            deadline_ms=PREFETCH_DEADLINE_MS,
        )
        if not answered.ok:
            logger.warning(
                "uffici: recall failed for request %s (%s)",
                request_id,
                answered.error_class,
                extra={"requestId": request_id, "class": answered.error_class},
            )
            return _failure_text(answered), 0
        rendered = render_recall(answered.payload)
        if rendered is None:
            logger.warning(
                "uffici: recall answered outside the wire's shape for request %s",
                request_id,
                extra={"requestId": request_id},
            )
            return UNREADABLE, 0
        logger.info(
            "uffici: recalled %d fact(s) for request %s",
            rendered[1],
            request_id,
            extra={"requestId": request_id, "hits": rendered[1]},
        )
        return rendered


def _op_payload(name: str, request_id: str, args: dict[str, Any]) -> dict[str, Any] | None:
    """A tool's arguments as its op's payload — only the fields the wire has, never a guess.

    None when a required argument is missing; the companion validates the rest against the
    op's schema and answers ``malformed_payload`` itself.
    """
    payload: dict[str, Any] = {"requestId": request_id}
    if name == "uffici_recall":
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return None
        payload["query"] = _clip_query(query)
        if isinstance(args.get("limit"), int) and not isinstance(args.get("limit"), bool):
            payload["limit"] = args["limit"]
        payload["deadlineMs"] = TOOL_RECALL_DEADLINE_MS
    elif name == "uffici_remember":
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        payload["text"] = text.strip()
    else:
        urn = args.get("urn")
        if not isinstance(urn, str) or not urn:
            return None
        payload["urn"] = urn
        if name == "uffici_forget" and isinstance(args.get("replacementUrn"), str):
            payload["replacementUrn"] = args["replacementUrn"]
        if (
            name == "uffici_unfold"
            and isinstance(args.get("depth"), int)
            and not isinstance(args.get("depth"), bool)
        ):
            payload["depth"] = args["depth"]
    return payload


# ------------------------------------------------------------------ the provider


def build_provider() -> Any:
    """Hermes's ``MemoryProvider`` for Uffici. Built here, not at import: the base class
    exists only inside Hermes."""
    from agent.memory_provider import MemoryProvider, RecallStatus  # type: ignore[import-not-found]

    class UfficiMemoryProvider(MemoryProvider):
        """Prefetch only. Everything else Hermes may ask of a provider is deliberately empty."""

        def __init__(self) -> None:
            self._last_count: int | None = None

        @property
        def name(self) -> str:
            return PROVIDER_NAME

        def is_available(self) -> bool:
            # No I/O: Hermes asks this at agent start, and the companion may come later.
            return True

        def initialize(self, session_id: str, **kwargs: Any) -> None:
            return None

        def get_tool_schemas(self) -> list[dict[str, Any]]:
            # 🔴 The four tools are PLUGIN tools (``MemoryTools``), registered on the
            # companion's hello; a provider tool would exist without one.
            return []

        def system_prompt_block(self) -> str:
            # The brief's context already rides ``channel_prompt``.
            return ""

        def prefetch(self, query: str, *, session_id: str = "") -> str:
            self._last_count = None
            conversation_id = session_conversation()
            if not conversation_id:
                # Another platform's turn: not a Uffici conversation, nothing to recall.
                return ""
            bridge = rendezvous().prefetch
            if bridge is None:
                self._last_count = 0
                return NOT_CONNECTED
            try:
                text, count = bridge(conversation_id, query)
            except Exception as exc:  # noqa: BLE001 — 🔴 a raise here reads as "nothing remembered"
                logger.warning("uffici: recall failed in the bridge (%s)", type(exc).__name__)
                return BRIDGE_FAILED
            self._last_count = count if text else None
            return text

        def recall_status(self) -> Any:
            if not self._last_count:
                return None
            return RecallStatus(provider_label=PROVIDER_LABEL, count=self._last_count)

        def sync_turn(self, *_args: Any, **_kwargs: Any) -> None:
            # The platform holds the transcript.
            return None

        def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
            return ""

    return UfficiMemoryProvider()


def is_memory_collector(ctx: Any) -> bool:
    """Whether ``register`` was handed Hermes's memory-provider collector rather than a
    plugin context.

    🔴 The collector (``plugins/memory/__init__.py``, ``_ProviderCollector``) declares
    ``register_memory_provider`` on its class and forwards every OTHER ``register_*`` to a
    plugin context of its own — so a ``register`` that went on to register the platform
    from there would register it a second time, from a second copy of this package.
    ``PluginContext`` declares both. Pinned against the real Hermes by
    ``dev/hermes_contract.py``.
    """
    kind = type(ctx)
    return callable(getattr(kind, "register_memory_provider", None)) and not callable(
        getattr(kind, "register_platform", None)
    )
