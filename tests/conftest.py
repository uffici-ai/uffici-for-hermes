"""A stand-in for the Hermes package the plugin runs inside.

🔴 Why a stand-in and not the real thing. Hermes is not a dependency of this
package and must not become one: the plugin is installed INTO Hermes's own
sealed virtual environment, which already provides it. PyPI's newest
``hermes-agent`` is 0.19.0 while this plugin targets 0.21.5 (commit
``f97608f1``), so taking it as a dev dependency would compile these tests
against an API two minors away from the one the plugin is written for — worse
than a stand-in, because it would look like verification.

🔴 What keeps the stand-in honest. Every signature, default and field below is
transcribed from the pinned source, and the pieces that matter are transcribed
verbatim:

- ``BasePlatformAdapter`` (``gateway/platforms/base.py``): ``connect(*, is_reconnect=False)``,
  ``disconnect()``, ``send(chat_id, content, reply_to=None, metadata=None)``,
  ``get_chat_info(chat_id)`` are abstract; ``authorization_is_upstream`` and
  ``enforces_own_access_policy`` are PROPERTIES defaulting False;
  ``supports_code_blocks`` is a class attribute defaulting False;
  ``build_source(...)`` builds a ``SessionSource`` from ``self.platform``.
- ``MessageEvent`` (``gateway/platforms/event.py``): every field this plugin sets,
  with ``allow_gateway_control`` defaulting **True** and ``is_command()``
  requiring it.
- The exec-approval seam (``gateway/platforms/base.py``, ``tools/approval.py``):
  ``ExecApprovalPrompt``'s fields, ``send_exec_approval`` building one and handing it to
  ``_send_exec_approval_prompt`` (default: ``SendResult(success=False, error="Not supported")``),
  ``supports_exec_approval_buttons`` answering whether that hook is overridden — which is
  what makes the runner call it instead of posting a plain-text ``/approve`` prompt
  through ``send`` — and ``resolve_gateway_approval(session_key, choice, resolve_all=False,
  reason=None, request_id=None)``, which answers how many pending approvals it resolved.
- ``tools.clarify_gateway``: the pending-clarify store the ``clarify`` tool blocks on —
  see ``ClarifyGateway`` below for exactly which functions and behaviours.
- ``build_session_key`` (``gateway/session.py``): the dm-versus-group rule only —
  a ``dm`` with a non-empty ``chat_id`` sets ``isolate_user = not chat_id``, so
  the participant is left out and one conversation is one session however many
  people write in it. The real function's Slack, WhatsApp, Discord-thread and
  profile branches are not reproduced; this plugin reaches none of them.

The real API is verified against #2014's run on a real Hermes. Anything below
that turns out to be wrong is a defect in this file, and #2014 is where it shows.
"""

# 🔴 ``Optional[X]`` rather than ``X | None``, and the import order below, are
# TRANSCRIBED from the pinned Hermes source. Modernising them would make this
# mirror harder to diff against upstream when the API moves, which is the one
# job this file has.
# ruff: noqa: UP045, UP006, UP007
from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import sys
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import pytest

# --------------------------------------------------------------------------- gateway.config


class Platform(Enum):
    """The pinned enum accepts unknown values for registered plugins via
    ``_missing_``; the stand-in accepts any string, which is the same thing from
    this plugin's side."""

    LOCAL = "local"

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str) or not value.strip():
            return None
        value = value.strip().lower()
        pseudo = object.__new__(cls)
        pseudo._value_ = value
        pseudo._name_ = value.upper()
        cls._value2member_map_[value] = pseudo
        return pseudo


@dataclass
class PlatformConfig:
    enabled: bool = False
    token: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)
    #: ``gateway/config.py``: the per-platform gate on Hermes's restart/shutdown notice to
    #: active chats (``_restart_notification_allowed``), defaulting True.
    gateway_restart_notification: bool = True


# --------------------------------------------------------------------------- gateway.session


@dataclass
class SessionSource:
    platform: Any
    chat_id: str
    chat_name: Optional[str] = None
    chat_type: str = "dm"
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    thread_id: Optional[str] = None
    profile: Optional[str] = None


def build_session_key(source: SessionSource, group_sessions_per_user: bool = True) -> str:
    """The pinned rule, for the branches this plugin can reach."""
    is_dm = source.chat_type == "dm"
    isolate_user = (not source.chat_id) if is_dm else group_sessions_per_user
    platform = getattr(source.platform, "value", str(source.platform))
    parts = ["agent:main", platform, source.chat_type]
    if source.chat_id:
        parts.append(source.chat_id)
    if isolate_user and source.user_id:
        parts.append(str(source.user_id))
    return ":".join(parts)


# ----------------------------------------------------------------- gateway.platforms.event


class MessageType(Enum):
    TEXT = "text"
    COMMAND = "command"


class ProcessingOutcome(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


@dataclass
class MessageEvent:
    text: str
    message_type: MessageType = MessageType.TEXT
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    source: Any = None
    message_id: Optional[str] = None
    channel_prompt: Optional[str] = None
    channel_context: Optional[str] = None
    internal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    allow_gateway_control: bool = True

    def is_command(self) -> bool:
        return self.allow_gateway_control and (self.text or "").lstrip().startswith("/")


# --------------------------------------------------------------------------- gateway.platforms.base


@dataclass
class ExecApprovalPrompt:
    chat_id: str
    session_key: str
    text: str
    actions: list[tuple[str, str, str]]
    command: str
    description: str
    smart_denied: bool
    metadata: Optional[dict[str, Any]] = None


@dataclass
class SendResult:
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None


class BasePlatformAdapter(ABC):
    supports_code_blocks: bool = False
    supports_status_text: bool = False

    def __init__(self, config: PlatformConfig, platform: Any) -> None:
        self.config = config
        self.platform = platform
        self._session_store: Any = None
        self.handled: list[MessageEvent] = []

    @property
    def enforces_own_access_policy(self) -> bool:
        return False

    @property
    def authorization_is_upstream(self) -> bool:
        return False

    def set_session_store(self, session_store: Any) -> None:
        self._session_store = session_store

    def build_source(
        self,
        chat_id: str,
        chat_name: Optional[str] = None,
        chat_type: str = "dm",
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        **_: Any,
    ) -> SessionSource:
        return SessionSource(
            platform=self.platform,
            chat_id=str(chat_id),
            chat_name=chat_name,
            chat_type=chat_type,
            user_id=str(user_id) if user_id else None,
            user_name=user_name,
        )

    async def send_private_notice(
        self,
        chat_id: str,
        user_id: Optional[str],
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Transcribed from base.py (0.21.5): the default is a normal send."""
        return await self.send(
            chat_id=chat_id, content=content, reply_to=reply_to, metadata=metadata
        )

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: Optional[list],
        clarify_id: str,
        session_key: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        """Transcribed from base.py (v2026.9.24, :2852): the TEXT fallback — a numbered
        list, ``mark_awaiting_text`` so the gateway's text intercept takes the next
        message as the answer, and a plain ``send``."""
        if choices:
            numbered = [f"{i}. {choice}" for i, choice in enumerate(choices, start=1)]
            text = "\n".join([f"❓ {question}", "", *numbered, "", "Reply with the number."])
            CLARIFY.mark_awaiting_text(clarify_id)
        else:
            text = f"❓ {question}"
        return await self.send(chat_id=chat_id, content=text, metadata=metadata)

    async def handle_message(self, event: MessageEvent) -> bool:
        """The stand-in records and admits. Tests that need a refusal replace it."""
        self.handled.append(event)
        return True

    @abstractmethod
    async def connect(self, *, is_reconnect: bool = False) -> bool: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult: ...

    @abstractmethod
    async def get_chat_info(self, chat_id: str) -> dict[str, Any]: ...

    async def on_processing_start(self, event: MessageEvent) -> None:  # noqa: B027 — transcribed
        """Hook called when background processing begins."""

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        return None

    async def _run_processing_hook(self, hook_name: str, *args: Any, **kwargs: Any) -> None:
        """Transcribed from base.py (v2026.9.24, :3469); the real one logs a raise and
        swallows it."""
        hook = getattr(self, hook_name, None)
        if not callable(hook):
            return
        with contextlib.suppress(Exception):
            await hook(*args, **kwargs)

    @classmethod
    def supports_exec_approval_buttons(cls) -> bool:
        return cls._send_exec_approval_prompt is not BasePlatformAdapter._send_exec_approval_prompt

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[dict[str, Any]] = None,
        allow_permanent: bool = True,
        allow_session: bool = True,
        smart_denied: bool = False,
    ) -> SendResult:
        prompt = ExecApprovalPrompt(
            chat_id=chat_id,
            session_key=session_key,
            metadata=metadata,
            command=str(command or ""),
            description=description,
            smart_denied=smart_denied,
            text=f"Hermes wants to run a command that needs your OK: {command}",
            actions=[("Allow Once", "once", "primary"), ("Deny", "deny", "danger")],
        )
        return await self._send_exec_approval_prompt(prompt)

    async def _send_exec_approval_prompt(self, prompt: ExecApprovalPrompt) -> SendResult:
        return SendResult(success=False, error="Not supported")


# --------------------------------------------------------------------------- tools.approval


class ApprovalQueue:
    """``tools.approval``'s pending gateway approvals, as much as this plugin touches.

    A test enqueues what the agent thread would be waiting on; ``resolve_gateway_approval``
    records the choice and reason on each entry it resolves, as the real one does.
    """

    def __init__(self) -> None:
        self.pending: dict[str, list[dict[str, Any]]] = {}
        self.resolved: list[dict[str, Any]] = []

    def resolve_gateway_approval(
        self,
        session_key: str,
        choice: str,
        resolve_all: bool = False,
        reason: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> int:
        queue = self.pending.get(session_key) or []
        if not queue:
            return 0
        targets = list(queue) if resolve_all else [queue[0]]
        for entry in targets:
            queue.remove(entry)
            entry.update(choice=choice, reason=reason)
            self.resolved.append(entry)
        return len(targets)


APPROVALS = ApprovalQueue()


# ----------------------------------------------------------------------- tools.clarify_gateway


@dataclass
class ClarifyEntry:
    """``tools.clarify_gateway._ClarifyEntry``: one pending clarify inside a session."""

    clarify_id: str
    session_key: str
    question: str
    choices: Optional[list[str]]
    multi_select: bool = False
    response: Optional[str] = None
    resolved: bool = False
    awaiting_text: bool = False


class ClarifyGateway:
    """``tools.clarify_gateway`` (pinned v2026.9.24), as much as this plugin touches.

    Transcribed: ``register`` starts an open-ended entry (no choices) in text mode;
    ``resolve_gateway_clarify(clarify_id, response)`` answers False for an unknown or
    already-resolved id; ``resolve_text_response_for_session(session_key, response)``
    resolves the oldest pending entry of the session when the text is a choice's number
    or label, or any text in text mode (the real coercion's multi-select and
    ``(Recommended)`` branches are not reproduced); ``mark_awaiting_text`` flips an
    entry into text mode; ``has_pending``; and ``clear_session(session_key)`` drops every
    pending entry, answering each waiter ``""`` and returning how many it cancelled.
    ``calls`` records every call in order, so a test can say what happened BEFORE what.
    """

    def __init__(self) -> None:
        self.entries: dict[str, ClarifyEntry] = {}
        self.calls: list[tuple[str, ...]] = []

    def register(
        self,
        clarify_id: str,
        session_key: str,
        question: str,
        choices: Optional[list[str]],
        multi_select: bool = False,
    ) -> ClarifyEntry:
        entry = ClarifyEntry(
            clarify_id,
            session_key,
            question,
            list(choices) if choices else None,
            bool(multi_select) and bool(choices),
            awaiting_text=not bool(choices),
        )
        self.entries[clarify_id] = entry
        return entry

    def _pending(self, session_key: str) -> list[ClarifyEntry]:
        return [e for e in self.entries.values() if e.session_key == session_key and not e.resolved]

    def resolve_gateway_clarify(self, clarify_id: str, response: str) -> bool:
        self.calls.append(("resolve", clarify_id, response))
        entry = self.entries.get(clarify_id)
        if entry is None or entry.resolved:
            return False
        entry.response, entry.resolved = str(response), True
        return True

    def resolve_text_response_for_session(self, session_key: str, response: str) -> bool:
        self.calls.append(("resolve_text", session_key, response))
        pending = self._pending(session_key)
        if not pending:
            return False
        entry, text = pending[0], str(response).strip()
        value: Optional[str] = None
        if not entry.choices or entry.awaiting_text:
            value = text
        elif text.isdigit() and 0 < int(text) <= len(entry.choices):
            value = entry.choices[int(text) - 1]
        else:
            value = next((c for c in entry.choices if c.casefold() == text.casefold()), None)
        if value is None:
            return False
        entry.response, entry.resolved = value, True
        return True

    def mark_awaiting_text(self, clarify_id: str) -> bool:
        self.calls.append(("mark_awaiting_text", clarify_id))
        entry = self.entries.get(clarify_id)
        if entry is not None:
            entry.awaiting_text = True
        return entry is not None

    def has_pending(self, session_key: str) -> bool:
        return bool(self._pending(session_key))

    def clear_session(self, session_key: str) -> int:
        self.calls.append(("clear_session", session_key))
        cancelled = 0
        for clarify_id, entry in list(self.entries.items()):
            if entry.session_key != session_key:
                continue
            del self.entries[clarify_id]
            if not entry.resolved:
                entry.response, entry.resolved = "", True
                cancelled += 1
        return cancelled


CLARIFY = ClarifyGateway()


# ----------------------------------------------------------------- gateway.session_context

#: TRANSCRIBED (v2026.9.24 ``gateway/session_context.py``): the turn's routing ids are
#: task-local ContextVars, inherited by the ``run_in_executor`` thread a turn's tools run
#: in, and ``get_session_env`` reads one by its legacy ``HERMES_SESSION_*`` name — the
#: ContextVar when it was ever set here, else ``os.environ``, else the default. The gateway
#: binds ``HERMES_SESSION_PLATFORM`` to the source's platform value and
#: ``HERMES_SESSION_CHAT_ID`` to its ``chat_id`` (``gateway/run.py`` ``_set_session_env``).
#: Only the two names this plugin reads are reproduced.
_SESSION_UNSET: Any = object()
_SESSION_VARS = {
    name: contextvars.ContextVar(name, default=_SESSION_UNSET)
    for name in ("HERMES_SESSION_PLATFORM", "HERMES_SESSION_CHAT_ID")
}


def get_session_env(name: str, default: str = "") -> str:
    var = _SESSION_VARS.get(name)
    if var is not None and (value := var.get()) is not _SESSION_UNSET:
        return value
    return os.getenv(name, default)


def set_session_vars(platform: str = "", chat_id: str = "", **_: Any) -> None:
    """What the gateway does for a turn, in the calling context."""
    _SESSION_VARS["HERMES_SESSION_PLATFORM"].set(platform)
    _SESSION_VARS["HERMES_SESSION_CHAT_ID"].set(chat_id)


# ----------------------------------------------------------------- hermes_logging

#: TRANSCRIBED (v2026.9.24 ``hermes_logging.setup_logging`` and ``gateway/run_turn.py``):
#: Hermes logs through Python ``logging``; ``setup_logging`` lowers the ROOT level to the
#: config's ``logging.level`` (INFO by default) and never raises it; and a turn logs on the
#: logger named ``gateway.run``, which propagates to the root. The plugin's log shipping
#: (#2265) rests on all three. ``dev/hermes_contract.py`` checks the real ones, inside the
#: e2e runs' Hermes.
HERMES_TURN_LOGGER = "gateway.run"
HERMES_DEFAULT_ROOT_LEVEL = logging.INFO

# ----------------------------------------------------------------- gateway.run_turn_followup_ack


def _followup_processing_hooks_apply(adapter: Any, event: Optional[MessageEvent]) -> bool:
    """TRANSCRIBED VERBATIM (v2026.9.24 ``gateway/run_turn_followup_ack.py``, byte-identical
    in 0.21.5): whether a message Hermes parked behind a running turn and drained
    IN-BAND as a queued follow-up gets its lifecycle hooks at all (#2246).

    🔴 Only for an adapter that OVERRIDES ``on_processing_start`` — Hermes brackets the two, so
    an adapter defining the completion hook alone is never handed a follow-up's completion.
    ``dev/hermes_contract.py`` checks the real one, inside the e2e runs' Hermes."""
    if adapter is None or event is None:
        return False
    if not (getattr(event, "message_id", None) or getattr(event, "raw_message", None)):
        return False
    start_hook = getattr(type(adapter), "on_processing_start", None)
    return start_hook is not None and start_hook is not BasePlatformAdapter.on_processing_start


async def _run_followup_processing_hook(
    adapter: Any, event: Optional[MessageEvent], hook_name: str, *args: Any
) -> None:
    """TRANSCRIBED VERBATIM: fire one lifecycle hook for a runner-drained follow-up."""
    if not _followup_processing_hooks_apply(adapter, event):
        return
    run_hook = getattr(adapter, "_run_processing_hook", None)
    if not callable(run_hook):
        return
    await run_hook(hook_name, event, *args)


# ----------------------------------------------------------------- agent.memory_provider


class MemoryProvider(ABC):
    """TRANSCRIBED (v2026.9.24 ``agent/memory_provider.py``): the four abstract members a
    provider must define, and the defaults of the hooks this plugin overrides. Hermes's
    memory manager runs ``prefetch`` on a thread of its own, bounded by
    ``_EXTERNAL_PREFETCH_TIMEOUT_S`` (8.0 s), and fences a non-empty answer into
    ``<memory-context>``. ``dev/hermes_contract.py`` checks the real ones."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def initialize(self, session_id: str, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_tool_schemas(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""

    def recall_status(self) -> Any:
        return None

    def sync_turn(self, user_content: str, assistant_content: str, **_: Any) -> None:
        return None

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        return ""


@dataclass(frozen=True)
class RecallStatus:
    """TRANSCRIBED: what the last prefetch injected, for Hermes's recall indicator."""

    provider_label: str
    count: int
    glyph: str = "🧠"


class ProviderCollector:
    """TRANSCRIBED in shape (``plugins/memory/__init__.py``, ``_ProviderCollector``): it
    declares ``register_memory_provider`` and forwards every other ``register_*`` through
    ``__getattr__`` — recorded here, so a test sees a second platform registration."""

    def __init__(self) -> None:
        self.provider: Any = None
        self.forwarded: list[str] = []

    def register_memory_provider(self, provider: Any) -> None:
        self.provider = provider

    def __getattr__(self, attr: str) -> Any:
        if not attr.startswith("register_"):
            raise AttributeError(attr)
        return lambda *_a, **_k: self.forwarded.append(attr)


# --------------------------------------------------------------------------- installation


def _install() -> None:
    """Put the stand-in on ``sys.modules`` before the plugin imports ``gateway``."""
    gateway = types.ModuleType("gateway")
    config = types.ModuleType("gateway.config")
    config.Platform, config.PlatformConfig = Platform, PlatformConfig
    session = types.ModuleType("gateway.session")
    session.SessionSource, session.build_session_key = SessionSource, build_session_key
    platforms = types.ModuleType("gateway.platforms")
    base = types.ModuleType("gateway.platforms.base")
    base.BasePlatformAdapter, base.SendResult = BasePlatformAdapter, SendResult
    base.ExecApprovalPrompt = ExecApprovalPrompt
    event = types.ModuleType("gateway.platforms.event")
    event.MessageEvent, event.MessageType = MessageEvent, MessageType
    event.ProcessingOutcome = ProcessingOutcome

    tools = types.ModuleType("tools")
    approval = types.ModuleType("tools.approval")
    approval.resolve_gateway_approval = APPROVALS.resolve_gateway_approval
    tools.approval = approval
    clarify = types.ModuleType("tools.clarify_gateway")
    for name in (
        "resolve_gateway_clarify",
        "resolve_text_response_for_session",
        "mark_awaiting_text",
        "has_pending",
        "clear_session",
    ):
        setattr(clarify, name, getattr(CLARIFY, name))
    tools.clarify_gateway = clarify

    followup = types.ModuleType("gateway.run_turn_followup_ack")
    followup._followup_processing_hooks_apply = _followup_processing_hooks_apply
    followup._run_followup_processing_hook = _run_followup_processing_hook

    session_context = types.ModuleType("gateway.session_context")
    session_context.get_session_env = get_session_env
    session_context.set_session_vars = set_session_vars

    agent = types.ModuleType("agent")
    memory_provider = types.ModuleType("agent.memory_provider")
    memory_provider.MemoryProvider, memory_provider.RecallStatus = MemoryProvider, RecallStatus
    agent.memory_provider = memory_provider

    gateway.config, gateway.session, gateway.platforms = config, session, platforms
    gateway.session_context = session_context
    gateway.run_turn_followup_ack = followup
    platforms.base, platforms.event = base, event
    for name, module in {
        "gateway": gateway,
        "gateway.config": config,
        "gateway.session": session,
        "gateway.session_context": session_context,
        "gateway.run_turn_followup_ack": followup,
        "gateway.platforms": platforms,
        "gateway.platforms.base": base,
        "gateway.platforms.event": event,
        "tools": tools,
        "agent": agent,
        "agent.memory_provider": memory_provider,
        "tools.approval": approval,
        "tools.clarify_gateway": clarify,
    }.items():
        sys.modules.setdefault(name, module)


_install()


@pytest.fixture
def platform_config() -> PlatformConfig:
    return PlatformConfig(enabled=True)


@pytest.fixture(autouse=True)
def _empty_approval_queue():
    """The approval stand-in is a module global: every test starts with nothing pending."""
    APPROVALS.pending.clear()
    APPROVALS.resolved.clear()
    CLARIFY.entries.clear()
    CLARIFY.calls.clear()
    yield


@pytest.fixture(autouse=True)
def _restore_adapter_timings():
    """🔴 Put the adapter's timing globals back after every test.

    A test that shortens `DONE_SETTLE_SECONDS` or `READY_POLL_SECONDS` and does
    not restore it changes the behaviour of every test that runs afterwards —
    and it does so INVISIBLY, because the later test still passes. Exactly that
    happened here: a leaked `DONE_SETTLE_SECONDS = 0` made a settling task
    complete instantly, so a test asserting that disconnect CANCELS it passed
    with the cancellation deleted.

    Autouse, because the next person to shorten a timer will not remember.

    🔴 The import is LOCAL, and must stay local: this file installs the fake
    Hermes modules into `sys.modules` at import time, and the adapter reads
    Hermes symbols when IT is imported. A module-level import here would run
    before the stand-ins exist. (The test files may hoist theirs — they are
    imported after this one has finished its work.)
    """
    import uffici_for_hermes.adapter as adapter_module

    saved = {
        name: getattr(adapter_module, name)
        for name in (
            "DONE_SETTLE_SECONDS",
            "EMPTY_SETTLE_SECONDS",
            "READY_POLL_SECONDS",
            "READY_TIMEOUT_SECONDS",
            "BUSY_RECHECK_SECONDS",
            "BUSY_MAX_SECONDS",
        )
    }
    yield
    for name, value in saved.items():
        setattr(adapter_module, name, value)


@pytest.fixture(autouse=True)
def _forget_the_shared_log_handler():
    """🔴 The plugin's log handler is one per process, installed on the ROOT logger at
    plugin load (#2313): a test that loads the plugin leaves it there, capturing every
    later test's lines into one queue. Every test starts and ends without it.

    The import is local for the reason `_restore_adapter_timings` gives.
    """
    from uffici_for_hermes import log_shipping

    log_shipping.forget_shared_handler()
    yield
    log_shipping.forget_shared_handler()
