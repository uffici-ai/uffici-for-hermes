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
        self, chat_id: str, chat_name: Optional[str] = None, chat_type: str = "dm",
        user_id: Optional[str] = None, user_name: Optional[str] = None, **_: Any,
    ) -> SessionSource:
        return SessionSource(
            platform=self.platform, chat_id=str(chat_id), chat_name=chat_name,
            chat_type=chat_type, user_id=str(user_id) if user_id else None, user_name=user_name,
        )

    async def send_private_notice(
        self, chat_id: str, user_id: Optional[str], content: str, reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Transcribed from base.py (0.21.5): the default is a normal send."""
        return await self.send(
            chat_id=chat_id, content=content, reply_to=reply_to, metadata=metadata
        )

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
        self, chat_id: str, content: str, reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult: ...

    @abstractmethod
    async def get_chat_info(self, chat_id: str) -> dict[str, Any]: ...

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        return None

    @classmethod
    def supports_exec_approval_buttons(cls) -> bool:
        return cls._send_exec_approval_prompt is not BasePlatformAdapter._send_exec_approval_prompt

    async def send_exec_approval(
        self, chat_id: str, command: str, session_key: str, description: str = "dangerous command",
        metadata: Optional[dict[str, Any]] = None, allow_permanent: bool = True,
        allow_session: bool = True, smart_denied: bool = False,
    ) -> SendResult:
        prompt = ExecApprovalPrompt(
            chat_id=chat_id, session_key=session_key, metadata=metadata, command=str(command or ""),
            description=description, smart_denied=smart_denied,
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
        self, session_key: str, choice: str, resolve_all: bool = False,
        reason: Optional[str] = None, request_id: Optional[str] = None,
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

    gateway.config, gateway.session, gateway.platforms = config, session, platforms
    platforms.base, platforms.event = base, event
    for name, module in {
        "gateway": gateway, "gateway.config": config, "gateway.session": session,
        "gateway.platforms": platforms, "gateway.platforms.base": base,
        "gateway.platforms.event": event, "tools": tools, "tools.approval": approval,
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
