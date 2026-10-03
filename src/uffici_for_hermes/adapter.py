"""The Uffici platform adapter for Hermes.

A format translator and nothing else: a brief becomes one inbound message, each
send becomes a post, the turn's end becomes a done, a cancel becomes Hermes's
own ``/stop``. It holds no policy and no memory. The agent's Uffici tools it
offers Hermes are the platform's catalog, and every call goes back through the
companion (``platform_tools.py``) — the companion holds every connection and
every secret, which is what keeps this file small enough to read.

🔴 Authorization is DELEGATED, never absent. The companion authenticates the
local wire with a token before a frame is delivered, and the platform decided
which agent is a participant before it rang. Hermes's default-deny allowlist
would therefore be asking the wrong question, so this adapter declares
``authorization_is_upstream`` — the same declaration, for the same reason, that
the bundled a2a adapter makes. It never sets an allow-all environment variable
and never carries an allowlist of its own.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import build_session_key

from . import log_shipping
from .budget import BudgetUnavailable, context_budget_tokens, resolve_share, resolve_window_tokens
from .calls import CallClient
from .context import render_context
from .hint import PLATFORM_HINT
from .log_shipping import LogFlusher, UfficiLogHandler, install, shared_handler
from .platform_tools import FEATURE_TOOLS, PlatformTools, on_post_tool_call
from .render import cause_author, render_entries, render_reason
from .wire_client import (
    WireClient,
    WireConfigError,
    describe_frame,
    env_reader,
    resolve_address,
    resolve_token,
)

logger = logging.getLogger(__name__)

#: ``off`` keeps Hermes's log lines and the plugin's on this machine (#2265).
LOG_SHIPPING_ENV = "UFFICI_LOG_SHIPPING"

#: The platform's name, in `config.yaml`, in the registry and on the wire.
PLATFORM_NAME = "uffici"

#: How long the window probe may take — Hermes's resolver may ask a provider (#2230).
WINDOW_PROBE_TIMEOUT_SECONDS = 60.0

#: What a plugin that was not installed from a release says it is (#2236).
DEV_VERSION = "0.0.0-dev"

#: The release manifest, which the release writes beside this package with the tag's
#: version stamped in (`.github/workflows/release.yml`, "Package").
MANIFEST_NAME = "uffici-plugin.json"

#: The wire's bound on `pluginVersion` (`hello.schema.json`).
_VERSION_RE = re.compile(r"^\S{1,64}$")


def plugin_version(package_dir: Path | None = None) -> str:
    """This plugin's installed version, sent in ``hello`` (#2236).

    🔴 Read from the release's own manifest, never a constant: every release reported
    ``0.1.0`` while ``v0.3.0`` was installed, and an "out of date" notice built on that
    would have been wrong for every agent. A source checkout has no stamped manifest
    beside the package — the tracked one says ``0.0.0`` and sits at the project root —
    and says so as :data:`DEV_VERSION`.
    """
    directory = package_dir if package_dir is not None else Path(__file__).resolve().parent
    try:
        stamped = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DEV_VERSION
    version = str(stamped.get("version") or "") if isinstance(stamped, dict) else ""
    if version in ("", "0.0.0") or not _VERSION_RE.match(version):
        return DEV_VERSION
    return version

#: 🔴 Every conversation is a ``dm``. Hermes keys a ``dm`` with a ``chat_id`` on
#: that id alone, so one Uffici conversation is one Hermes session however many
#: people and agents write in it — which is what the transcript already is. A
#: ``group`` would key per user and split one conversation into a session per
#: participant, each seeing a different half of it.
CHAT_TYPE = "dm"

#: How long to wait after the harness says a turn is over before telling the
#: platform, restarted by any reply that arrives in the meantime.
#:
#: 🔴 Because `on_processing_complete` is NOT a reliable "the turn is over"
#: signal. Hermes's own log says so — "Normal final-send NOT suppressed despite
#: active stream consumer" — and measured against a real harness the final reply
#: landed about two seconds AFTER the completion hook had already fired. Closing
#: on the hook alone therefore threw away the answer to the first turn of every
#: session: the companion recorded `empty`, and the real reply arrived for a
#: request that no longer existed.
#:
#: The cost is that the platform learns a turn ENDED a couple of seconds late.
#: It does not delay a single reply — posts go out the moment they are produced
#: — so what it actually delays is the offset advancing. Losing a member's
#: answer is much worse than that.
DONE_SETTLE_SECONDS = 2.5

#: The same window, for a turn that has produced NOTHING yet.
#:
#: 🔴 Asymmetric on purpose. The window above is sized for "the replies have
#: stopped" — once one has arrived, the turn is demonstrably producing and each
#: post restarts the clock. The risk of closing early is highest in the opposite
#: case, where nothing has arrived and there is no evidence either way, and that
#: is the case 2.5s got wrong: on a COLD harness — CI, where Hermes is still
#: registering tools and probing model metadata when the first brief lands — the
#: first turn's reply came about three seconds after the completion hook. The
#: plugin closed at 2.5s as `empty`, forgot the request, and Hermes's answer was
#: then dropped with "no open request for conversation …". A member's answer,
#: lost, and reported to the platform as a turn that said nothing.
#:
#: The cost of being wrong the other way is that a genuinely empty turn's CLOSE
#: is late by this much — the offset advances a few seconds later, and no reply
#: is delayed by it at all. That is the cheaper mistake, by a wide margin.
EMPTY_SETTLE_SECONDS = 15.0

#: Hermes's reasons on a session it FORCE-interrupted because a drain timed out
#: (``gateway/run_shutdown.py``, ``_mark_running_sessions_resume_pending``).
#:
#: 🔴 #2306. Hermes marks every running session ``resume_pending`` before it drains,
#: and clears the mark from each one that finished once the drain completes; when the
#: drain times out it marks again and interrupts. So by the time the wire goes, a
#: session still carrying one of these reasons is a turn the restart CUT, never one it
#: let finish (#2300) — and that turn is still owed: the companion redelivers it.
#: ``restart_interrupted`` is Hermes's crash recovery at the next boot, not a drain,
#: and never reaches a disconnect. Pinned against the real Hermes by
#: ``dev/hermes_contract.py``.
SHUTDOWN_INTERRUPT_REASONS = frozenset({"restart_timeout", "shutdown_timeout"})

#: Hermes's replies for a turn its agent NEVER ran (``gateway/run.py``,
#: ``_normalize_empty_agent_response``): a drain or a ``/stop`` interrupted it before
#: the first model call returned, and Hermes tells the person to send the message again.
#:
#: 🔴 #2306. On Uffici the platform holds the message and delivers it again itself, so
#: the sentence is wrong to the person — the #2247 reasoning, for the notice Hermes
#: sends through ``send`` instead of ``send_private_notice`` — and it is not an
#: answer: posting it made a cut turn look ``answered``. It is never posted, and a turn
#: that produced only this closes as ``error``, not ``empty``, so the platform does not
#: count the message as handled. Exact strings, pinned against the real Hermes by
#: ``dev/hermes_contract.py`` so a reworded release fails the probe rather than this
#: list silently matching nothing.
HERMES_UNPROCESSED_REPLIES = frozenset({
    "⚠️ Your message was interrupted before processing started "
    "(likely by a recent /stop). Please send it again.",
    "⚠️ Your message wasn't processed (the previous turn was still "
    "being cleaned up). Please send it again.",
})

#: How often a close that found Hermes still working on the turn looks again.
#:
#: 🔴 The settle window above is a guess about when replies STOP; it cannot see a turn
#: that has not STARTED. `on_processing_complete` fires when Hermes's message handler
#: returns, and on some paths that is before the agent has run at all — during the
#: startup restore the message is queued and the handler returns at once, and a message
#: for a busy session is merged into the running turn (#2164). So once the window has
#: elapsed, the close also waits for Hermes itself to say the session is idle
#: (`_harness_busy`). Bounded by the platform, not here: the request's deadline expires
#: it, and the companion's cancel ends the wait.
BUSY_RECHECK_SECONDS = 1.0

#: The longest a close waits on Hermes's busy state before closing anyway.
#:
#: 🔴 A backstop, not the bound anyone should meet: the platform's deadline expires
#: the request long before this and the companion's cancel ends the wait. It exists
#: for the case nothing else covers — a Hermes internal left set (a stale guard, a
#: restore flag that never cleared) would otherwise hold the request and its maps
#: for the life of the process.
BUSY_MAX_SECONDS = 900.0

#: How often to check whether the harness has become able to answer.
READY_POLL_SECONDS = 0.25

#: How long to wait for that before saying so. Generous: a cold harness loading
#: a model legitimately takes a while, and the companion holds its rings the
#: whole time — a late attach costs a slow first turn, a premature one costs the
#: turn itself.
READY_TIMEOUT_SECONDS = 180.0

#: The wire's own bound on a post's text (`post.schema.json`).
#: 🔴 The platform bounded the INBOUND brief to the budget this plugin declared;
#: it did not bound the harness's reply. A frame past this is refused by the
#: companion's validating decoder as `malformed_frame`, which closes the
#: connection — so an over-long reply would cost the whole conversation.
POST_TEXT_MAX = 131072

#: How many closed request ids to remember, so a late lifecycle hook can be
#: told from a request that never existed. Bounded: this is a daemon.
CLOSED_MEMORY = 256

#: What the agent is told when Hermes asks a person to approve a command (#2164).
#:
#: 🔴 Wave 1 has no approval surface: the tool call and its approval arrive in stage 3
#: (#2125, #2128). Hermes relays this text inside the BLOCKED result the model reads,
#: so it is written TO the agent, and it asks for an answer rather than silence — the
#: person must always learn something (#1958 decision 32).
APPROVAL_UNAVAILABLE_REASON = (
    "Uffici cannot ask a person to approve a command yet, so nobody was asked and it did "
    "not run. Answer the person now with what you can do without it, and say that you "
    "could not run it here"
)

#: The header the opener is placed under, so the model can tell recalled history
#: from what was said this turn.
OPENER_HEADER = "[Uffici] Earlier in this conversation:"

#: The companion's word for "a post may carry a question" (#2232), in its hello.
FEATURE_QUESTION = "question"

#: The companion's word for "a restarted harness process is handed again every request
#: its predecessor had accepted" (#2234, #2273), in its hello.
FEATURE_REDELIVERY = "redelivery"

#: How long Hermes's startup resume waits for the companion's hello before deciding.
HELLO_WAIT_SECONDS = 10.0

#: What the wire admits for a question (`post.schema.json`): an id the answer can name,
#: at most ten choices of at most 200 characters. A question outside these bounds is
#: asked in plain text instead — never clipped, which would offer the person choices
#: the harness did not.
QUESTION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,200}$")
QUESTION_MAX_CHOICES = 10
QUESTION_CHOICE_MAX_CHARS = 200

#: Who can answer a question: a person, never another agent or the platform.
PERSON_AUTHORS = frozenset({"member", "correspondent"})


class UfficiAdapter(BasePlatformAdapter):
    """One conversation service, many conversations, over one local wire."""

    supports_code_blocks = True

    def __init__(
        self, config: PlatformConfig, *, get_env=None, connect_fn=None, plugin_ctx=None
    ) -> None:
        super().__init__(config, Platform(PLATFORM_NAME))
        # 🔴 Hermes's own operational notices ("No home channel is set …") go to
        # `send_private_notice`, never `send` (#2102). With the default `public`
        # they arrive through `send` exactly like a reply, and became the agent's
        # post to a person. Set unless the operator chose otherwise.
        extra = getattr(config, "extra", None)
        if extra is None:
            # No `extra` at all is the case that matters most: without the key
            # Hermes reads `public` and #2102 is back.
            extra = {}
            config.extra = extra
        if isinstance(extra, dict):
            extra.setdefault("notice_delivery", "private")
        # 🔴 Hermes's restart and shutdown notice ("⚠️ Hermes is restarting — your current task
        # will be interrupted … send any message …") goes to every active chat through `send`, and
        # landed in the conversation as the agent's own message (#2247) — telling the person
        # to do what the platform already does, since it redelivers the request itself
        # (#2234). Hermes gates it per platform, on this very config object (the runner
        # hands its own `PlatformConfig` to the adapter). Forced, not defaulted: there is no
        # reading of that notice that is right on Uffici, and Hermes's loader leaves no way
        # to tell an operator's `true` from its own default.
        config.gateway_restart_notification = False
        self._get_env = get_env or env_reader()
        self._connect_fn = connect_fn
        self._client: WireClient | None = None
        self._runner: Any = None
        self._ready_runner: Any = None

        #: conversation id → the request currently open on it. A send with no
        #: entry here has nowhere to go and is dropped rather than guessed at.
        self._open_request: dict[str, str] = {}
        #: request id → conversation id, for the frames that name only a request.
        self._conversation_of: dict[str, str] = {}
        #: request id → how many posts it has produced, which decides `answered`
        #: versus `empty` when the turn ends.
        self._posts: dict[str, int] = {}
        #: requests waiting for their opener before the event is handed over.
        self._awaiting_opener: dict[str, dict[str, Any]] = {}
        #: request id → Hermes's session key for its conversation, which is what the
        #: busy probe asks about.
        self._session_key_of: dict[str, str] = {}
        #: requests whose close is waiting for the replies to stop.
        self._settling: dict[str, Any] = {}
        #: the outcome the harness reported for each settling request, so a
        #: restart does not lose it.
        self._settling_outcome: dict[str, ProcessingOutcome] = {}
        #: requests whose turn Hermes reported as never run (#2306) — see
        #: HERMES_UNPROCESSED_REPLIES. Bounded like the other per-request maps: forgotten
        #: with the request.
        self._unprocessed: set[str] = set()
        #: requests this adapter has already closed — by a cancel, or by a
        #: refusal after the fresh ack. 🔴 A second terminal frame for one of
        #: these would be a transition the companion's state machine does not
        #: have, and `on_processing_complete` firing after a cancel would report
        #: `empty` for a turn that actually answered (its post count is gone).
        # 🔴 A dict, not a set, for its INSERTION ORDER. `set.pop()` removes an
        # arbitrary element — as likely the id just added as the oldest — so the
        # bound held while the guarantee did not (#2024). The companion's mirror
        # of this memory is a FIFO ring, and `docs/local-wire.md` describes the
        # two as the same mechanism.
        self._closed: dict[str, None] = {}
        #: The call tunnel (#2028): platform operations through the companion,
        #: matched to their results by callId — the log push (#2265) and the
        #: agent's tools (#2350) ride it.
        self._calls = CallClient(self._send)
        #: What the attached companion can carry beyond every v1 companion (#2232).
        self._companion_features: frozenset[str] = frozenset()
        #: Whether a companion hello has arrived, so a decision that reads the features
        #: waits for them rather than reading an empty set (#2273).
        self._hello_seen: Any = None
        #: conversation id → the question a turn asked through the card and is waiting
        #: on: Hermes's clarify id, its session key, and the request that asked (#2232).
        self._pending_questions: dict[str, dict[str, str]] = {}
        #: request that asked → request whose brief carried the answer. The turn that
        #: asked keeps running after the answer, so its completion closes the LATER one.
        self._continues: dict[str, str] = {}
        #: Hermes's log lines and this plugin's, for the companion to ship (#2265). The
        #: process's one handler (#2313), already capturing since the plugin loaded, so
        #: its queue survives a reconnect and a second adapter; none at all when
        #: UFFICI_LOG_SHIPPING=off.
        self._log_handler: UfficiLogHandler | None = (
            shared_handler() if log_shipping_on(self._get_env) else None
        )
        self._log_flusher: Any = None
        #: The flusher the task above runs, kept so a disconnect can finish it (#2313).
        self._log_flusher_obj: LogFlusher | None = None
        #: The agent's Uffici tools as Hermes tools (#2350): registered through the plugin's
        #: own context, each call forwarded on the request open on the conversation the call
        #: was made in. Inert without the companion's ``tools`` feature.
        self._tools = PlatformTools(
            getattr(plugin_ctx, "register_tool", None),
            call=self._calls.call,
            request_for=self._open_request.get,
            enabled=lambda: FEATURE_TOOLS in self._companion_features,
        )
        #: The last conversation a brief named — what the catalog is asked for on `ready`.
        #: Kept across reconnects on purpose: the catalog is the agent's, and a reconnect is
        #: exactly when it may have moved.
        self._known_conversation: str | None = None
        #: requests whose admission waits on the catalog refresh in flight (#2350).
        self._admitting: set[str] = set()

    # ---------------------------------------------------------------- policy

    @property
    def authorization_is_upstream(self) -> bool:
        """🔴 See the module docstring. Delegated, not absent."""
        return True

    # ---------------------------------------------------------------- lifecycle

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        # 🔴 Hermes calls this again on its own reconnect path. Without stopping
        # the previous client first, two loops read the same wire and every
        # brief is handled twice — two admits, two acks — while the orphaned
        # task reconnects for the life of the process.
        await self.disconnect()
        import asyncio

        try:
            address = resolve_address(self._get_env)
            token = resolve_token(self._get_env)
            # 🔴 Off the event loop: Hermes's resolver may ask a provider over HTTP
            # (#2230), and this coroutine runs on the gateway's own loop.
            try:
                window = await asyncio.wait_for(
                    asyncio.to_thread(resolve_window_tokens, self._get_env, self._window_probes()),
                    WINDOW_PROBE_TIMEOUT_SECONDS,
                )
            except TimeoutError as exc:
                # Hermes's resolver can ask a provider; one that hangs costs a retry of the
                # connect, never a gateway start that waits for ever.
                raise BudgetUnavailable(
                    "Hermes did not report its context window in time; the next connect "
                    "asks again"
                ) from exc
            budget = context_budget_tokens(window, resolve_share(self._get_env))
        except (WireConfigError, BudgetUnavailable) as exc:
            # The message names what is missing and how to supply it, and carries
            # no path and no token: these refusals are read from a log.
            logger.error("uffici: cannot start — %s", exc)
            return False

        self._client = WireClient(
            address=address,
            token=token,
            harness_version=self._harness_version(),
            plugin_version=plugin_version(),
            budget_tokens=budget,
            connect_fn=self._connect_fn or _default_connect,
            on_frame=self._on_frame,
            on_disconnected=self._calls.fail_all,
            # 🔴 Readiness is announced per CONNECTION, not once per process.
            # The companion forgets an attached plugin when the connection ends,
            # so a reconnect has to say `ready` again — and the frame can only
            # go out once there is a connection to carry it, which is what this
            # callback means.
            on_connected=self._on_companion_hello,
        )
        logger.info(
            "uffici: connecting over %s, budget %d token(s), reconnect=%s",
            address.describe(),
            budget,
            is_reconnect,
            extra={
                "transport": address.describe(),
                "budgetTokens": budget,
                "isReconnect": is_reconnect,
            },
        )
        self._runner = _spawn(self._client.run())
        self._tools.bind_loop(asyncio.get_running_loop())
        if self._log_handler is not None:
            # 🔴 On the ROOT logger, never changing its level; the flusher pushes only
            # while no call is in flight.
            install(self._log_handler)
            self._log_flusher_obj = LogFlusher(self._log_handler, self._calls)
            self._log_flusher = _spawn(self._log_flusher_obj.run())
        return True

    async def _on_companion_hello(self, frame: dict[str, Any]) -> None:
        """The wire is up. Announce readiness once the harness can answer.

        🔴 On its own task, because this runs inside the client's read loop —
        waiting here would stop the plugin reading the very frames it is about
        to ask for.
        """
        features = frame.get("features") if isinstance(frame, dict) else None
        self._companion_features = frozenset(
            str(name) for name in (features or []) if isinstance(name, str)
        )
        self._hello_event().set()
        runner, self._ready_runner = self._ready_runner, None
        if runner is not None:
            runner.cancel()
        self._ready_runner = _spawn(self._announce_ready_when_able())

    @property
    def calls(self) -> CallClient:
        """The call tunnel: one platform operation per call, through the companion."""
        return self._calls

    def harness_is_ready(self) -> bool:
        """Whether Hermes has wired this adapter up enough to answer.

        🔴 The message handler is the real signal, and Hermes says so itself:
        without one, "every inbound message is discarded" on an adapter that is
        otherwise connected and can send. The session store matters too — the
        fresh-versus-resumed decision reads it — but its absence costs a wrong
        `fresh`, not a lost turn, so it is not part of the gate.

        🔴 And not while the runner is restoring its sessions at boot (#2164): a
        message that arrives then is QUEUED, reported complete at once, and run
        later — the first ring of every real-provider run closed `empty` that way
        while its answer was still coming. Held here, the companion keeps the ring.
        """
        if getattr(self, "_message_handler", None) is None:
            return False
        runner = getattr(self, "gateway_runner", None)
        return not bool(getattr(runner, "_startup_restore_in_progress", False))

    async def _announce_ready_when_able(self) -> None:
        """Wait for the harness, then tell the companion to start sending."""
        import asyncio

        waited = 0.0
        while not self.harness_is_ready():
            if waited >= READY_TIMEOUT_SECONDS:
                logger.error(
                    "uffici: the harness did not become able to answer within %.0fs, so no "
                    "brief will be delivered. The platform is holding this agent's turns",
                    READY_TIMEOUT_SECONDS,
                )
                return
            await asyncio.sleep(READY_POLL_SECONDS)
            waited += READY_POLL_SECONDS

        logger.info("uffici: the harness is ready after %.1fs; asking for work", waited)
        # 🔴 The result is CHECKED. A `ready` that fails to send leaves the
        # companion holding every ring for the life of this connection — there
        # is no retry on the far side and nothing else will say it — and the
        # line above has already claimed we asked. Saying so loudly is the least
        # of it; dropping the connection is what actually recovers, because the
        # reconnect announces readiness again.
        if not await self._send(
            {"v": 1, "type": "ready", "harnessVersion": self._harness_version()}
        ):
            logger.error(
                "uffici: the ready frame could not be sent, so the platform is holding "
                "every turn for this agent. Dropping the connection to try again"
            )
            client = self._client
            if client is not None:
                await client.stop()
            return
        # The catalog, once the companion is taking work — for the conversation last seen,
        # so a reconnect's first turn does not wait on it (#2350).
        self._tools.kick(self._known_conversation)

    async def handle_message(self, event: MessageEvent) -> Any:
        """Every event Hermes hands this platform, briefs and Hermes's own alike.

        🔴 Hermes's startup auto-resume is declined when the companion redelivers
        (#2273). On boot Hermes runs an empty internal turn for every session a restart
        interrupted, and the companion hands the restarted process the same requests
        again (#2234) — two answers to one message, the first a reply to Hermes's note
        about the restart. The companion's record is the platform's, so its brief is the
        one answered; Hermes keeps a session it did not resume ``resume_pending`` and
        puts its interruption note into that brief's turn. An older companion does not
        redeliver, and there Hermes's resume is the only thing that finishes the work.
        """
        if self._is_startup_resume(event) and await self._companion_redelivers():
            logger.info(
                "uffici: declined Hermes's startup resume for conversation %s; the "
                "companion redelivers the request it interrupted",
                str(getattr(event.source, "chat_id", "") or ""),
            )
            return None
        return await super().handle_message(event)

    def _hello_event(self) -> Any:
        import asyncio

        if self._hello_seen is None:
            self._hello_seen = asyncio.Event()
        return self._hello_seen

    async def _companion_redelivers(self) -> bool:
        """Whether the companion hands a restarted harness its requests again.

        🔴 Waits for the companion's hello, briefly: Hermes resumes on boot, and this
        plugin connects to the companion on the same boot, so the resume can arrive
        before the features do — and an empty set would admit it, answering twice.
        No hello within the wait is treated as an older companion and keeps Hermes's
        resume: a companion slower than that to connect after a reboot answers twice,
        which is the lesser failure — never the turn left unanswered.
        """
        import asyncio

        try:
            await asyncio.wait_for(self._hello_event().wait(), HELLO_WAIT_SECONDS)
        except TimeoutError:
            return False
        return FEATURE_REDELIVERY in self._companion_features

    @staticmethod
    def _is_startup_resume(event: MessageEvent) -> bool:
        """Hermes's synthesized resume: internal, empty, and no brief of ours."""
        return (
            bool(getattr(event, "internal", False))
            and getattr(event, "message_type", None) is MessageType.TEXT
            and not str(getattr(event, "text", "") or "").strip()
            and not (getattr(event, "metadata", None) or {}).get("uffici_request_id")
        )

    async def disconnect(self) -> None:
        # What a companion can carry is that companion's: the next one says so again.
        self._companion_features = frozenset()
        self._hello_seen = None
        # 🔴 A question waiting on the card holds its Hermes turn (#2232 review): forgetting
        # it here without releasing it would leave that turn blocked, holding the session,
        # until Hermes's own clarify timeout.
        for pending in list(self._pending_questions.values()):
            self._release_questions(pending["sessionKey"])
        # Before the wire goes: a turn the harness let finish is not answered twice (#2300).
        await self._close_finished_turns()
        # And the lines that say why it is going (#2313).
        await self._flush_logs_once()
        if self._client is not None:
            await self._client.stop()
        self._client = None
        for name in ("_runner", "_ready_runner"):
            runner = getattr(self, name, None)
            setattr(self, name, None)
            if runner is not None:
                runner.cancel()
        self._forget_everything()

    async def _flush_logs_once(self) -> None:
        """Finish the flusher: its push in flight, then what is queued, bounded (#2313).

        🔴 The handler stays installed: it is the process's, and the lines logged
        between this disconnect and the next connect queue for that one. A push the
        bound cuts puts its batch back — kept, never lost uncounted — and the stop is
        never held past ``FINAL_FLUSH_SECONDS``. Hermes's own reconnect runs through
        here too (``connect`` disconnects first), deliberately: it costs that bound at
        most, and on a dead wire the push fails at once.
        """
        import asyncio

        flusher, task = self._log_flusher_obj, self._log_flusher
        self._log_flusher_obj = self._log_flusher = None
        if flusher is None:
            return
        try:
            await asyncio.wait_for(flusher.finish(task), log_shipping.FINAL_FLUSH_SECONDS)
        except Exception:  # noqa: BLE001 — a stop never fails on its logs; the lines stay queued
            return

    def _forget_everything(self) -> None:
        """Drop every per-request map.

        🔴 On a DISCONNECT. Not on re-attach, deliberately: the companion
        redelivers what it never got an ack for, and a plugin that wiped its maps
        on re-attach would drop the reply to a turn the harness is still working
        on. The disconnect is the moment both sides agree nothing is owed — the
        companion's own `Detach` releases its side there too.
        """
        self._open_request.clear()
        self._conversation_of.clear()
        self._session_key_of.clear()
        self._posts.clear()
        self._awaiting_opener.clear()
        self._closed.clear()
        for task in self._settling.values():
            task.cancel()
        self._settling.clear()
        self._settling_outcome.clear()
        self._unprocessed.clear()
        self._pending_questions.clear()
        self._continues.clear()
        self._admitting.clear()
        self._tools.forget_requests()
        self._calls.fail_all()

    def _harness_version(self) -> str:
        """Hermes's version, or a stated unknown — never a guessed number."""
        try:  # pragma: no cover - the import exists only inside Hermes
            from hermes_cli import __version__ as version  # type: ignore[import-not-found]

            return str(version)
        except Exception:  # noqa: BLE001 — a version we cannot read is not a reason to refuse
            return "unknown"

    def _window_probes(self):
        """Ways to learn Hermes's context window, best first.

        🔴 Each reaches into the harness's internals, which move between
        versions; ``resolve_window_tokens`` treats a raising probe as a
        fallback. What must never happen is this plugin inventing a window.
        """

        def from_hermes_config() -> int | None:
            """🔴 The window Hermes itself compacts against (#2230).

            Hermes's OWN configuration names the model, and its own resolver answers
            for it — the same call its agent makes at start. The two probes below read
            the model from the gateway's session-store config, which carries none, so
            on a real install neither ever asked and the plugin refused to start.
            """
            from agent.model_metadata import get_model_context_length  # type: ignore
            from hermes_cli import config as hermes_config  # type: ignore

            # The read-only loader where Hermes has one (it skips a deep copy): this
            # probe only reads, and never mutates what it is handed.
            load = getattr(hermes_config, "load_config_readonly", None) or hermes_config.load_config
            section = (load() or {}).get("model")
            if isinstance(section, str):
                model, section = section, {}
            elif isinstance(section, dict):
                model = section.get("default") or section.get("model")
            else:
                return None
            if not model:
                return None
            configured = section.get("context_length")
            try:
                configured = int(configured) if configured is not None else None
            except (TypeError, ValueError):
                # Hermes warns and auto-detects on the same value; so does this.
                configured = None
            return get_model_context_length(
                str(model),
                base_url=str(section.get("base_url") or ""),
                config_context_length=configured,
                provider=str(section.get("provider") or ""),
            )

        def from_session_store() -> int | None:
            store = getattr(self, "_session_store", None)
            config = getattr(store, "config", None)
            for name in ("context_length", "context_window", "max_context_tokens"):
                value = getattr(config, name, None)
                if isinstance(value, int) and value > 0:
                    return value
            return None

        def from_agent_defaults() -> int | None:
            from agent.model_metadata import get_cached_context_length  # type: ignore

            store = getattr(self, "_session_store", None)
            config = getattr(store, "config", None)
            model = getattr(config, "model", None)
            base_url = getattr(config, "base_url", "") or ""
            return get_cached_context_length(model, base_url) if model else None

        return (from_hermes_config, from_session_store, from_agent_defaults)

    # ---------------------------------------------------------------- outbound

    async def send_private_notice(
        self,
        chat_id: str,
        user_id: str | None,
        content: str,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        """A Hermes platform notice is NOT posted (#2102).

        Hermes addresses these to an operator at a terminal — setup hints,
        slash commands to type — and this adapter has no private channel to a
        person: every post it makes is the agent speaking in a conversation.
        So the notice is logged by length only and dropped, and SUCCESS is
        answered, because a failure makes Hermes fall back to a public `send`.
        """
        logger.info(
            "uffici: a Hermes platform notice was not posted (%d chars)",
            len(content or ""),
            extra={"conversationId": str(chat_id)},
        )
        return SendResult(success=True)

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: list[str] | None,
        clarify_id: str,
        session_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        """Hermes's ``clarify`` becomes the platform's question card (#2232).

        🔴 Posted with the question, then the request is SETTLED at once: the person sees
        the card instead of a working line, and the request's deadline no longer runs
        against a turn that is waiting on them. The turn that asked stays blocked in
        Hermes's clarify; the person's answer arrives as the next brief and resolves it
        (``_answer_pending_question``), and that turn's replies then go to the request
        which carried the answer.

        Hermes's own text fallback — a numbered list and its text intercept — is used
        when the companion cannot carry a question (an older one), when the question does
        not fit the wire's bounds, or when nothing is open to ask on.
        """
        conversation_id = str(chat_id)
        request_id = self._open_request.get(conversation_id)
        card = _question_card(question, choices, clarify_id, self._is_multi_select(clarify_id))
        if (
            request_id is None
            or card is None
            or FEATURE_QUESTION not in self._companion_features
        ):
            return await BasePlatformAdapter.send_clarify(
                self, chat_id, question, choices, clarify_id, session_key, metadata
            )

        text, offered = card
        if not await self._send(
            {"v": 1, "type": "post", "requestId": request_id, "text": text, "question": offered}
        ):
            # A failure sends Hermes to its plain-text retry, which is the right fallback.
            return SendResult(success=False, error="the wire dropped the question")
        self._posts[request_id] = self._posts.get(request_id, 0) + 1
        self._pending_questions[conversation_id] = {
            "clarifyId": str(clarify_id),
            "sessionKey": str(session_key or self._session_key_of.get(request_id, "")),
            "askedBy": request_id,
        }
        await self._send({"v": 1, "type": "done", "requestId": request_id, "outcome": "answered"})
        # 🔴 The request is over; the TURN is not — it waits on the answer and speaks on
        # the request that brings it. Pruning the continuation into this request here cut
        # the chain the turn's end follows, so a turn that asked twice never closed its
        # last request (#2296).
        self._forget(request_id, keep_chain=True)
        logger.info(
            "uffici: request %s asked a question on the card (%d choice(s)) and was settled",
            request_id, len(offered["choices"]),
            extra={"requestId": request_id, "conversationId": conversation_id},
        )
        return SendResult(success=True, message_id=request_id)

    async def retire_clarify_card(self, clarify_id: str, notice: str | None = None) -> None:
        """Hermes gave up on a question without an answer — its timeout, a reset (#2232).

        The pending question is forgotten, so the person's next message reaches the
        agent as an ordinary turn. The notice is Hermes's own text and is not posted.
        """
        for conversation_id, pending in list(self._pending_questions.items()):
            if pending.get("clarifyId") == str(clarify_id):
                del self._pending_questions[conversation_id]
                logger.info(
                    "uffici: a question on conversation %s ended without an answer",
                    conversation_id, extra={"conversationId": conversation_id},
                )

    def _is_multi_select(self, clarify_id: str) -> bool:
        """Whether Hermes asked for several choices — read where its base adapter reads it."""
        try:
            from tools import clarify_gateway  # type: ignore[import-not-found]

            with clarify_gateway._lock:  # noqa: SLF001 — Hermes's own base adapter reads it so
                entry = clarify_gateway._entries.get(clarify_id)  # noqa: SLF001
            return bool(getattr(entry, "multi_select", False))
        except Exception:  # noqa: BLE001 — a moved internal asks a single-choice question
            return False

    async def send_or_update_status(
        self,
        chat_id: str,
        status_key: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        """A Hermes status line is NOT posted (#2233).

        Hermes routes its own progress lines here when the adapter has this hook, and
        through plain ``send`` otherwise — which made every "⏳ still waiting" line a post
        by the agent, every fifteen seconds, and let those posts count as the agent
        answering, so the platform's "did not answer" note never fired. A status is Hermes
        talking about itself; the person's view of a working agent is the platform's own
        working line. Logged by key and length only, and SUCCESS is answered, because a
        failure would send Hermes back to ``send``.
        """
        logger.debug(
            "uffici: a Hermes status line was not posted (%s, %d chars)",
            str(status_key), len(content or ""),
            extra={"conversationId": str(chat_id)},
        )
        return SendResult(success=True)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        """One reply becomes one post on that conversation's open request."""
        request_id = self._open_request.get(str(chat_id))
        if request_id is None:
            # 🔴 Dropped, never guessed. Posting to another request would put
            # this reply in a turn it does not belong to, and the platform would
            # attribute it to whoever that turn was answering.
            # 🔴 The id is in the MESSAGE, not only in `extra`. A harness's own
            # formatter decides whether `extra` is rendered at all — Hermes's
            # does not render it — so a line carrying its identifying detail
            # only there is a line nobody can act on. Still classified: an id,
            # never the text that was going to be sent.
            logger.warning(
                "uffici: no open request for conversation %s; dropping a send (open: %s)",
                str(chat_id),
                ",".join(sorted(self._open_request)) or "none",
                extra={"conversationId": str(chat_id)},
            )
            return SendResult(success=False, error="no open request")

        text = content or ""
        if not text.strip():
            # 🔴 `post.schema.json` requires at least one character. An empty
            # reply is a real thing for a harness to produce, and emitting it
            # would be a frame the companion refuses — taking the connection
            # with it. It closes as `empty` instead, which is an outcome the
            # platform understands.
            logger.info(
                "uffici: an empty reply was not posted", extra={"requestId": request_id}
            )
            return SendResult(success=False, error="empty reply")
        if text.strip() in HERMES_UNPROCESSED_REPLIES:
            # 🔴 #2306: Hermes's "send it again" for a turn its agent never ran. Not an
            # answer and not true here — the platform delivers the message again. Logged
            # by length only, like every other text this plugin declines to post.
            self._unprocessed.add(request_id)
            logger.info(
                "uffici: Hermes's never-processed reply was not posted (%d chars)",
                len(text), extra={"requestId": request_id},
            )
            return SendResult(success=True, message_id=request_id)
        if len(text) > POST_TEXT_MAX:
            # Clipped rather than refused: a truncated answer reaches the person,
            # and a refused frame reaches nobody.
            logger.warning(
                "uffici: a reply was clipped to the wire's bound",
                extra={"requestId": request_id, "textChars": len(text)},
            )
            text = text[:POST_TEXT_MAX]

        frame: dict[str, Any] = {"v": 1, "type": "post", "requestId": request_id, "text": text}
        # The turn's own tool calls this answer rests on, and what it did with Hermes's
        # tools (#2350) — only to a companion that carries them.
        extras = self._tools.take_extras(request_id)
        frame.update(extras)
        sent = await self._send(frame)
        if not sent:
            self._tools.give_back(request_id, extras)
            return SendResult(success=False, error="the wire dropped the post")
        self._posts[request_id] = self._posts.get(request_id, 0) + 1

        # 🔴 A reply during the settle means the turn is still producing, so the
        # window restarts. Without this, a harness that streams several messages
        # around the completion hook would have the tail cut off.
        settling = self._settling.get(request_id)
        if settling is not None:
            settling.cancel()
            outcome = self._settling_outcome.get(request_id, ProcessingOutcome.SUCCESS)
            self._schedule_done(request_id, outcome)
        return SendResult(success=True, message_id=request_id)

    async def _send_exec_approval_prompt(self, prompt: Any) -> SendResult:
        """Hermes wants a person to approve a command: deny it at once, with a reason.

        🔴 Overriding this hook is what makes Hermes route an approval HERE. Without it the
        runner posts its plain-text ``/approve`` prompt through ``send`` — as the agent's
        message, to a person who cannot type ``/approve`` into a transcript — and the turn
        then waits on nobody until the request expires. Measured on the first
        real-provider run (#2164): the smart-approval guardian escalates on every call
        against a reasoning model, and that swallowed the answer to "can you take a look?".

        SUCCESS is answered whether or not anything was pending: a failure sends Hermes to
        that same plain-text fallback. The command and its description are never logged —
        they are the model's text and can carry a credential.
        """
        try:
            from tools.approval import resolve_gateway_approval  # type: ignore[import-not-found]
        except ImportError:  # pragma: no cover - the module exists inside Hermes
            logger.error("uffici: cannot answer an approval request; the harness has no resolver")
            return SendResult(success=True)
        session_key = str(getattr(prompt, "session_key", "") or "")
        if not session_key:
            # The runner's context had no key; the conversation's own request knows it.
            request_id = self._open_request.get(str(getattr(prompt, "chat_id", "") or ""))
            session_key = self._session_key_of.get(request_id or "", "")
        denied = resolve_gateway_approval(
            session_key,
            "deny",
            resolve_all=True,
            reason=APPROVAL_UNAVAILABLE_REASON,
        )
        logger.warning(
            "uffici: denied %d approval request(s) for conversation %s — this platform has "
            "no approval channel yet, and the agent was told so",
            denied,
            str(getattr(prompt, "chat_id", "") or ""),
            extra={"conversationId": str(getattr(prompt, "chat_id", "") or "")},
        )
        return SendResult(success=True)

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        return {"id": str(chat_id), "type": CHAT_TYPE, "platform": PLATFORM_NAME}

    async def on_processing_start(self, event: MessageEvent) -> None:
        """Nothing to do when a turn starts — but the override itself is load-bearing.

        🔴 A brief that lands on a BUSY session (Hermes's own startup auto-resume after a
        restart, a goal continuation, a cron wake-up) is parked and drained IN-BAND as a
        queued follow-up, and that drain fires the lifecycle hooks only for an adapter
        that overrides THIS hook (``gateway/run_turn_followup_ack.py``,
        ``_followup_processing_hooks_apply`` — Hermes brackets start and complete). Without
        it the follow-up's completion never reached ``on_processing_complete``, and an
        answered request sat open until the platform's deadline cancelled it (#2246).
        """

    async def on_processing_complete(
        self, event: MessageEvent, outcome: ProcessingOutcome
    ) -> None:
        """The turn ended: close its request.

        Posts went out as they were produced, so this frame only says HOW it
        ended. ``empty`` is a real outcome and not a failure — an agent that had
        nothing to add said so, and the platform advances its offset either way.
        """
        request_id = str((event.metadata or {}).get("uffici_request_id") or "")
        if not request_id:
            return
        # 🔴 A turn that asked a question on the card settled its own request and was
        # answered on a later one (#2232): its end closes THAT request.
        while request_id in self._continues:
            request_id = self._continues.pop(request_id)
        if request_id in self._closed:
            # Already terminal here: a cancel closed it, or the harness refused
            # it after the ack. The companion has moved on.
            return

        # 🔴 The close SETTLES rather than firing here. See DONE_SETTLE_SECONDS:
        # a reply can still arrive after this hook, and closing now throws it
        # away.
        self._schedule_done(request_id, outcome)

    def _schedule_done(self, request_id: str, outcome: ProcessingOutcome) -> None:
        """Close the request once the replies have actually stopped."""
        existing = self._settling.pop(request_id, None)
        if existing is not None:
            existing.cancel()
        self._settling_outcome[request_id] = outcome
        self._settling[request_id] = _spawn(self._close_when_settled(request_id, outcome))

    def _settle_window(self, request_id: str) -> float:
        """How long to wait before closing: longer when nothing has arrived yet."""
        return DONE_SETTLE_SECONDS if self._posts.get(request_id, 0) else EMPTY_SETTLE_SECONDS

    async def _close_when_settled(self, request_id: str, outcome: ProcessingOutcome) -> None:
        import asyncio

        try:
            await asyncio.sleep(self._settle_window(request_id))
            # 🔴 See BUSY_RECHECK_SECONDS: the hook that started this close is not
            # evidence the turn ran. A reply that arrives meanwhile cancels this task
            # and starts a fresh window, exactly as during the settle.
            waited = 0.0
            while self._still_answering(request_id) and self._harness_busy(
                self._session_key_of.get(request_id, "")
            ):
                if waited >= BUSY_MAX_SECONDS:
                    logger.warning(
                        "uffici: request %s is closing although Hermes still shows its "
                        "session busy after %.0fs",
                        request_id, waited, extra={"requestId": request_id},
                    )
                    break
                await asyncio.sleep(BUSY_RECHECK_SECONDS)
                waited += BUSY_RECHECK_SECONDS
        except asyncio.CancelledError:
            # A reply arrived, or a cancel took the request. Either way the
            # close is somebody else's job now.
            return

        self._settling.pop(request_id, None)
        if request_id in self._closed:
            return

        # 🔴 The post count is read HERE, not when the hook fired — a reply that
        # arrived during the settle is exactly what this exists to catch, and it
        # is what turns `empty` into `answered`.
        posts = self._posts.pop(request_id, 0)
        if not posts and request_id in self._unprocessed:
            # 🔴 #2306: Hermes said its agent never ran this turn. `empty` would tell the
            # platform the message was handled and advance past it.
            outcome = ProcessingOutcome.FAILURE
        await self._send(_done_frame(request_id, outcome, posts))
        self._forget(request_id)

    def _interrupted_by_shutdown(self, request_id: str) -> bool:
        """Whether Hermes force-interrupted this request's turn at a drain that timed out.

        🔴 #2306. Read from Hermes's own record (``lookup_by_session_key``, its public
        reader) rather than inferred: see SHUTDOWN_INTERRUPT_REASONS. A store that cannot
        answer reads as "not interrupted", which keeps #2300's close.
        """
        key = self._session_key_of.get(request_id)
        lookup = getattr(getattr(self, "_session_store", None), "lookup_by_session_key", None)
        if not key or not callable(lookup):
            return False
        try:
            entry = lookup(key)
        except Exception:  # noqa: BLE001 — a harness internal; never fatal to a disconnect
            logger.warning(
                "uffici: could not read Hermes's session record for request %s",
                request_id, extra={"requestId": request_id},
            )
            return False
        return bool(getattr(entry, "resume_pending", False)) and (
            getattr(entry, "resume_reason", None) in SHUTDOWN_INTERRUPT_REASONS
        )

    async def _close_finished_turns(self) -> None:
        """Close, now, every settling turn that ended well and posted (#2300).

        🔴 Called as the wire goes. A draining restart (`hermes gateway restart`) lets the
        running turn FINISH, then exits inside its settle window; cancelling that close
        left the request open, the companion redelivered it to the new process, and the
        person was answered twice. No reply can follow from a process that is leaving,
        so the settle has nothing left to wait for. A turn that failed, was cancelled or
        posted nothing is NOT closed: it is still owed, and the companion's redelivery
        is how it gets answered (#2273).

        🔴 Nor is a turn the restart CUT (#2306). When the drain times out Hermes interrupts
        the running turn, and the turn still "ends" — with SUCCESS and whatever it had
        sent — before the wire goes. Closing it `answered` lost the request: nothing was
        redelivered and the person never got the answer. Hermes's own mark says which
        turns those are.
        """
        for request_id in list(self._settling):
            outcome = self._settling_outcome.get(request_id)
            if outcome != ProcessingOutcome.SUCCESS or request_id in self._closed:
                continue
            posts = self._posts.get(request_id, 0)
            if not posts:
                continue
            if self._interrupted_by_shutdown(request_id):
                logger.info(
                    "uffici: request %s was cut by the restart; leaving it to be redelivered",
                    request_id, extra={"requestId": request_id},
                )
                continue
            task = self._settling.pop(request_id)
            task.cancel()
            self._posts.pop(request_id, None)
            await self._send(_done_frame(request_id, outcome, posts))
            self._forget(request_id)

    # ---------------------------------------------------------------- inbound

    async def _on_frame(self, frame: dict[str, Any]) -> None:
        kind = frame.get("type")
        if kind == "brief":
            await self._on_brief(frame)
        elif kind == "opener":
            await self._on_opener(frame)
        elif kind == "cancel":
            await self._on_cancel(frame)
        elif kind == "result":
            self._calls.on_result(frame)
        else:
            # 🔴 Ignored, never fatal. A companion newer than this plugin may
            # send a frame it has never heard of, and refusing the connection
            # over one would take the whole conversation down with it.
            logger.debug("uffici: ignoring an unknown frame", extra=describe_frame(frame))

    async def _on_brief(self, frame: dict[str, Any]) -> None:
        request_id = str(frame.get("requestId") or "")
        conversation_id = str(frame.get("conversationId") or "")
        if not request_id or not conversation_id:
            logger.warning("uffici: a brief without ids was discarded")
            return

        # 🔴 A REDELIVERY is an ordinary event, not an error: the wire client
        # reconnects on its own, and the companion resends whatever it never got
        # an ack for. Admitting the same request twice would put the same delta
        # into the session twice and post two replies to one turn. The ack is
        # re-sent because the one that was lost is why this arrived again.
        if request_id in self._conversation_of or request_id in self._awaiting_opener:
            logger.info(
                "uffici: brief %s for conversation %s was redelivered",
                request_id, conversation_id,
                extra={"requestId": request_id, "conversationId": conversation_id},
            )
            session = "fresh" if request_id in self._awaiting_opener else "resumed"
            await self._send(
                {"v": 1, "type": "ack", "requestId": request_id,
                 "outcome": "accepted", "session": session}
            )
            return

        delta = frame.get("delta") or []
        if await self._answer_pending_question(frame, request_id, conversation_id, delta):
            return
        author = cause_author(frame.get("cause"), delta)
        source = self.build_source(
            chat_id=conversation_id,
            chat_type=CHAT_TYPE,
            user_id=str(author.get("id") or "") or None,
            user_name=str(author.get("label") or "") or None,
        )

        fresh = bool(frame.get("resetSession")) or not self._session_exists(source)
        if frame.get("resetSession"):
            self._reset_session(source)

        event = MessageEvent(
            text=render_entries(delta),
            message_type=MessageType.TEXT,
            user_id=source.user_id,
            user_name=source.user_name,
            source=source,
            message_id=request_id,
            # 🔴 The reason goes in the USER-side slot. ``channel_prompt`` is the
            # system-side one and is reserved for what is stable across a
            # session (decision 43); a per-turn line there would rewrite the
            # system prompt every turn and cost the provider's cache each time.
            channel_context=render_reason(
                str(frame.get("reason") or ""), frame.get("cause"), delta
            ),
            # 🔴 The platform's context goes in the SYSTEM-side slot, on EVERY
            # brief — fresh or resumed — so an edit lands on the next answer.
            # Unlike the reason above, it is stable across a session's turns
            # (it changes only when someone edits it), which is decision 43's
            # test for that slot; Hermes adds it at API time and never writes it
            # into the session's messages. It composes with the static
            # ``PLATFORM_HINT``, which Hermes places separately as the platform's
            # context prompt: the hint says what Uffici is, this says what the
            # organization and this place tell the agent, and neither replaces
            # the other. ``None`` (an older companion) leaves the hint alone,
            # exactly as before.
            channel_prompt=render_context(frame.get("context")),
            # 🔴 False: this text is a transcript, and a member who types
            # "/stop" into a conversation must not be able to drive the harness.
            allow_gateway_control=False,
            metadata={"uffici_request_id": request_id, "uffici_conversation_id": conversation_id},
        )

        self._conversation_of[request_id] = conversation_id
        self._session_key_of[request_id] = self._session_key_for(source)
        self._open_request[conversation_id] = request_id
        self._posts.setdefault(request_id, 0)
        self._known_conversation = conversation_id
        # Asked for now, so a fresh brief's opener round trip overlaps it (#2350).
        self._tools.kick(conversation_id)
        logger.info(
            "uffici: brief %s opened on conversation %s (%s, %d delta entr%s)",
            request_id, conversation_id, "fresh" if fresh else "resumed",
            len(delta), "y" if len(delta) == 1 else "ies",
            extra={"requestId": request_id, "conversationId": conversation_id},
        )

        if fresh:
            # 🔴 ACK FIRST, then hold. The companion sends the opener ONLY after
            # an accepted `fresh` ack (`ack.schema.json`: "An opener follows only
            # an accepted ack with session fresh"), so a plugin that waited for
            # the opener before acking would wait forever — and so would the
            # companion. Both sides stall until the deadline expires the turn,
            # which is exactly what a real Hermes did against a real companion:
            # three briefs, zero openers, every `fresh` expectation unmet.
            #
            # Acknowledging on receipt is also what the wire says to do
            # (`docs/local-wire.md`: "The doorbell is acknowledged on receipt").
            # What the harness then does with the event is the `done` frame's
            # business, not the ack's.
            self._awaiting_opener[request_id] = {"event": event, "deadline": frame.get("deadline")}
            await self._send(
                {
                    "v": 1, "type": "ack", "requestId": request_id,
                    "outcome": "accepted", "session": "fresh",
                }
            )
            return

        # 🔴 Acked on RECEIPT here too, not after the harness has taken it.
        #
        # `docs/local-wire.md`: "The doorbell is acknowledged on receipt." Only
        # the fresh path obeyed that, because the opener depends on it — and the
        # resumed path's late ack turned out to matter just as much: under a
        # real model `handle_message` can take longer than the companion's ack
        # timeout, so the companion redelivered a brief that was being answered
        # perfectly well. Measured on a real run: one redelivery, and posts that
        # arrived against a request the script had moved past.
        #
        # What the harness then does with the event is the `done` frame's
        # business, exactly as on the fresh path.
        await self._send(
            {
                "v": 1, "type": "ack", "requestId": request_id,
                "outcome": "accepted", "session": "resumed",
            }
        )
        await self._admit(event, request_id)

    async def _answer_pending_question(
        self, frame: dict[str, Any], request_id: str, conversation_id: str, delta: list
    ) -> bool:
        """Whether this brief answered the question a turn is waiting on (#2232).

        🔴 The answer is NOT a new turn. The turn that asked is still running, blocked in
        Hermes's clarify; a second turn on the same conversation would queue behind it
        and answer a conversation the first had not finished. So the person's words
        resolve the clarify — a number or a label picks a choice, anything else is the
        free "Other" answer — and the running turn carries on, speaking on THIS request.
        A brief in which no person spoke, a reset, or a question Hermes has already given
        up on is an ordinary turn.
        """
        pending = self._pending_questions.get(conversation_id)
        if pending is None:
            return False
        if frame.get("resetSession"):
            self._pending_questions.pop(conversation_id, None)
            self._release_questions(pending["sessionKey"])
            return False
        answer = _person_answer(delta)
        if answer is None:
            # 🔴 Nobody the question was put to has spoken — another agent's mention, a card
            # arrival. Queued behind the turn blocked in clarify it would expire, and its
            # cancel would end the person's question. Closed `cancelled` at once instead:
            # that leaves the offset, so the ring that brings the answer re-covers it.
            await self._send(
                {"v": 1, "type": "ack", "requestId": request_id,
                 "outcome": "accepted", "session": "resumed"}
            )
            await self._send(
                {"v": 1, "type": "done", "requestId": request_id, "outcome": "cancelled"}
            )
            self._forget(request_id)
            logger.info(
                "uffici: brief %s was held — the agent is waiting on a person's answer",
                request_id, extra={"requestId": request_id, "conversationId": conversation_id},
            )
            return True
        if len([e for e in delta if isinstance(e, dict)]) > 1:
            # 🔴 More than the answer arrived: all of it is what the waiting turn reads, so
            # nothing a person wrote beside it is dropped (the offset moves past it).
            answer = render_entries(delta)
        self._pending_questions.pop(conversation_id, None)
        if not _resolve_question(pending, answer):
            return False

        self._continues[pending["askedBy"]] = request_id
        self._conversation_of[request_id] = conversation_id
        self._session_key_of[request_id] = pending["sessionKey"]
        self._open_request[conversation_id] = request_id
        self._posts.setdefault(request_id, 0)
        await self._send(
            {"v": 1, "type": "ack", "requestId": request_id,
             "outcome": "accepted", "session": "resumed"}
        )
        logger.info(
            "uffici: brief %s answered the question request %s asked; the turn carries on",
            request_id, pending["askedBy"],
            extra={"requestId": request_id, "conversationId": conversation_id},
        )
        return True

    async def _on_opener(self, frame: dict[str, Any]) -> None:
        request_id = str(frame.get("requestId") or "")
        held = self._awaiting_opener.pop(request_id, None)
        if held is None:
            # An opener for a request we are not holding: the companion sends it
            # only after a fresh ack, so this means the turn is already gone.
            logger.debug(
                "uffici: an opener arrived for no held request", extra=describe_frame(frame)
            )
            return

        event: MessageEvent = held["event"]
        recalled = render_entries(frame.get("entries") or [])
        if recalled:
            # 🔴 Above the reason line and still user-side: one block, entered
            # once, stored with the first message of the session.
            reason_line = event.channel_context or ""
            event.channel_context = f"{OPENER_HEADER}\n{recalled}\n\n{reason_line}".strip()
        # Already acked when the brief arrived — that ack is what asked for this
        # opener. A harness that refuses the event now is a `done`, not a
        # retroactive ack.
        await self._admit(event, request_id)

    async def _admit(self, event: MessageEvent, request_id: str) -> None:
        """Hand the event to Hermes, and close the turn if it will not take it.

        🔴 There is no ack here. BOTH paths acknowledge on receipt now — the
        fresh one because the opener depends on it, the resumed one because a
        slow `handle_message` otherwise outlasts the companion's ack timeout and
        the brief is redelivered. So by the time this runs the request is already
        acknowledged, and the only thing left to say about a refusal is that the
        turn is over: `done {error}`.

        That makes the wire's `rejected` ack unreachable from this plugin. It
        stays in the protocol for a harness that can decide before it commits.

        🔴 While the tool catalog is being read, the event waits for it (#2350): Hermes
        builds a session's agent — and freezes its tool list — on that session's first
        turn, so a turn admitted before its tools are registered never sees them. It waits
        on its own task, because this runs inside the wire's read loop and the catalog's
        answer arrives on that same loop.
        """
        if self._tools.refreshing():
            self._admitting.add(request_id)
            _spawn(self._admit_after_refresh(event, request_id))
            return
        await self._admit_now(event, request_id)

    async def _admit_after_refresh(self, event: MessageEvent, request_id: str) -> None:
        await self._tools.wait()
        if request_id not in self._admitting:
            # Withdrawn, or the connection went, while it waited: nothing to hand over.
            return
        self._admitting.discard(request_id)
        await self._admit_now(event, request_id)

    async def _admit_now(self, event: MessageEvent, request_id: str) -> None:
        try:
            admitted = await self.handle_message(event)
        except Exception as exc:  # noqa: BLE001 — classified, never re-raised at the wire
            logger.warning("uffici: handing a brief to the harness failed (%s)", type(exc).__name__)
            refused = True
        else:
            refused = admitted is False

        if refused:
            await self._send(
                {
                    "v": 1, "type": "done", "requestId": request_id,
                    "outcome": "error", "errorClass": "harness_error",
                }
            )
            self._forget(request_id)

    async def _on_cancel(self, frame: dict[str, Any]) -> None:
        """A withdrawal becomes Hermes's own stop.

        🔴 ``allow_gateway_control=True`` on this one event, and only this one:
        it is the companion speaking over an authenticated local wire, not
        transcript text. Every brief carries False precisely so a member cannot
        type ``/stop`` into a conversation and drive the harness.
        """
        request_id = str(frame.get("requestId") or "")
        if frame.get("reason") == "failed":
            # The platform refused or failed a call this turn needed (#2077).
            # The request is already closed as an error on the platform, which
            # tells the person; the CLASS is logged here — platform vocabulary,
            # never tenant content — and the turn stops like any other cancel.
            error = frame.get("error") if isinstance(frame.get("error"), dict) else {}
            logger.warning(
                "uffici: the platform failed this turn (%s)", str(error.get("class") or "unknown")
            )
        conversation_id = self._conversation_of.get(request_id)
        self._awaiting_opener.pop(request_id, None)
        if request_id in self._admitting:
            # Withdrawn before Hermes ever had it (#2350): there is no turn to stop, and a
            # `/stop` now would reach whatever else is running on that conversation.
            self._admitting.discard(request_id)
            self._forget(request_id)
            return
        if conversation_id is None:
            # 🔴 A request this plugin already SETTLED — the one that asked on the card —
            # can still be cancelled: its question post failed upstream (the platform had
            # expired it, a 5xx), and the companion withdraws it. Its question must not
            # linger: the turn would hold Hermes's session, and the person's next message
            # would be taken as the answer to a card they never saw.
            # 🔴 And the turn STOPS (#2272). Released and left running, it carried on with
            # an empty answer, Hermes merged the person's next message into it, and that
            # message's request never closed: the turn's completion names this one.
            # The stop goes by conversation, and cannot reach a NEWER turn there: a brief
            # arriving after the question either answered it or was held, and both take
            # the pending question out of this map before a cancel could find it.
            for asked_on, pending in list(self._pending_questions.items()):
                if pending.get("askedBy") == request_id:
                    del self._pending_questions[asked_on]
                    self._release_questions(pending["sessionKey"])
                    await self._stop_turn(asked_on, request_id)
            return

        source = self.build_source(chat_id=conversation_id, chat_type=CHAT_TYPE)
        # 🔴 A turn waiting on a question ignores `/stop` (#2233): Hermes's clarify wait
        # returns only on an answer, its timeout or `clear_session`, and its gateway leaves
        # a pending clarify in place for a slash command. The blocked turn kept Hermes's
        # session lease, and the next turn waited behind it past its request's deadline.
        # So the question is released FIRST; then the stop takes.
        self._pending_questions.pop(conversation_id, None)
        self._release_questions(
            self._session_key_of.get(request_id) or self._session_key_for(source)
        )
        await self._stop_turn(conversation_id, request_id)
        # 🔴 The companion has already moved this request to cancelled, so the
        # `done` Hermes's completion hook would otherwise send is a frame with no
        # transition — and it would say `empty`, because the post count went with
        # the request.
        self._forget(request_id)

    # ---------------------------------------------------------------- helpers

    async def _stop_turn(self, conversation_id: str, request_id: str) -> None:
        """Hermes's own ``/stop`` for the turn running on a conversation."""
        stop = MessageEvent(
            text="/stop",
            message_type=MessageType.COMMAND,
            source=self.build_source(chat_id=conversation_id, chat_type=CHAT_TYPE),
            allow_gateway_control=True,
            metadata={"uffici_request_id": request_id},
        )
        try:
            await self.handle_message(stop)
        except Exception as exc:  # noqa: BLE001
            logger.warning("uffici: the stop event was refused (%s)", type(exc).__name__)

    def _release_questions(self, session_key: str) -> int:
        """Answer every question pending on a session with nothing, so its turn can end.

        Best-effort: a Hermes whose clarify store has moved costs today's behaviour —
        the stop alone — and never the cancel.
        """
        if not session_key:
            return 0
        try:
            from tools.clarify_gateway import clear_session  # type: ignore[import-not-found]

            released = int(clear_session(session_key) or 0)
        except Exception as exc:  # noqa: BLE001 — a moved internal is a fallback
            logger.warning(
                "uffici: a pending question could not be released (%s); the stop may wait "
                "for Hermes's own timeout", type(exc).__name__,
            )
            return 0
        if released:
            logger.info("uffici: released %d pending question(s) so the stop can take", released)
        return released

    def _forget(self, request_id: str, *, keep_chain: bool = False) -> None:
        # Remembered as closed, bounded, so a late lifecycle hook for it is a
        # no-op rather than a second terminal frame.
        self._closed[request_id] = None
        self._unprocessed.discard(request_id)
        self._admitting.discard(request_id)
        self._tools.forget(request_id)
        # A continuation into a request that is over leads nowhere (#2232): bounded by the
        # requests still open, never by the life of the process. Unless the request closed
        # because its turn ASKED: that turn carries on through it (#2296), and its end
        # pops the chain.
        if not keep_chain:
            for asked, answering in list(self._continues.items()):
                if answering == request_id:
                    del self._continues[asked]
        if len(self._closed) > CLOSED_MEMORY:
            del self._closed[next(iter(self._closed))]
        conversation_id = self._conversation_of.pop(request_id, None)
        self._session_key_of.pop(request_id, None)
        if conversation_id is not None and self._open_request.get(conversation_id) == request_id:
            self._open_request.pop(conversation_id, None)
        self._posts.pop(request_id, None)
        self._awaiting_opener.pop(request_id, None)
        settling = self._settling.pop(request_id, None)
        if settling is not None:
            settling.cancel()
        self._settling_outcome.pop(request_id, None)

    async def _send(self, frame: dict[str, Any]) -> bool:
        if self._client is None:
            logger.warning("uffici: not connected; dropping frame", extra=describe_frame(frame))
            return False
        return await self._client.send(frame)

    def _session_key_for(self, source: Any) -> str:
        """Hermes's own key for this conversation's session, or empty when unreadable.

        The base adapter's ``_source_session_key`` is preferred because it applies the
        operator's session settings exactly as the runner does; the plain rule is the
        fallback, and for a ``dm`` with a chat id the two agree.
        """
        try:
            own = getattr(self, "_source_session_key", None)
            return str(own(source) if callable(own) else build_session_key(source))
        except Exception as exc:  # noqa: BLE001 — an unreadable key means the probe sees idle
            # 🔴 Said, because it silently turns the busy wait off: every close would
            # trust the completion hook again, which is the `empty` of #2164.
            logger.warning(
                "uffici: Hermes's session key could not be read (%s); a close will not "
                "wait for the harness to go idle",
                type(exc).__name__,
            )
            return ""

    def _still_answering(self, request_id: str) -> bool:
        """Whether this request is still the one open on its conversation.

        A newer brief on the same conversation supersedes it, and the busy state from
        then on is the NEW turn's — waiting on it would hold the old request open for
        as long as the conversation stays active.
        """
        conversation_id = self._conversation_of.get(request_id)
        return conversation_id is not None and self._open_request.get(conversation_id) == request_id

    def _harness_busy(self, session_key: str) -> bool:
        """Whether Hermes is still working on this session.

        🔴 Idle is FOUR things, all read from Hermes (#2164): the base adapter's
        processing guard (``_active_sessions`` — held until after the final send), no
        message waiting to be merged in (``_pending_messages``), no running-turn slot in
        the runner (``_is_session_running``), and no startup restore in flight (which
        queues the message and reports it complete before it has run). Each is an
        internal that can move between versions, so an absent one reads as idle: a
        renamed attribute costs the old behaviour, never a request held forever.
        """
        if not session_key:
            return False
        try:
            for name in ("_active_sessions", "_pending_messages"):
                held = getattr(self, name, None)
                if isinstance(held, dict) and session_key in held:
                    return True
            runner = getattr(self, "gateway_runner", None)
            if runner is None:
                return False
            if bool(getattr(runner, "_startup_restore_in_progress", False)):
                return True
            running = getattr(runner, "_is_session_running", None)
            return bool(callable(running) and running(session_key))
        except Exception:  # noqa: BLE001 — a probe that raises reads as idle
            return False

    def _session_exists(self, source: Any) -> bool:
        """Whether Hermes already has a session for this conversation.

        🔴 When it cannot be answered, the answer is FRESH. A wrong "resumed"
        costs the model its whole recalled history — it answers a conversation
        it cannot see. A wrong "fresh" costs one extra read. The two mistakes
        are not the same size.
        """
        store = getattr(self, "_session_store", None)
        if store is None:
            return False
        try:
            key = build_session_key(source)
            entries = getattr(store, "_entries", None)
            if isinstance(entries, dict):
                return key in entries
            getter = getattr(store, "get_entry", None)
            if callable(getter):
                return getter(key) is not None
        except Exception:  # noqa: BLE001 — an internal that moved costs a fresh session
            return False
        return False

    def _reset_session(self, source: Any) -> None:
        store = getattr(self, "_session_store", None)
        reset = getattr(store, "reset_session", None)
        if not callable(reset):
            return
        try:
            reset(build_session_key(source))
        except Exception as exc:  # noqa: BLE001
            logger.warning("uffici: resetting the session failed (%s)", type(exc).__name__)


def _question_card(
    question: str, choices: list[str] | None, clarify_id: str, multi_select: bool
) -> tuple[str, dict[str, Any]] | None:
    """The question as the wire carries it, or ``None`` when the wire cannot (#2232)."""
    text = (question or "").strip()
    if not text or not QUESTION_ID_RE.match(str(clarify_id or "")):
        return None
    offered = [str(choice).strip() for choice in (choices or [])]
    if (
        len(offered) > QUESTION_MAX_CHOICES
        or any(not choice or _utf16_len(choice) > QUESTION_CHOICE_MAX_CHARS for choice in offered)
    ):
        return None
    return text[:POST_TEXT_MAX], {
        "id": str(clarify_id),
        "choices": offered,
        "multiSelect": bool(multi_select) and bool(offered),
    }


def _utf16_len(text: str) -> int:
    """Length as the platform's schemas count it: UTF-16 units, not code points."""
    return len(text.encode("utf-16-le")) // 2


def _done_frame(request_id: str, outcome: ProcessingOutcome, posts: int) -> dict[str, Any]:
    """The `done` a turn's end sends: how it ended, never why in the harness's words."""
    if outcome == ProcessingOutcome.CANCELLED:
        return {"v": 1, "type": "done", "requestId": request_id, "outcome": "cancelled"}
    if outcome == ProcessingOutcome.FAILURE:
        # 🔴 An error CLASS, never the provider's message: that text is the
        # model's, or the provider's, and it travels to the platform's audit.
        return {
            "v": 1, "type": "done", "requestId": request_id,
            "outcome": "error", "errorClass": "harness_error",
        }
    return {
        "v": 1, "type": "done", "requestId": request_id,
        "outcome": "answered" if posts else "empty",
    }


def _person_answer(delta: list) -> str | None:
    """The last thing a PERSON said in a delta — what answers a question."""
    for entry in reversed(delta or []):
        if not isinstance(entry, dict):
            continue
        author = entry.get("author") if isinstance(entry.get("author"), dict) else {}
        text = str(entry.get("text") or "").strip()
        if author.get("type") in PERSON_AUTHORS and entry.get("kind") == "message" and text:
            return text
    return None


def _resolve_question(pending: dict[str, str], answer: str) -> bool:
    """Hand Hermes the answer: its own coercion first (a number, a label, a list for a
    multi-select), then the words as they are — the free answer. Best-effort: a moved
    internal answers False, and the brief becomes an ordinary turn."""
    try:
        from tools.clarify_gateway import (  # type: ignore[import-not-found]
            resolve_gateway_clarify,
            resolve_text_response_for_session,
        )

        if resolve_text_response_for_session(pending["sessionKey"], answer):
            return True
        return bool(resolve_gateway_clarify(pending["clarifyId"], answer))
    except Exception as exc:  # noqa: BLE001 — a moved internal is a fallback
        logger.warning(
            "uffici: a question could not be answered in Hermes (%s)", type(exc).__name__
        )
        return False


async def _default_connect(address: Any) -> Any:  # pragma: no cover - needs a real socket
    """Open the WebSocket, through the library Hermes already ships."""
    import websockets

    if address.kind == "socket":
        return await websockets.unix_connect(address.socket_path, address.url)
    return await websockets.connect(address.url)


def _spawn(coro: Any) -> Any:
    """Run the wire loop beside the gateway."""
    import asyncio

    return asyncio.ensure_future(coro)


def check_requirements() -> bool:
    """PASSIVE probe: are this plugin's dependencies importable right now?

    Side-effect free, because Hermes calls it from status displays that must
    never install anything.
    """
    try:
        import websockets  # noqa: F401
    except ImportError:
        return False
    return True


def is_connected(config: PlatformConfig) -> bool:
    """Whether this platform is configured enough to connect."""
    get_env = env_reader()
    try:
        resolve_address(get_env)
        resolve_token(get_env)
    except WireConfigError:
        return False
    return True


def log_shipping_on(get_env: Callable[[str], str | None]) -> bool:
    """Whether this machine's lines leave it: on unless ``UFFICI_LOG_SHIPPING=off``."""
    return (get_env(LOG_SHIPPING_ENV) or "").strip().lower() != "off"


def register(ctx: Any) -> None:
    """The plugin's entry point, called by Hermes at startup."""
    # 🔴 Capture starts HERE, at plugin load (#2313): Hermes logs its startup warnings
    # after loading its plugins and before connecting them, and a handler installed at
    # connect never saw one.
    if log_shipping_on(env_reader()):
        install(shared_handler())
    # What the agent did with Hermes's OWN tools, by name, for its posts (#2350). The hook
    # is Hermes's observer seam (`post_tool_call`); it never changes a call.
    register_hook = getattr(ctx, "register_hook", None)
    if callable(register_hook):
        register_hook("post_tool_call", on_post_tool_call)
    ctx.register_platform(
        name=PLATFORM_NAME,
        label="Uffici",
        # 🔴 The plugin's context goes to the adapter: the agent's Uffici tools are
        # registered through it, as this plugin's (#2350).
        adapter_factory=lambda cfg: UfficiAdapter(cfg, plugin_ctx=ctx),
        check_fn=check_requirements,
        is_connected=is_connected,
        install_hint="pip install websockets   # already a Hermes dependency",
        emoji="🏢",
        # 🔴 0 means NO LIMIT in the registry, and that is deliberate: the
        # platform already bounded the transcript to the budget this plugin
        # declared, and a second cap here would chop a reply the platform sized.
        max_message_length=0,
        platform_hint=PLATFORM_HINT,
        # 🔴 Both deliberately absent: no allowlist and no allow-all switch.
        # Authorization is upstream (see ``UfficiAdapter``), and an allow-all
        # environment variable on this platform would be a way to widen access
        # to a conversation the platform never granted.
    )
