"""The Uffici platform adapter for Hermes.

A format translator and nothing else: a brief becomes one inbound message, each
send becomes a post, the turn's end becomes a done, a cancel becomes Hermes's
own ``/stop``. It holds no policy, no memory and no tools — the companion holds
every connection and every secret, which is what keeps this file small enough
to read.

🔴 Authorization is DELEGATED, never absent. The companion authenticates the
local wire with a token before a frame is delivered, and the platform decided
which agent is a participant before it rang. Hermes's default-deny allowlist
would therefore be asking the wrong question, so this adapter declares
``authorization_is_upstream`` — the same declaration, for the same reason, that
the bundled a2a adapter makes. It never sets an allow-all environment variable
and never carries an allowlist of its own.
"""

from __future__ import annotations

import logging
from typing import Any

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import build_session_key

from .budget import BudgetUnavailable, context_budget_tokens, resolve_share, resolve_window_tokens
from .calls import CallClient
from .hint import PLATFORM_HINT
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

#: The platform's name, in `config.yaml`, in the registry and on the wire.
PLATFORM_NAME = "uffici"

#: This plugin's own version, sent in ``hello``.
PLUGIN_VERSION = "0.1.0"

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


class UfficiAdapter(BasePlatformAdapter):
    """One conversation service, many conversations, over one local wire."""

    supports_code_blocks = True

    def __init__(self, config: PlatformConfig, *, get_env=None, connect_fn=None) -> None:
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
        #: matched to their results by callId. Nothing calls it yet; its first
        #: consumer is conversation-scoped memory (stage 4).
        self._calls = CallClient(self._send)

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
        try:
            address = resolve_address(self._get_env)
            token = resolve_token(self._get_env)
            budget = context_budget_tokens(
                resolve_window_tokens(self._get_env, self._window_probes()),
                resolve_share(self._get_env),
            )
        except (WireConfigError, BudgetUnavailable) as exc:
            # The message names what is missing and how to supply it, and carries
            # no path and no token: these refusals are read from a log.
            logger.error("uffici: cannot start — %s", exc)
            return False

        self._client = WireClient(
            address=address,
            token=token,
            harness_version=self._harness_version(),
            plugin_version=PLUGIN_VERSION,
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
        return True

    async def _on_companion_hello(self, _frame: dict[str, Any]) -> None:
        """The wire is up. Announce readiness once the harness can answer.

        🔴 On its own task, because this runs inside the client's read loop —
        waiting here would stop the plugin reading the very frames it is about
        to ask for.
        """
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

    async def disconnect(self) -> None:
        if self._client is not None:
            await self._client.stop()
        self._client = None
        for name in ("_runner", "_ready_runner"):
            runner = getattr(self, name, None)
            setattr(self, name, None)
            if runner is not None:
                runner.cancel()
        self._forget_everything()

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

        return (from_session_store, from_agent_defaults)

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
        if len(text) > POST_TEXT_MAX:
            # Clipped rather than refused: a truncated answer reaches the person,
            # and a refused frame reaches nobody.
            logger.warning(
                "uffici: a reply was clipped to the wire's bound",
                extra={"requestId": request_id, "textChars": len(text)},
            )
            text = text[:POST_TEXT_MAX]

        sent = await self._send({"v": 1, "type": "post", "requestId": request_id, "text": text})
        if not sent:
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
        if outcome == ProcessingOutcome.CANCELLED:
            frame = {"v": 1, "type": "done", "requestId": request_id, "outcome": "cancelled"}
        elif outcome == ProcessingOutcome.FAILURE:
            # 🔴 An error CLASS, never the provider's message: that text is the
            # model's, or the provider's, and it travels to the platform's audit.
            frame = {
                "v": 1, "type": "done", "requestId": request_id,
                "outcome": "error", "errorClass": "harness_error",
            }
        else:
            frame = {
                "v": 1, "type": "done", "requestId": request_id,
                "outcome": "answered" if posts else "empty",
            }

        await self._send(frame)
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
            # 🔴 False: this text is a transcript, and a member who types
            # "/stop" into a conversation must not be able to drive the harness.
            allow_gateway_control=False,
            metadata={"uffici_request_id": request_id, "uffici_conversation_id": conversation_id},
        )

        self._conversation_of[request_id] = conversation_id
        self._session_key_of[request_id] = self._session_key_for(source)
        self._open_request[conversation_id] = request_id
        self._posts.setdefault(request_id, 0)
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
        """
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
        if conversation_id is None:
            return

        source = self.build_source(chat_id=conversation_id, chat_type=CHAT_TYPE)
        stop = MessageEvent(
            text="/stop",
            message_type=MessageType.COMMAND,
            source=source,
            allow_gateway_control=True,
            metadata={"uffici_request_id": request_id},
        )
        try:
            await self.handle_message(stop)
        except Exception as exc:  # noqa: BLE001
            logger.warning("uffici: the stop event was refused (%s)", type(exc).__name__)
        # 🔴 The companion has already moved this request to cancelled, so the
        # `done` Hermes's completion hook would otherwise send is a frame with no
        # transition — and it would say `empty`, because the post count went with
        # the request.
        self._forget(request_id)

    # ---------------------------------------------------------------- helpers

    def _forget(self, request_id: str) -> None:
        # Remembered as closed, bounded, so a late lifecycle hook for it is a
        # no-op rather than a second terminal frame.
        self._closed[request_id] = None
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


def register(ctx: Any) -> None:
    """The plugin's entry point, called by Hermes at startup."""
    ctx.register_platform(
        name=PLATFORM_NAME,
        label="Uffici",
        adapter_factory=lambda cfg: UfficiAdapter(cfg),
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
