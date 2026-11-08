"""What this plugin assumes about Hermes, checked against the REAL Hermes.

The test suite runs against a stand-in transcribed from the pinned Hermes source
(``tests/conftest.py``), and a stand-in is an assertion about Hermes that nothing
checks. Each assumption below once failed on a real machine while every test
passed, so each is checked here against Hermes's own code:

- **A brief Hermes parks behind a running turn still closes.** Hermes drains it as
  a queued follow-up and fires its lifecycle hooks only for an adapter that
  overrides ``on_processing_start``. Without that, the answered request stayed
  open until Uffici's deadline cancelled it.
- **Hermes's restart notice is off for the Uffici platform.** Otherwise it is
  posted as the agent's message.
- **The setup manifest's switches are the ones Hermes reads.** These are the
  "⏳ Working" line, per platform, and the memory-review note, for the whole
  Hermes.
- **The brief's context reaches the system side and is never stored.** The plugin
  places it in ``MessageEvent.channel_prompt``; Hermes must join that slot into the
  ephemeral system prompt it sends on every call, and keep it out of the session's
  ``messages``, which it persists.
- **The agent's Uffici tools reach the model, and each call finds its conversation.**
  The plugin registers them through its own plugin context in the ``uffici`` toolset and
  writes no toolset setting: Hermes must enable that toolset for the Uffici platform by
  itself, offer the tool (directly or behind its tool-search bridge), dispatch the
  plugin's handler, bind the turn's chat id where the handler reads it, and remove a tool
  the plugin disposes.
- **Uffici memory is Hermes's memory, and Hermes's own is off.** The plugin's directory
  is the provider ``memory.provider`` names: Hermes's memory loader must find it there,
  load it WITHOUT registering the platform a second time, and reach the adapter's prefetch
  through the process-global rendezvous from its own copy of the package. The manifest's
  keys must switch off MEMORY.md and USER.md (and with them the memory nudge). A prefetch
  must land fenced on the user side before the call, unwrapped by the plugin, inside a
  bound that is still the 8 s the plugin's deadline assumes, and its recall count must
  become Hermes's status line (which the adapter never posts — pinned by its own suite).
- **The hello's drift is Hermes's own reading of its configuration.** An MCP server
  Hermes would start, a platform its gateway would connect besides Uffici, and Hermes's
  own memory back on must each be reported, through Hermes's own readers — and the
  configuration ``uffici setup`` writes must report none of them.
- **A brief's freshness is Hermes's routing decision.** The plugin routes through the
  store's own ``get_or_create_session`` and reads ``load_transcript`` on the session it
  returns: a new conversation, a session ``/stop`` suspended and one whose row was ended
  in ``state.db`` must each route to a session with no transcript (``fresh``), and a
  session with turns must route to itself (``resumed``). And the Hermes this probe runs
  in must be inside the range the plugin declares.
- **Files cross in Hermes's own conventions.** A person's file reaches Hermes as a local
  path in ``MessageEvent.media_urls`` with its type in ``media_types``; a file the agent
  makes leaves through a ``MEDIA:`` tag ``extract_media`` takes out of the reply, which
  Hermes's own delivery hands to the adapter's ``send_document`` / ``send_image_file`` —
  never to the base class's failure notice. And the manifest makes the workspace a root
  Hermes's strict delivery policy accepts.
- **Hermes's own log lines reach a root handler.** The plugin ships them with one
  handler on the root logger: that works only while Hermes logs a turn through
  Python ``logging`` on a named logger that propagates, with the root at INFO or
  below.

Run inside the Hermes image, by both end-to-end runners, with this file on stdin::

    docker exec -i <hermes> /opt/hermes/.venv/bin/python - <manifest json> < hermes_contract.py

inside the running gateway's container, so Hermes discovers the plugin from the same
directory and configuration the gateway loaded it from.

It prints one line per check and exits 1 naming the first that fails.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def as_yaml(config: dict) -> str:
    """A config file Hermes's own reader accepts, written without a YAML library.

    JSON is YAML, so this needs nothing Hermes may stop shipping: 0.23.0 replaced PyYAML
    with ruamel.yaml, and the probe's ``import yaml`` stopped finding a module.
    """
    return json.dumps(config, indent=2)


def registered_factory():
    """The adapter factory Hermes registered for ``uffici``, loaded by Hermes's own discovery.

    Not an import of the package: the probe builds the adapter exactly the way the gateway
    does, from whatever Hermes loaded out of its plugin directory under its own config.
    """
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import discover_plugins

    discover_plugins()
    entry = platform_registry.get("uffici")
    if entry is None:
        raise SystemExit(
            "hermes contract: Hermes registered no uffici platform — the plugin did not load"
        )
    return entry.adapter_factory


def configured(manifest: dict) -> dict:
    """The config the manifest's ``hermes config set`` lines produce.

    Each value goes through Hermes's OWN coercion for ``config set``, so ``false`` and
    ``off`` land exactly as the setup command stores them.
    """
    from hermes_cli.config import _coerce_config_set_value

    config: dict = {}
    for entry in manifest.get("config", []):
        *parents, leaf = entry["key"].split(".")
        node = config
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = _coerce_config_set_value(entry["key"], entry["value"])
    return config


def read_site(module, line: str) -> str | None:
    """A line of Hermes this probe relies on, still there verbatim.

    Where Hermes decides inline, inside a method no probe can call, the probe applies the
    same rule itself — and pins the line it copied, so a Hermes that changes the rule fails
    here instead of leaving the copy to drift.
    """
    if line not in inspect.getsource(module):
        return f"{module.__name__} no longer contains: {line}"
    return None


def follow_up_hooks_reach_the_adapter(factory) -> str | None:
    from gateway.config import PlatformConfig
    from gateway.platforms.event import MessageEvent
    from gateway.run_turn_followup_ack import _followup_processing_hooks_apply

    adapter = factory(PlatformConfig(enabled=True))
    event = MessageEvent(text="probe", message_id="probe-request")
    if _followup_processing_hooks_apply(adapter, event) is not True:
        return (
            "Hermes would not fire the lifecycle hooks for a brief it drains as a queued "
            "follow-up: that request would never close"
        )
    return None


def startup_resume_reaches_the_adapter(factory) -> str | None:
    """Hermes's boot resume is the event the plugin declines (#2273), by its shape.

    The resume is synthesized inline and dispatched through the adapter's own
    ``handle_message``; the plugin recognises it as internal, empty and carrying no
    request of ours. Both lines are pinned, and the adapter is checked to decline
    exactly that event, so a Hermes that reshapes its resume fails here instead of
    answering the person twice. A declined resume runs no turn, and Hermes clears a
    session's resume-pending flag only after a turn succeeds — pinned too, because that
    is what puts the interruption note into the redelivered brief's turn instead.
    """
    import gateway.run_startup as startup
    import gateway.run_turn_persistence as run_turn_persistence
    from gateway.config import PlatformConfig
    from gateway.platforms.event import MessageEvent, MessageType

    for line in (
        'event = MessageEvent(text="", message_type=MessageType.TEXT, '
        "source=source, internal=True)",
        "await adapter.handle_message(event)",
    ):
        if drifted := read_site(startup, line):
            return drifted
    if drifted := read_site(
        run_turn_persistence,
        "if session_key and _should_clear_resume_pending_after_turn(agent_result):",
    ):
        return drifted
    adapter = factory(PlatformConfig(enabled=True))
    event = MessageEvent(
        text="",
        message_type=MessageType.TEXT,
        source=adapter.build_source(chat_id="probe", chat_type="dm"),
        internal=True,
    )
    if not adapter._is_startup_resume(event):  # noqa: SLF001 — the rule under test
        return "the plugin would not recognise Hermes's startup resume, and would answer twice"
    return None


def restart_notice_is_off(_factory) -> str | None:
    """The switch is set on the config the gateway READS, not on a copy of it."""
    import gateway.run_startup as startup
    from gateway.config import PlatformConfig
    from gateway.platform_registry import platform_registry
    from gateway.run_shutdown import GatewayShutdownMixin

    # The gateway builds each adapter from the platform's entry in its own config...
    for line in (
        "for platform, platform_config in self.config.platforms.items():",
        "adapter = self._create_adapter(platform, platform_config)",
    ):
        if drifted := read_site(startup, line):
            return drifted
    # ...through the registry, which hands the factory that same object.
    config = PlatformConfig(enabled=True)
    adapter = platform_registry.create_adapter("uffici", config)
    if adapter is None or adapter.config is not config:
        return "Hermes's registry did not hand the adapter the platform's own config object"
    runner = SimpleNamespace(config=SimpleNamespace(platforms={adapter.platform: config}))
    if GatewayShutdownMixin._restart_notification_allowed(runner, adapter.platform) is not False:
        return "Hermes would post its restart and shutdown notice into Uffici conversations"
    return None


def restart_cut_is_hermes_own_record(factory) -> str | None:
    """What the plugin reads to tell a turn the restart CUT from one it let finish (#2306).

    Hermes marks every running session ``resume_pending`` before a drain, clears the mark
    from each that finished once the drain completes, and marks again before it
    interrupts the rest; the plugin reads that mark through the store's public reader at
    the disconnect. And the sentences Hermes sends for a turn its agent never ran are
    produced by its own normalizer here, so a reworded release fails this probe instead
    of leaving the plugin's list matching nothing.
    """
    import dataclasses

    import gateway.run_shutdown as shutdown
    from gateway.config import PlatformConfig
    from gateway.run import _normalize_empty_agent_response
    from gateway.session import SessionEntry, SessionStore

    # The plugin's own module, as Hermes's discovery loaded it — not an import of the
    # package, which Hermes loads under its own namespace.
    plugin = sys.modules[type(factory(PlatformConfig(enabled=True))).__module__]
    reasons = plugin.SHUTDOWN_INTERRUPT_REASONS
    sentences = plugin.HERMES_UNPROCESSED_REPLIES

    for line in (
        'reason = "restart_timeout" if self._restart_requested else "shutdown_timeout"',
        "await self.async_session_store.clear_resume_pending(_sk)",
        'await GatewayRunner._mark_running_sessions_resume_pending(self, "mark_resume_pending")',
    ):
        if drifted := read_site(shutdown, line):
            return drifted
    if {"restart_timeout", "shutdown_timeout"} != reasons:
        return "the plugin reads other reasons than the two Hermes marks a cut drain with"
    if not callable(getattr(SessionStore, "lookup_by_session_key", None)):
        return "Hermes's session store has no lookup_by_session_key for the plugin to read"
    fields = {field.name for field in dataclasses.fields(SessionEntry)}
    if not {"resume_pending", "resume_reason"} <= fields:
        return "Hermes's session entry no longer carries resume_pending / resume_reason"
    produced = {
        _normalize_empty_agent_response({"interrupted": True, "api_calls": 0}, ""),
        _normalize_empty_agent_response({"api_calls": 0}, ""),
    }
    if produced != set(sentences):
        return (
            "Hermes's never-processed replies differ from the plugin's list, so one would "
            "reach the person as the agent's answer"
        )
    return None


def working_line_is_off(manifest: dict) -> str | None:
    import gateway.run_turn as run_turn
    from gateway.display_config import OVERRIDEABLE_KEYS, resolve_display_setting

    if "long_running_notifications" not in OVERRIDEABLE_KEYS:
        return "Hermes no longer has a per-platform long_running_notifications switch"
    # `_display_surface_mode` posts the line when the resolved value is truthy.
    if drifted := read_site(run_turn, 'return "raw" if bool(value) else "off"'):
        return drifted
    value = resolve_display_setting(
        configured(manifest), "uffici", "long_running_notifications", True
    )
    if bool(value):
        return f"the manifest leaves the Working line on for uffici (resolved {value!r})"
    return None


def memory_note_is_off(manifest: dict) -> str | None:
    import gateway.run_turn_runner as turn_runner
    from agent.background_review import summarize_background_review_actions

    # Hermes reads the switch inline, from the WHOLE display section, never per platform.
    for line in (
        'mem_notif = (ctx.user_config.get("display") or {}).get("memory_notifications")',
        'agent.memory_notifications = str(mem_notif).lower() if mem_notif else "on"',
    ):
        if drifted := read_site(turn_runner, line):
            return drifted
    mem_notif = (configured(manifest).get("display") or {}).get("memory_notifications")
    if isinstance(mem_notif, bool):
        mem_notif = "on" if mem_notif else "off"
    mode = str(mem_notif).lower() if mem_notif else "on"

    # One successful memory write, as the review agent's transcript carries it. The positive
    # control proves it is a note Hermes WOULD post, so an empty answer below means "off".
    review = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "probe-call",
                    "type": "function",
                    "function": {
                        "name": "memory",
                        "arguments": '{"action": "add", "target": "memory"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "probe-call",
            "content": '{"success": true, "message": "Entry added", "target": "memory"}',
        },
    ]
    if not summarize_background_review_actions(review, [], "on"):
        return "the probe's memory write is not one Hermes would announce; the check proves nothing"
    if summarize_background_review_actions(review, [], mode):
        return f"the manifest leaves the memory-review note on (mode {mode!r})"
    return None


#: Never real context: a marker the probe looks for on each side of the boundary.
CONTEXT_CANARY = "uffici-context-canary-7f3a"


def context_reaches_the_system_side_only(_factory) -> str | None:
    """``channel_prompt`` joins the system message Hermes sends, and never ``messages``.

    Runs Hermes's OWN two functions on stand-ins: the gateway's join of the event's
    ``channel_prompt`` into the agent's ephemeral system prompt, and the agent's build of
    one API call from the session's ``messages``. The line that hands the join to the
    agent is pinned, because no probe can call the method it sits in.
    """
    import agent.turn_context as turn_context
    import gateway.run_turn_runner as run_turn_runner
    import gateway.session_execution as session_execution

    if drifted := read_site(
        session_execution, "ephemeral_system_prompt=combined_ephemeral or None"
    ):
        return drifted
    turn = SimpleNamespace(
        _ctx=SimpleNamespace(
            context_prompt="the platform hint",
            channel_prompt=CONTEXT_CANARY,
            source=SimpleNamespace(platform="uffici", chat_id="probe"),
        ),
        _runner=SimpleNamespace(_get_system_prompt_for_channel=lambda *_a, **_k: None),
    )
    combined = run_turn_runner.TurnRunner._combined_ephemeral_prompt(turn)  # noqa: SLF001
    if CONTEXT_CANARY not in combined or "the platform hint" not in combined:
        return (
            "Hermes no longer joins the event's channel_prompt into the ephemeral system "
            "prompt: the brief's context would never reach the model"
        )
    agent = SimpleNamespace(
        ephemeral_system_prompt=combined,
        _current_turn_timestamp=0.0,
        model="probe",
        _copy_reasoning_content_for_api=lambda _msg, _api: None,
        _should_sanitize_tool_calls=lambda: False,
    )
    messages = [{"role": "user", "content": "hello", "api_content": "hello"}]
    before = json.dumps(messages, sort_keys=True)
    api_messages, _effective = turn_context.build_api_messages(
        agent,
        messages,
        current_turn_user_idx=0,
        ext_prefetch_cache=None,
        plugin_user_context=None,
        moa_config=None,
        active_system_prompt="the cached prompt",
    )
    system = api_messages[0] if api_messages else {}
    if system.get("role") != "system" or CONTEXT_CANARY not in str(system.get("content")):
        return "the channel_prompt no longer reaches the system message Hermes sends"
    if json.dumps(messages, sort_keys=True) != before or CONTEXT_CANARY in before:
        return "the channel_prompt reached the session's messages, which Hermes persists"
    return None


PROBE_TOOL = "uffici_contract_probe"


def platform_tools_reach_the_model(factory, manifest: dict) -> str | None:
    """The agent's Uffici tools, end to end inside the real Hermes but for the wire (#2350)."""
    import json as _json
    import threading

    from gateway.config import PlatformConfig
    from gateway.session_context import set_session_vars
    from hermes_cli.plugins import VALID_HOOKS
    from hermes_cli.tools_config import _get_platform_tools
    from model_tools import get_tool_definitions
    from tools.registry import registry

    if "post_tool_call" not in VALID_HOOKS:
        return "Hermes no longer has the post_tool_call hook the plugin reports activity from"
    if any(
        entry["key"].startswith(("platform_toolsets", "tools."))
        for entry in manifest.get("config", [])
    ):
        return "the manifest writes a toolset setting, which replaces the operator's selection"
    adapter = factory(PlatformConfig(enabled=True))
    tools = adapter._tools  # noqa: SLF001 — the plugin's own registrar, as the gateway built it
    plugin = sys.modules[type(tools).__module__]
    adapter._companion_features = frozenset({plugin.FEATURE_TOOLS})  # noqa: SLF001
    tools.apply(
        {
            PROBE_TOOL: {
                "name": PROBE_TOOL,
                "description": "Contract probe.",
                "permission": "allow",
                "parameters": {"type": "object", "properties": {}},
            }
        }
    )
    entry = registry.get_entry(PROBE_TOOL)
    if entry is None or entry.toolset != plugin.TOOLSET:
        return "the plugin's context registered no tool in the uffici toolset"
    enabled = sorted(_get_platform_tools(configured(manifest), "uffici"))
    if plugin.TOOLSET not in enabled:
        return f"Hermes does not enable the {plugin.TOOLSET} toolset for the uffici platform"
    definitions = get_tool_definitions(enabled_toolsets=enabled, quiet_mode=True) or []
    names = {d.get("function", {}).get("name") for d in definitions}
    if PROBE_TOOL not in names and not (
        "tool_call" in names and PROBE_TOOL in _json.dumps(definitions)
    ):
        return "a registered Uffici tool never reaches the model, directly or by tool search"

    seen: dict[str, object] = {}

    def turn() -> None:
        set_session_vars(platform="uffici", chat_id="probe-conversation")
        seen["conversation"] = plugin.session_conversation()
        # No loop is bound in this process: the handler must answer, and say why.
        seen["result"] = registry.dispatch(PROBE_TOOL, {})

    worker = threading.Thread(target=turn)
    worker.start()
    worker.join(timeout=30)
    if seen.get("conversation") != "probe-conversation":
        return "the turn's chat id is not where the plugin's tool handler reads it"
    try:
        answered = _json.loads(str(seen.get("result")))
    except ValueError:
        answered = {}
    if answered.get("errorClass") != "unknown_op":
        return "Hermes did not dispatch the plugin's handler with the call's arguments"
    tools.apply({})
    if registry.get_entry(PROBE_TOOL) is not None:
        return "a tool the plugin disposed is still registered in Hermes"
    return None


#: Never a real fact: a marker the memory checks look for.
MEMORY_CANARY = "uffici-memory-canary-5d1e"


def memory_provider_is_the_plugin(factory, manifest: dict) -> str | None:
    """Hermes's memory loader activates the plugin's directory as the provider, and the
    provider copy reaches the gateway copy's adapter through the rendezvous."""
    import threading

    from agent.memory_provider import MemoryProvider
    from gateway.config import PlatformConfig
    from gateway.platform_registry import platform_registry
    from gateway.session_context import set_session_vars
    from plugins.memory import load_memory_provider

    name = (configured(manifest).get("memory") or {}).get("provider")
    adapter = factory(PlatformConfig(enabled=True))
    package = sys.modules[type(adapter).__module__.rsplit(".", 1)[0]]
    plugin_memory = sys.modules[f"{package.__name__}.memory"]
    if name != plugin_memory.PROVIDER_DIRECTORY:
        return f"the manifest selects memory provider {name!r}, not the plugin's directory"
    before = platform_registry.get("uffici").adapter_factory
    provider = load_memory_provider(name)
    if not isinstance(provider, MemoryProvider) or provider.name != plugin_memory.PROVIDER_NAME:
        return f"Hermes's memory loader did not activate the plugin as {name!r}"
    if type(provider).__module__.split(".")[0] == package.__name__.split(".")[0]:
        return "the provider was not loaded as Hermes's own copy; the rendezvous proves nothing"
    if platform_registry.get("uffici").adapter_factory is not before:
        return "loading the memory provider registered the uffici platform a second time"
    if provider.get_tool_schemas() != [] or provider.system_prompt_block() != "":
        return "the provider offers tools or a prompt block of its own"

    from agent.memory_manager import MemoryManager

    seen: dict[str, object] = {}

    def bridge(conversation: str, query: str) -> tuple[str, int]:
        seen["asked"] = (conversation, query)
        return MEMORY_CANARY, 2

    # 🔴 Through Hermes's OWN manager, from a turn's thread: the manager runs the prefetch
    # on a thread of its own, so this is what proves the turn's session survives the hop
    # (a provider called directly would prove only the probe's thread).
    manager = MemoryManager()
    manager.add_provider(provider)
    plugin_memory.publish(bridge)
    try:

        def turn() -> None:
            set_session_vars(platform="uffici", chat_id="probe-conversation")
            seen["answer"] = manager.prefetch_all("what was decided?", session_id="probe")
            seen["status"] = manager.describe_recall()

        worker = threading.Thread(target=turn)
        worker.start()
        worker.join(timeout=30)
    finally:
        plugin_memory.withdraw(bridge)
    if seen.get("asked") != ("probe-conversation", "what was decided?"):
        return (
            "a prefetch through Hermes's memory manager never reached the adapter for the "
            "turn's conversation: the session is lost on its thread, or the rendezvous failed"
        )
    if MEMORY_CANARY not in str(seen.get("answer")):
        return "Hermes's memory manager dropped the provider's recall"
    if "Uffici" not in str(seen.get("status")) or "2" not in str(seen.get("status")):
        return "the provider's recall count no longer becomes Hermes's status line"
    return None


def builtin_memory_is_off(manifest: dict) -> str | None:
    """The manifest's keys switch off MEMORY.md and USER.md — and with no store, the memory
    nudge has nothing to tick, so ``nudge_interval`` needs no key of its own."""
    import agent.turn_context as turn_context
    from tools.memory_tool import get_builtin_memory_store_flags

    if get_builtin_memory_store_flags(configured(manifest)) != (False, False):
        return "the manifest leaves Hermes's own MEMORY.md or USER.md on"
    if get_builtin_memory_store_flags({}) != (True, True):
        return "Hermes's built-in memory is no longer on by default; the check proves nothing"
    if drifted := read_site(turn_context, "and agent._memory_store):"):
        return drifted
    agent = SimpleNamespace(
        _memory_nudge_interval=1,
        valid_tool_names={"memory"},
        _memory_store=None,
        _turns_since_memory=5,
    )
    if turn_context._tick_memory_nudge(agent) is not False:  # noqa: SLF001
        return "Hermes would nudge the agent toward a memory store that is off"
    return None


def prefetch_lands_fenced_on_the_user_side(factory) -> str | None:
    """A prefetch rides the user message Hermes sends, inside its ``<memory-context>``
    fence, never the system message — and the plugin's text needs no unwrapping."""
    import agent.memory_manager as memory_manager
    import agent.turn_context as turn_context
    from gateway.config import PlatformConfig

    adapter = factory(PlatformConfig(enabled=True))
    package = type(adapter).__module__.rsplit(".", 1)[0]
    plugin_memory = sys.modules[f"{package}.memory"]

    if memory_manager._EXTERNAL_PREFETCH_TIMEOUT_S != plugin_memory.HERMES_PREFETCH_BOUND_SECONDS:  # noqa: SLF001
        return (
            f"Hermes bounds a prefetch at {memory_manager._EXTERNAL_PREFETCH_TIMEOUT_S}s, "  # noqa: SLF001
            f"not the {plugin_memory.HERMES_PREFETCH_BOUND_SECONDS}s the plugin's deadline assumes"
        )
    # A real hit, from the wire's own golden — never a URN spelled here.
    golden = (
        Path(plugin_memory.__file__).parent
        / "wire"
        / "v1"
        / "golden"
        / "ops"
        / "valid"
        / "memory.recall.result-levels.json"
    )
    hit = {**json.loads(golden.read_text())["hits"][0], "text": MEMORY_CANARY}
    rendered, _count = plugin_memory.render_recall({"hits": [hit], "degraded": False})
    if memory_manager.sanitize_context(rendered) != rendered:
        return "Hermes strips something from the plugin's recall text: the plugin must not wrap it"
    for line in (
        "ext_prefetch_cache = agent._memory_manager.prefetch_all("
        '_query, session_id=agent.session_id) or ""',
        "agent._emit_status(_recall_indicator)",
    ):
        if drifted := read_site(turn_context, line):
            return drifted
    agent = SimpleNamespace(
        ephemeral_system_prompt=None,
        _current_turn_timestamp=0.0,
        model="probe",
        _copy_reasoning_content_for_api=lambda _msg, _api: None,
        _should_sanitize_tool_calls=lambda: False,
    )
    messages = [{"role": "user", "content": "hello"}]
    api_messages, _effective = turn_context.build_api_messages(
        agent,
        messages,
        current_turn_user_idx=0,
        ext_prefetch_cache=rendered,
        plugin_user_context=None,
        moa_config=None,
        active_system_prompt="the cached prompt",
    )
    users = [m for m in api_messages if m.get("role") == "user"]
    systems = [m for m in api_messages if m.get("role") == "system"]
    user_text = json.dumps(users)
    if MEMORY_CANARY not in user_text or "<memory-context>" not in user_text:
        return "a prefetch no longer reaches the user message, fenced, before the call"
    if MEMORY_CANARY in json.dumps(systems):
        return "a prefetch reached the system message"

    return None


#: The issue's fixture (#2406): over what setup writes, an extra MCP server (and a disabled
#: one, which is not drift), a second platform the gateway would connect, and Hermes's own
#: memory switched back on. The bot token is a shape Hermes accepts, never a real one.
DRIFT_FIXTURE = {
    "mcp_servers": {
        "probe-extra": {"url": "https://example.invalid/mcp"},
        "probe-disabled": {"command": "true", "enabled": False},
    },
    "gateway": {
        "platforms": {
            "uffici": {"enabled": True},
            "telegram": {
                "enabled": True,
                "token": "1234567890:AAEprobeprobeprobeprobeprobeprobe",  # gitleaks:allow
            },
        }
    },
}


def drift_reads_hermes_own_config(factory, manifest: dict) -> str | None:
    """The plugin's drift reader, pointed at a Hermes home holding the fixture, reports
    each kind through Hermes's OWN readers — and the setup's configuration reports none."""
    import copy
    import importlib
    import tempfile

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.memory_tool import get_builtin_memory_store_flags

    package = factory.__module__.rsplit(".", 1)[0]
    drift = importlib.import_module(f"{package}.drift")
    clean = configured(manifest)
    clean.setdefault("gateway", {}).setdefault("platforms", {})["uffici"] = {"enabled": True}
    drifted = copy.deepcopy(clean)
    drifted.update(copy.deepcopy(DRIFT_FIXTURE))
    drifted["memory"]["memory_enabled"] = True

    def read(config: dict) -> dict:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as home:
            Path(home, "config.yaml").write_text(as_yaml(config), encoding="utf-8")
            token = set_hermes_home_override(home)
            try:
                return drift.read_drift("uffici")
            finally:
                reset_hermes_home_override(token)

    # 🔴 A platform Hermes enables from its ENVIRONMENT is a live channel too (the e2e
    # rig's image connects `api_server` that way), and no config file can switch it off —
    # so the clean read is the baseline, and the fixture must add exactly `telegram`.
    baseline = read(clean)
    if baseline.get("mcpServers") != [] or baseline.get("localMemory") != []:
        return f"the setup's own configuration read as drift: {baseline!r}"
    if "channels" not in baseline:
        return "Hermes's gateway configuration could not be read, so no channel was"
    if "telegram" in baseline["channels"]:
        return f"the baseline already connects telegram, so the check proves nothing: {baseline!r}"
    report = read(drifted)
    want = {
        "mcpServers": ["probe-extra"],
        "channels": sorted([*baseline["channels"], "telegram"]),
        "localMemory": ["memory.memory_enabled"],
    }
    if report != want:
        return f"the fixture's drift read as {report!r}, not {want!r}"
    # The memory kind is the plugin's transcription of Hermes's switch: held to Hermes's.
    provider = importlib.import_module(f"{package}.memory").PROVIDER_DIRECTORY
    for config in (clean, drifted, {"memory": {"provider": provider}}):
        on = get_builtin_memory_store_flags(config)
        keys = drift.local_memory_keys(config)
        if on != tuple(key in keys for key in drift.BUILTIN_STORE_KEYS):
            return f"Hermes reads its own memory as {on}, the plugin as {keys!r}"
    return None


def freshness_is_hermes_routing(factory) -> str | None:
    """The plugin's ``fresh`` / ``resumed`` against a REAL session store (#2414).

    The suite's store double routes the way ``gateway/session.py`` does; this holds the
    double to Hermes: each way Hermes replaces a session at routing must read ``fresh``,
    and the session the plugin read must be the one the turn's own routing then lands on.
    """
    import asyncio
    import importlib
    import tempfile
    import threading

    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.session import SessionStore, build_session_key
    from hermes_cli import __version__ as hermes_version
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    package = factory.__module__.rsplit(".", 1)[0]
    support = importlib.import_module(f"{package}.support")
    if support.harness_support(hermes_version).get("supported") is not True:
        return (
            f"the probe's Hermes {hermes_version} is outside the plugin's {support.SUPPORTED_RANGE}"
        )

    if "touch_activity" not in inspect.signature(SessionStore.get_or_create_session).parameters:
        return "Hermes's routing call no longer takes touch_activity, which the plugin passes"

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as home:
        token = set_hermes_home_override(home)
        try:
            store = SessionStore(Path(home, "sessions"), GatewayConfig())
            # The plugin calls the store from a worker thread, as Hermes's own async facade
            # does: that is safe only while the store guards its index with a thread lock.
            if not isinstance(getattr(store, "_lock", None), type(threading.Lock())):
                return "Hermes's session store no longer guards its index with a thread lock"
            adapter = factory(PlatformConfig(enabled=True))
            adapter.set_session_store(store)
            source = adapter.build_source(chat_id="conv_probe", chat_type="dm")
            key = build_session_key(source)

            def fresh() -> bool:
                return asyncio.run(adapter._session_is_fresh(source))  # noqa: SLF001

            def a_turn() -> str:
                session_id = store.get_or_create_session(source).session_id
                for role in ("user", "assistant"):
                    store.append_to_transcript(session_id, {"role": role, "content": "probe"})
                return session_id

            if not fresh():
                return "a conversation Hermes has never seen read as resumed"
            # The turn creates a new conversation's session, so Hermes still sees it as new
            # (created_at == updated_at) and fires session:start.
            if store.lookup_by_session_key(key) is not None:
                return "the plugin's read created the session the turn should create"
            ran = a_turn()
            if fresh():
                return "a session with turns read as fresh"
            if store.get_or_create_session(source).session_id != ran:
                return "a healthy session did not route to itself"
            if not store.suspend_session(key):
                return "Hermes could not suspend the probe's session"
            if not fresh():
                return "a session /stop suspended read as resumed"
            entry = store.lookup_by_session_key(key)
            replaced = entry.session_id
            if replaced == ran:
                return "Hermes did not replace the suspended session at routing"
            # The turn's own new-session handling keys off this flag, which the plugin's
            # routing must leave for it to consume.
            if not getattr(entry, "was_auto_reset", False):
                return "the replaced session lost was_auto_reset before the turn could act on it"
            if store.get_or_create_session(source).session_id != replaced:
                return "the turn's routing landed on another session than the one read"
            ended = a_turn()
            store._db_for_session_id(ended).end_session(ended, "session_reset")  # noqa: SLF001
            if not fresh():
                return "a session ended in state.db read as resumed"
        finally:
            reset_hermes_home_override(token)
    return None


def harness_logs_reach_a_root_handler(factory) -> str | None:
    """A turn's INFO line, from Hermes's own logger, reaches the plugin's root handler."""
    import logging
    import tempfile

    import gateway.run_turn as run_turn

    if drifted := read_site(run_turn, 'logger = logging.getLogger("gateway.run")'):
        return drifted
    # 🔴 Hermes's setup opens file handlers under the home, and one can still be writing
    # when the directory goes: close what the setup added, and never let the probe's
    # cleanup be the verdict (a flake, found by the #2267 soak).
    root = logging.getLogger()
    before = list(root.handlers)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as home:
        try:
            return _root_handler_verdict(factory, home, root)
        finally:
            for added in [h for h in root.handlers if h not in before]:
                root.removeHandler(added)
                added.close()


def _root_handler_verdict(factory, home: str, root) -> str | None:
    import importlib
    import logging
    from pathlib import Path

    import hermes_logging

    # Hermes's own setup, at its own default level (a no-op if the gateway ran it).
    hermes_logging.setup_logging(hermes_home=Path(home), mode="gateway")
    if root.getEffectiveLevel() > logging.INFO:
        return f"Hermes sets the root level to {logging.getLevelName(root.level)}, above INFO"
    package = factory.__module__.rsplit(".", 1)[0]
    shipping = importlib.import_module(f"{package}.log_shipping")
    handler = shipping.UfficiLogHandler()
    handler.set_levels("info", "info")
    shipping.install(handler, root)
    try:
        logging.getLogger("gateway.run").info("uffici contract probe")
    finally:
        shipping.uninstall(handler, root)
    kept = handler.drain()
    probe = [e for e in kept if e["message"] == "uffici contract probe"]
    if not probe or probe[0]["source"] != "harness":
        return "an INFO line on Hermes's gateway.run logger never reached a root handler"
    return None


def media_reaches_the_adapters_hooks(factory) -> str | None:
    """Files in and out, in Hermes's own conventions (#2674).

    In: ``MessageEvent`` carries a file as a local path in ``media_urls`` with its MIME type
    in ``media_types``, and reads ``media_text_inlined`` False as "tell the agent to open
    it". Out: ``extract_media`` takes a ``MEDIA:<path>`` tag out of a reply, and Hermes's
    own delivery hands a document to ``send_document`` and an image (through its batch) to
    ``send_image_file`` — both the ADAPTER's, so the file becomes an attachment on the
    reply instead of Hermes's failure notice. Run through Hermes's own delivery method with
    the upload stubbed: the companion is not here, and the hooks are what is pinned.
    """
    import asyncio
    import dataclasses
    import importlib
    import tempfile

    import gateway.run_inbound as run_inbound
    from gateway.config import PlatformConfig
    from gateway.platforms import base
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.platforms.event import MessageEvent

    fields = {field.name for field in dataclasses.fields(MessageEvent)}
    if not {"media_urls", "media_types", "media_text_inlined"} <= fields:
        return "Hermes's MessageEvent no longer carries media_urls / media_types"
    if drifted := read_site(
        run_inbound, "display_name, agent_path, mtype, content_inlined=inline_flag is not False,"
    ):
        return drifted
    media, cleaned = BasePlatformAdapter.extract_media(
        "Here is the report.\nMEDIA:/tmp/uffici-probe/report.pdf"
    )
    if media != [("/tmp/uffici-probe/report.pdf", False)] or "MEDIA:" in cleaned:
        return f"extract_media no longer takes a MEDIA: tag out of a reply: {media!r}"
    package = factory.__module__.rsplit(".", 1)[0]
    files = importlib.import_module(f"{package}.files")
    if files.HERMES_IMAGE_TYPES != base.SUPPORTED_IMAGE_DOCUMENT_TYPES:
        return "the images Hermes shows a model are not the plugin's image table"
    adapter = factory(PlatformConfig(enabled=True))
    for name in ("send_document", "send_image_file"):
        ours, theirs = getattr(type(adapter), name), getattr(BasePlatformAdapter, name)
        if ours is theirs:
            return f"the adapter does not override {name}: a delivered file is a failure notice"
        wanted = inspect.signature(theirs).parameters
        have = inspect.signature(ours).parameters
        missing = [
            p for p in wanted if p not in have and wanted[p].kind is not wanted[p].VAR_KEYWORD
        ]
        if missing:
            return f"Hermes calls {name} with {missing}, which the adapter does not take"

    delivered: list[tuple[str, str | None]] = []

    async def attach(request_id: str, source: str, name: str | None = None):
        delivered.append((Path(source).name, request_id))
        return {"id": f"probe-{len(delivered)}", "name": Path(source).name, "ref": "r"}, None

    adapter._files.attach = attach  # noqa: SLF001 — the upload, without a companion
    adapter._open_request["probe"] = "probe-request"  # noqa: SLF001
    results: list = []
    with tempfile.TemporaryDirectory() as directory:
        pdf, png = Path(directory, "report.pdf"), Path(directory, "chart.png")
        pdf.write_bytes(b"%PDF-1.7")
        png.write_bytes(b"\x89PNG")
        event = SimpleNamespace(source=SimpleNamespace(chat_id="probe"))
        asyncio.run(
            adapter._deliver_media_attachments(  # noqa: SLF001 — Hermes's own delivery
                event,
                [(str(pdf), False), (str(png), False)],
                [],
                force_document_attachments=False,
                human_delay=0.0,
                metadata={},
                record_delivery=results.append,
            )
        )
    if sorted(delivered) != [("chart.png", "probe-request"), ("report.pdf", "probe-request")]:
        return f"Hermes's delivery did not reach the adapter's hooks with both files: {delivered!r}"
    if not all(getattr(result, "success", False) for result in results):
        return "Hermes recorded a delivered file as failed"
    outbox = adapter._outbox.get("probe-request") or {}  # noqa: SLF001
    if len(outbox.get("attachments", [])) != 2:
        return "the files Hermes delivered are not waiting for the request's post"
    return None


def workspace_is_a_media_root(manifest: dict) -> str | None:
    """The manifest makes the workspace a root Hermes delivers media from (#2674).

    ``uffici setup`` writes ``UFFICI_HERMES_MEDIA_DIR`` (the harness home and the manifest's
    workspace) into Hermes's environment and sets ``gateway.media_delivery_allow_dirs`` to
    a list naming it; Hermes expands ``${…}`` when it loads its configuration. Checked with
    Hermes's STRICT policy, where only a root (or a fresh file) is delivered: an old file in
    the workspace must be deliverable, and one beside it outside any root must not.
    """
    import os
    import tempfile

    from gateway.platforms.base import validate_media_delivery_path
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    variable = "UFFICI_HERMES_MEDIA_DIR"
    template = manifest.get("env", {}).get(variable)
    if not template or not manifest.get("workspace"):
        return "the manifest names no workspace for Hermes to deliver media from"
    previous = os.environ.get(variable)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as home:
        media_dir = template.replace("${home}", home)
        if Path(media_dir) != Path(home, manifest["workspace"]):
            return f"{variable} is not the workspace the companion records: {template!r}"
        Path(media_dir).mkdir(parents=True)
        config = configured(manifest)
        config.setdefault("gateway", {})["strict"] = True
        Path(home, "config.yaml").write_text(as_yaml(config), encoding="utf-8")
        inside, outside = Path(media_dir, "made.pdf"), Path(home, "elsewhere.pdf")
        for path in (inside, outside):
            path.write_bytes(b"%PDF-1.7")
            os.utime(path, (1, 1))
        os.environ[variable] = media_dir
        token = set_hermes_home_override(home)
        try:
            delivered_inside = validate_media_delivery_path(str(inside))
            delivered_outside = validate_media_delivery_path(str(outside))
        finally:
            reset_hermes_home_override(token)
            if previous is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = previous
    if delivered_outside is not None:
        return "strict policy delivered an old file outside every root: the probe proves nothing"
    if delivered_inside is None:
        return "Hermes would not deliver a file the agent saved in the Uffici workspace"
    return None


def brief_passes_admission(factory) -> str | None:
    """A brief is ADMITTED by Hermes's durable queue, and the field it closes by survives.

    Hermes 0.23 admits every inbound message into its session authority's queue and
    refuses one whose shape it does not accept — then answers "Hermes refused this
    message" instead of running the turn. A brief carries its request as ``message_id``
    and no ``metadata`` (#3124); this runs Hermes's own admission rule on that shape, and
    pins that ``message_id`` is a field the queue keeps.
    """
    import gateway.session_envelope as envelope
    from gateway.config import PlatformConfig
    from gateway.platforms.event import MessageEvent, MessageType

    adapter = factory(PlatformConfig(enabled=True))
    brief = MessageEvent(
        text="a member's message",
        message_type=MessageType.TEXT,
        source=adapter.build_source(chat_id="probe", chat_type="dm"),
        message_id="req_probe",
        channel_prompt="the organization's context",
        channel_context="why the agent is asked",
        allow_gateway_control=False,
    )
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False))
    if clause := envelope._event_clause(runner, brief, None, False):  # noqa: SLF001
        return f"Hermes's admission refuses the plugin's brief ({clause})"
    if "message_id" not in envelope._EVENT_FIELDS:  # noqa: SLF001
        return "Hermes's queue no longer keeps message_id, which names the brief's request"
    with_metadata = dataclasses.replace(brief, metadata={"uffici_request_id": "req_probe"})
    if envelope._event_clause(runner, with_metadata, None, False) != "metadata":  # noqa: SLF001
        return "Hermes's metadata rule moved: re-read what the queue admits before relying on it"
    return None


def queue_reconciles_through_hermes_own_paths(factory) -> str | None:
    """What the plugin reads and calls on Hermes's durable input queue (#3124).

    A brief's ``message_id`` names its queue row; a restart turns a started row
    ``unknown`` (never re-run) and leaves a ``queued`` one to run again; and the plugin
    discards a lost turn through the function Hermes's own ``/discard`` calls. Each read
    is pinned here, so a Hermes that reshapes its queue fails this probe instead of
    answering a person twice or pausing their conversation.
    """
    import gateway.session_authority as authority_module
    import gateway.slash_commands_session as commands
    import hermes_state_runtime as runtime
    from gateway.config import PlatformConfig
    from gateway.session import SessionStore
    from gateway.session_authorities import active_authority

    adapter = factory(PlatformConfig(enabled=True))
    package = type(adapter).__module__.rsplit(".", 1)[0]
    queue = sys.modules[f"{package}.hermes_queue"]
    for module, line in (
        (
            authority_module,
            "request_id = str(payload['native_text_v1']['event']['message_id'] "
            "or uuid.uuid4().hex)",
        ),
        (
            runtime,
            "UPDATE session_admissions SET status='unknown' WHERE status='started' "
            "AND owner_epoch!=?",
        ),
        (commands, "lost = [row for row in rows if row['status'] == 'unknown']"),
        # It RETURNS its confirmation for the command to reply with; it never sends it, so a
        # discard the plugin makes posts nothing into the conversation.
        (
            commands,
            'return t("gateway.discard.done", count=done) if done else '
            't("gateway.discard.nothing")',
        ),
    ):
        if drifted := read_site(module, line):
            return drifted
    if list(inspect.signature(commands.discard_lost_turns).parameters) != [
        "authority",
        "session_id",
    ]:
        return "Hermes's discard_lost_turns takes other arguments than the plugin passes"
    if "pending_only" not in inspect.signature(runtime.list_session_admissions).parameters:
        return "Hermes's list_session_admissions no longer filters pending rows"
    if not callable(getattr(SessionStore, "peek_session_id", None)):
        return "Hermes's session store has no peek_session_id for the plugin to read"
    if not callable(getattr(authority_module.SessionAuthority, "logical_owner", None)):
        return "Hermes's session authority has no logical_owner"
    if not callable(active_authority):
        return "Hermes no longer exposes active_authority"
    if queue.request_of(queue.delivery_id("req_probe", queue.new_delivery_token())) != "req_probe":
        return "a delivery's message_id no longer names its request"
    return None


def queue_notices_are_never_posted(factory) -> str | None:
    """Hermes's notices about its queue are told apart by the plugin, never posted (#3124).

    Runs Hermes's OWN producers: ``pause_notice`` for each reason it pauses with, and its
    queue-full reply; the refusal and dropped sentences are f-strings inside a method no
    probe can call, so their lines are pinned.
    """
    import gateway.session_ingress as ingress
    from gateway.config import PlatformConfig

    plugin = sys.modules[type(factory(PlatformConfig(enabled=True))).__module__]
    if plugin.hermes_queue_notice(ingress.QUEUE_FULL_REPLY) != "unanswered":
        return "Hermes's queue-full reply would reach the person as the agent's answer"
    for reason, kind in (
        ("runtime_draining", "saved"),
        ("session_busy", "saved"),
        ("unknown_execution", "unanswered"),
        ("route", "unanswered"),
    ):
        live = SimpleNamespace(
            pause_notified=False,
            source=SimpleNamespace(platform=SimpleNamespace(value="uffici")),
        )
        text = ingress.pause_notice(
            SimpleNamespace(sessions={"probe": live}), SimpleNamespace(session_id="probe"), reason
        )
        if plugin.hermes_queue_notice(text or "") != kind:
            return f"Hermes's {reason} pause notice would reach the person as the agent's answer"
    for line in (
        "return (f'⚠️ Hermes refused this message ({named}), so it was not saved '",
        "return (f'⚠️ Hermes did not answer this message ({exc.reason}): the conversation it was '",
    ):
        if drifted := read_site(ingress, line):
            return drifted
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python - <manifest json>", file=sys.stderr)
        return 2
    manifest = json.loads(argv[0])
    factory = registered_factory()
    checks = [
        ("follow-up hooks", lambda: follow_up_hooks_reach_the_adapter(factory)),
        ("restart notice", lambda: restart_notice_is_off(factory)),
        ("brief admission", lambda: brief_passes_admission(factory)),
        ("startup resume", lambda: startup_resume_reaches_the_adapter(factory)),
        ("queue reconcile", lambda: queue_reconciles_through_hermes_own_paths(factory)),
        ("queue notices", lambda: queue_notices_are_never_posted(factory)),
        ("restart cut", lambda: restart_cut_is_hermes_own_record(factory)),
        ("working line", lambda: working_line_is_off(manifest)),
        ("memory-review note", lambda: memory_note_is_off(manifest)),
        ("harness logs", lambda: harness_logs_reach_a_root_handler(factory)),
        ("context slot", lambda: context_reaches_the_system_side_only(factory)),
        ("platform tools", lambda: platform_tools_reach_the_model(factory, manifest)),
        ("memory provider", lambda: memory_provider_is_the_plugin(factory, manifest)),
        ("built-in memory", lambda: builtin_memory_is_off(manifest)),
        ("memory prefetch", lambda: prefetch_lands_fenced_on_the_user_side(factory)),
        ("harness drift", lambda: drift_reads_hermes_own_config(factory, manifest)),
        ("session freshness", lambda: freshness_is_hermes_routing(factory)),
        ("media hooks", lambda: media_reaches_the_adapters_hooks(factory)),
        ("media root", lambda: workspace_is_a_media_root(manifest)),
    ]
    for name, check in checks:
        failure = check()
        if failure:
            print(f"hermes contract: {name} FAILED — {failure}", file=sys.stderr)
            return 1
        print(f"hermes contract: {name} ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
