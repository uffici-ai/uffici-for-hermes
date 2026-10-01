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

import inspect
import json
import sys
from types import SimpleNamespace


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
    import gateway.run_turn as run_turn
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
        run_turn, "if session_key and _should_clear_resume_pending_after_turn(agent_result):"
    ):
        return drifted
    adapter = factory(PlatformConfig(enabled=True))
    event = MessageEvent(
        text="", message_type=MessageType.TEXT,
        source=adapter.build_source(chat_id="probe", chat_type="dm"), internal=True,
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


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python - <manifest json>", file=sys.stderr)
        return 2
    manifest = json.loads(argv[0])
    factory = registered_factory()
    checks = [
        ("follow-up hooks", lambda: follow_up_hooks_reach_the_adapter(factory)),
        ("restart notice", lambda: restart_notice_is_off(factory)),
        ("startup resume", lambda: startup_resume_reaches_the_adapter(factory)),
        ("restart cut", lambda: restart_cut_is_hermes_own_record(factory)),
        ("working line", lambda: working_line_is_off(manifest)),
        ("memory-review note", lambda: memory_note_is_off(manifest)),
        ("harness logs", lambda: harness_logs_reach_a_root_handler(factory)),
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
