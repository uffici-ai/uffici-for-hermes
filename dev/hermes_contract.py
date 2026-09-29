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


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python - <manifest json>", file=sys.stderr)
        return 2
    manifest = json.loads(argv[0])
    factory = registered_factory()
    checks = [
        ("follow-up hooks", lambda: follow_up_hooks_reach_the_adapter(factory)),
        ("restart notice", lambda: restart_notice_is_off(factory)),
        ("working line", lambda: working_line_is_off(manifest)),
        ("memory-review note", lambda: memory_note_is_off(manifest)),
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
