"""The install manifest agrees with the package it describes.

``uffici-plugin.json`` rides at the top of every release tarball and is what
``uffici setup`` acts on: the plugin directory, the enable command, the
configuration keys, the environment the plugin reads. The companion validates
it against its own schema and refuses a field it does not know, so this side
pins the manifest to the PACKAGE: the name it enables is the manifest's name in
``plugin.yaml``, every environment variable it writes is one the plugin reads,
and the tracked version is the placeholder the release replaces.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "uffici-plugin.json"
PLUGIN_YAML = PROJECT_ROOT / "src" / "uffici_for_hermes" / "plugin.yaml"
WIRE_CLIENT = PROJECT_ROOT / "src" / "uffici_for_hermes" / "wire_client.py"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _plugin_yaml_name() -> str:
    match = re.search(r"^name:\s*(\S+)\s*$", PLUGIN_YAML.read_text(encoding="utf-8"), re.M)
    assert match, "plugin.yaml declares no name"
    return match.group(1)


def test_manifest_is_for_hermes_and_names_the_plugin_yaml_name() -> None:
    manifest = _manifest()
    assert manifest["manifestVersion"] == 1
    assert manifest["harness"] == "hermes"
    assert manifest["name"] == _plugin_yaml_name()
    assert manifest["enable"] == ["hermes", "plugins", "enable", manifest["name"]]


def test_tracked_version_is_the_placeholder_the_release_stamps() -> None:
    assert _manifest()["version"] == "0.0.0", (
        "the release stamps the version; the tracked file stays 0.0.0"
    )


def test_plugin_dir_is_relative_and_named_after_the_plugin() -> None:
    manifest = _manifest()
    plugin_dir = manifest["pluginDir"]
    assert not plugin_dir.startswith("/") and ".." not in plugin_dir.split("/")
    assert plugin_dir.endswith("/" + manifest["name"]), (
        "Hermes finds the plugin by its directory name"
    )


#: The variables the manifest writes for HERMES to read, not the plugin: each is named, as
#: ``${NAME}``, in a configuration value the manifest sets (Hermes expands ``${…}`` in its
#: configuration when it loads it).
HERMES_READ_ENV = {"UFFICI_HERMES_MEDIA_DIR"}


def test_every_env_the_manifest_writes_is_one_the_plugin_reads() -> None:
    manifest = _manifest()
    source = WIRE_CLIENT.read_text(encoding="utf-8")
    for name, template in manifest["env"].items():
        assert name.startswith("UFFICI_"), name
        if name in HERMES_READ_ENV:
            values = [entry["value"] for entry in manifest["config"]]
            assert any(f"${{{name}}}" in value for value in values), (
                f"{name} is written for Hermes but no configuration value reads it"
            )
            continue
        assert f'"{name}"' in source, (
            f"{name} is written by the installer but read nowhere in wire_client.py"
        )
        assert re.fullmatch(r"\$\{(wireSocket|wireTokenFile|companionHome|home)\}", template), (
            f"{name} uses a placeholder the companion does not provide: {template}"
        )
    needed = {"UFFICI_WIRE_SOCKET", "UFFICI_WIRE_TOKEN_FILE"}
    assert needed <= set(manifest["env"]), (
        "on one machine the plugin needs the socket and the token file, nothing less"
    )


def test_config_switches_the_platform_on() -> None:
    manifest = _manifest()
    keys = {entry["key"]: entry["value"] for entry in manifest["config"]}
    assert keys.get("gateway.platforms.uffici.enabled") == "true"
    assert manifest["configSet"] == ["hermes", "config", "set"]
    assert manifest["envPath"] == ["hermes", "config", "env-path"]


def test_config_silences_hermess_own_chatter() -> None:
    """Hermes sends its operational notices through the same hook as the agent's replies, so
    on Uffici they would read as the agent speaking: the "⏳ Working" heartbeat (switchable
    per platform) and the memory-review notice (switchable only for the whole Hermes; the
    review still runs). The restart notice is switched off by the adapter itself.

    🔴 Never ``display.background_process_notifications: off``: in Hermes that also stops
    INJECTING a finished process into the agent's session, which is the agent's work, not
    chatter."""
    keys = {entry["key"]: entry["value"] for entry in _manifest()["config"]}
    assert keys.get("display.platforms.uffici.long_running_notifications") == "false"
    assert keys.get("display.memory_notifications") == "off"
    assert "display.background_process_notifications" not in keys


def test_the_uffici_toolset_is_left_to_hermess_own_default() -> None:
    """🔴 No toolset setting is written. The agent's Uffici tools register in the ``uffici``
    toolset, and the pinned Hermes enables a plugin's toolset on every platform unless an
    operator's saved selection leaves it out (``hermes_cli/tools_config.py``,
    ``_enabled_plugin_toolsets``); naming it after the platform also puts it in Hermes's
    implicit ``hermes-uffici`` bundle. The only per-platform key, ``platform_toolsets.uffici``,
    REPLACES the whole selection — ``[hermes-uffici, uffici]`` there left the agent with no
    core tools at all, measured in the pinned image — and ``tools.tool_search`` is the whole
    Hermes's. ``dev/hermes_contract.py`` checks the default against the real Hermes."""
    keys = [entry["key"] for entry in _manifest()["config"]]
    assert not [
        k
        for k in keys
        if k.startswith(
            ("platform_toolsets", "known_plugin_toolsets", "tools.", "agent.disabled_toolsets")
        )
    ]


def test_the_workspace_is_a_root_hermes_delivers_media_from() -> None:
    """🔴 #2674: the manifest names the workspace (inside Hermes's home, recorded by
    ``uffici setup`` for the companion) and makes it one of Hermes's media roots, so a file
    the agent saves there is delivered even under Hermes's strict delivery policy.

    The pinned Hermes takes its roots from ``gateway.media_delivery_allow_dirs`` — a LIST,
    refused as a plain string by ``hermes config set`` — and expands ``${VAR}`` in its
    configuration when it loads it. A config value cannot name the harness home, so the
    path rides an environment variable the companion renders from ``${home}`` and the
    manifest's own workspace: the same path the companion records, by construction.
    ``dev/hermes_contract.py`` checks the whole chain against the real Hermes."""
    manifest = _manifest()
    workspace = manifest["workspace"]
    assert re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*(/[A-Za-z0-9_-][A-Za-z0-9._-]*)*", workspace)
    assert manifest["env"]["UFFICI_HERMES_MEDIA_DIR"] == "${home}/" + workspace
    keys = {entry["key"]: entry for entry in manifest["config"]}
    root = keys["gateway.media_delivery_allow_dirs"]
    assert json.loads(root["value"]) == ["${UFFICI_HERMES_MEDIA_DIR}"]
    assert "every platform" in root["note"], "a Hermes-wide key says so (the schema's note)"


def test_upgrade_restarts_hermes_only_through_its_own_service() -> None:
    """`uffici upgrade` runs `restartCommand` only when `restartService` names a
    unit that exists (#2250): without a service, `hermes gateway restart` runs the
    gateway in the foreground and would hang the upgrade. The names are Hermes's
    own for the default home (`hermes_cli/gateway.py` `get_service_name`,
    `get_launchd_plist_path`)."""
    manifest = _manifest()
    assert manifest["restartCommand"] == ["hermes", "gateway", "restart"]
    assert manifest["restartService"] == {
        "systemd": "hermes-gateway",
        "launchd": "ai.hermes.gateway",
    }


def test_manifest_carries_only_fields_the_companion_schema_knows() -> None:
    # The companion's schema is `additionalProperties: false`; this is its
    # field list, kept in step by hand and by the platform's parity guard.
    known = {
        "manifestVersion",
        "harness",
        "name",
        "version",
        "pluginDir",
        "workspace",
        "homeEnv",
        "defaultHome",
        "enable",
        "configSet",
        "config",
        "envPath",
        "envFile",
        "env",
        "restart",
        "restartCommand",
        "restartService",
    }
    unknown = set(_manifest()) - known
    assert not unknown, f"fields the companion would refuse: {sorted(unknown)}"


# ------------------------------------------------------------- the version it reports (#2236)


def test_the_plugin_reports_the_version_its_release_stamped(tmp_path: Path) -> None:
    """🔴 #2236: every release reported ``0.1.0`` — a constant — whatever was installed.

    The release writes ``uffici-plugin.json``, with the tag's version, BESIDE the package
    (the tarball's root is the package directory). That file is the one honest source.
    """
    from uffici_for_hermes.adapter import plugin_version

    (tmp_path / "uffici-plugin.json").write_text(json.dumps({"version": "0.4.0"}))
    assert plugin_version(tmp_path) == "0.4.0"


def test_a_source_checkout_says_it_is_one(tmp_path: Path) -> None:
    from uffici_for_hermes.adapter import DEV_VERSION, plugin_version

    assert plugin_version(tmp_path) == DEV_VERSION
    (tmp_path / "uffici-plugin.json").write_text("{not json")
    assert plugin_version(tmp_path) == DEV_VERSION
    (tmp_path / "uffici-plugin.json").write_text(json.dumps({"version": "x" * 65}))
    assert plugin_version(tmp_path) == DEV_VERSION, "a version the wire refuses is not sent"
    # The tracked placeholder is not a version either.
    (tmp_path / "uffici-plugin.json").write_text(json.dumps({"version": "0.0.0"}))
    assert plugin_version(tmp_path) == DEV_VERSION


def test_the_release_writes_the_manifest_beside_the_package() -> None:
    """What makes the lookup above find it: the Package step stages the PACKAGE directory
    and writes the stamped manifest into that same directory."""
    release = (PROJECT_ROOT / ".github" / "workflows" / "release.yml").read_text()
    assert 'cp -R src/uffici_for_hermes/. "$stage/"' in release
    assert '"$stage/uffici-plugin.json"' in release
