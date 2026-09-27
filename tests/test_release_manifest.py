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


def test_every_env_the_manifest_writes_is_one_the_plugin_reads() -> None:
    manifest = _manifest()
    source = WIRE_CLIENT.read_text(encoding="utf-8")
    for name, template in manifest["env"].items():
        assert name.startswith("UFFICI_"), name
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


def test_manifest_carries_only_fields_the_companion_schema_knows() -> None:
    # The companion's schema is `additionalProperties: false`; this is its
    # field list, kept in step by hand and by the platform's parity guard.
    known = {
        "manifestVersion", "harness", "name", "version", "pluginDir", "homeEnv",
        "defaultHome", "enable", "configSet", "config", "envPath", "envFile", "env", "restart",
    }
    unknown = set(_manifest()) - known
    assert not unknown, f"fields the companion would refuse: {sorted(unknown)}"
