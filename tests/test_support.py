"""The Hermes versions this release supports, and the verdict the hello carries (#2414)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

import pytest

from test_connect import adapter_with
from uffici_for_hermes import support
from uffici_for_hermes.wire_client import plugin_hello

ROOT = Path(__file__).resolve().parent.parent
WIRE_V1 = Path(support.__file__).parent / "wire" / "v1"


@pytest.mark.parametrize(
    ("version", "verdict"),
    [
        ("0.23.0", True),
        ("0.23.9", True),
        ("v0.23.0", True),
        ("0.23.0rc1", True),
        ("0.23.0+dev", True),
        ("0.22.7", False),
        ("0.21.5", False),
        ("0.24.0", False),
        ("1.0.0", False),
        ("0.24", False),
    ],
)
def test_the_verdict_is_the_range(version: str, verdict: bool) -> None:
    assert support.harness_support(version) == {
        "range": support.SUPPORTED_RANGE,
        "supported": verdict,
    }


@pytest.mark.parametrize("version", ["unknown", "", "dev", "main-f97608f1"])
def test_a_version_that_cannot_be_read_carries_no_verdict(version: str) -> None:
    """🔴 Absent, never guessed: the doctor then says the plugin could not tell."""
    assert support.harness_support(version) == {"range": support.SUPPORTED_RANGE}


def test_the_range_words_are_the_range_numbers() -> None:
    """The words the doctor prints and the numbers the verdict uses are one statement."""
    low, high = re.fullmatch(r">=([\d.]+),<([\d.]+)", support.SUPPORTED_RANGE).groups()
    assert support.parse_version(low) == support.MINIMUM
    assert support.parse_version(high) == support.UPPER_EXCLUSIVE


def test_the_range_fits_the_wire() -> None:
    schema = json.loads((WIRE_V1 / "schema" / "hello.schema.json").read_text(encoding="utf-8"))
    rule = schema["$defs"]["pluginHello"]["properties"]["harnessSupport"]["properties"]["range"]
    assert rule["minLength"] <= len(support.SUPPORTED_RANGE) <= rule["maxLength"]


def test_the_hermes_the_readme_names_and_the_rig_pins_is_supported() -> None:
    """The README's Hermes and the dev rig's image are the release this plugin is tested
    against: a range that excluded them would mark the tested Hermes unsupported."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    named = re.search(r"\*\*Hermes (\d+\.\d+\.\d+)\*\*", readme)
    assert named, "the README no longer names the Hermes it is written against"
    assert support.harness_support(named.group(1))["supported"] is True
    assert f"`{support.SUPPORTED_RANGE}`" in readme, "the README does not state the range"
    dockerfile = (ROOT / "dev" / "Dockerfile").read_text(encoding="utf-8")
    release = re.search(r"release\s+`(v[\d.]+)`", readme)
    assert release and f"hermes-agent:{release.group(1)}" in dockerfile, (
        "the README's release is not the one the rig runs"
    )


def test_the_hello_carries_the_statement_only_when_given() -> None:
    statement = support.harness_support("0.23.0")
    hello = plugin_hello("t" * 40, "0.23.0", "0.1.0", 1000, None, statement)
    golden = json.loads(
        (WIRE_V1 / "golden" / "valid" / "hello-plugin-with-harness-support.json").read_text()
    )
    assert hello["harnessSupport"] == golden["harnessSupport"] == statement
    assert "harnessSupport" not in plugin_hello("t" * 40, "0.21.5", "0.1.0", 1000)


@pytest.mark.parametrize(("version", "warned"), [("0.21.5", True), ("0.23.0", False)])
def test_connect_says_an_unsupported_hermes_and_never_refuses_it(
    version: str, warned: bool, caplog
) -> None:
    """🔴 Said in the log and in the hello — and the connect still succeeds."""
    caplog.set_level(logging.WARNING)

    async def scenario():
        adapter, opened = adapter_with(
            UFFICI_WIRE_TOKEN="t" * 40, UFFICI_CONTEXT_WINDOW_TOKENS="100000"
        )
        adapter._harness_version = lambda: version  # noqa: SLF001
        assert await adapter.connect() is True, "an unsupported Hermes was refused"
        for _ in range(500):
            if opened and opened[0].sent:
                break
            await asyncio.sleep(0.01)
        await adapter.disconnect()
        return json.loads(opened[0].sent[0])

    hello = asyncio.run(scenario())
    assert hello["harnessSupport"] == {"range": support.SUPPORTED_RANGE, "supported": not warned}
    said = "outside the versions this plugin supports" in caplog.text
    assert said is warned


def test_the_plugin_still_parses_on_the_python_an_older_hermes_runs() -> None:
    """🔴 Never refused (#2414): on an older Hermes (Python 3.13 before 0.22) the plugin must
    still IMPORT, connect and say the Hermes is unsupported. One line of 3.14-only syntax —
    PEP 758's unparenthesized ``except A, B:``, which a formatter targeting 3.14 writes on its
    own — turns that into a SyntaxError at load, with nothing reported anywhere (#3124)."""
    import ast

    package = Path(support.__file__).parent
    sources = sorted(package.rglob("*.py"))
    assert len(sources) >= 15, "the plugin's sources were not found"
    for source in sources:
        ast.parse(source.read_text(encoding="utf-8"), str(source), feature_version=(3, 13))
