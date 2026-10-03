"""The plugin's copy of the wire contract is real, and reachable through ``WIRE_V1``.

🔴 The mirror image of the companion's ``TestTheContractIsEmbedded``. Without it
the only thing checking this copy is the platform's byte-for-byte parity test —
and that reads the files *by path*, never through ``WIRE_V1``. So a refactor of
that path expression, or a packaging change that drops the JSON out of the
wheel, would leave every gate green and the installed plugin unable to find its
own contract.
"""

import json

from uffici_for_hermes.wire import WIRE_V1

#: Ten frame types plus the entry and the brief's context. `ready` joined them in
#: #2014 (a plugin is connected before its harness can answer); `call` and `result`
#: in #2028, the generic tunnel that keeps new platform operations off the wire's
#: version; `context` in #2331, the place's context every brief carries.
EXPECTED_SCHEMAS = 12


def test_wire_v1_resolves_to_a_real_directory() -> None:
    assert WIRE_V1.is_dir(), (
        f"{WIRE_V1} is not a directory — the plugin cannot find its own copy of the "
        "wire contract, however green the parity test is"
    )


def test_every_schema_is_present() -> None:
    schemas = sorted(p.name for p in (WIRE_V1 / "schema").glob("*.schema.json"))
    assert len(schemas) == EXPECTED_SCHEMAS, (
        f"found {len(schemas)} schema(s) in the plugin's copy, "
        f"expected {EXPECTED_SCHEMAS}: {schemas}"
    )


def test_the_goldens_are_present() -> None:
    for kind in ("valid", "invalid"):
        goldens = list((WIRE_V1 / "golden" / kind).glob("*.json"))
        # Non-vacuity: an empty directory would make a "they all parse" loop
        # below assert nothing.
        assert goldens, f"the plugin's copy carries no {kind} goldens"
        for golden in goldens:
            json.loads(golden.read_text(encoding="utf-8"))


def test_the_manifest_agrees_with_the_files_beside_it() -> None:
    manifest = json.loads((WIRE_V1 / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["version"] == 1

    valid = list((WIRE_V1 / "golden" / "valid").glob("*.json"))
    invalid = list((WIRE_V1 / "golden" / "invalid").glob("*.json"))
    assert len(valid) == manifest["totals"]["valid"], (
        f"the manifest counts {manifest['totals']['valid']} valid golden(s), "
        f"the copy has {len(valid)}"
    )
    assert len(invalid) == manifest["totals"]["invalid"], (
        f"the manifest counts {manifest['totals']['invalid']} invalid golden(s), "
        f"the copy has {len(invalid)}"
    )
    assert len(manifest["schemas"]) == EXPECTED_SCHEMAS
