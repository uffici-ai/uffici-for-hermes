"""The README's install commands match what Hermes actually loads.

Hermes loads a plugin from ONE directory that is both an importable package and
the home of ``plugin.yaml``, and enables it by that manifest's ``name``. The
first README told readers to copy the repository root and enable ``uffici``:
Hermes discovered the plugin, refused it ("No __init__.py"), and the enable
command named nothing. So the README's two commands are pinned here to the
package directory and to the manifest, and a renamed manifest fails the README
before it fails a customer.

Since #3128 the by-hand install unpacks the RELEASE tarball (its manifest stamped
with the version) rather than copying a source checkout: so the release workflow
is pinned too, to stage exactly the package directory under the name the README
unpacks into Hermes's plugin directory.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
README = PROJECT_ROOT / "README.md"
RELEASE = PROJECT_ROOT / ".github" / "workflows" / "release.yml"
PACKAGE = PROJECT_ROOT / "src" / "uffici_for_hermes"
MANIFEST = PACKAGE / "plugin.yaml"


def _manifest_name() -> str:
    match = re.search(r"^name:\s*(\S+)\s*$", MANIFEST.read_text(encoding="utf-8"), re.M)
    assert match, f"{MANIFEST} declares no name"
    return match.group(1)


def _install_block() -> str:
    text = README.read_text(encoding="utf-8")
    section = text.split("## Installing it", 1)
    assert len(section) == 2, "README.md has no '## Installing it' section — the portal cites it"
    blocks = re.findall(r"```sh\n(.*?)```", section[1], re.S)
    by_hand = [b for b in blocks if "tar -xzf" in b]
    assert by_hand, "the install section has no by-hand sh block unpacking the release"
    return by_hand[0]


def test_readme_unpacks_the_release_package_directory_into_hermess_plugins() -> None:
    block = _install_block()
    assert (PACKAGE / "__init__.py").is_file(), "the package directory must be importable"
    assert MANIFEST.is_file(), "the package directory must hold the manifest"
    assert (
        'tar -xzf uffici-for-hermes.tar.gz -C "${HERMES_HOME:-$HOME/.hermes}/plugins"' in block
    ), block
    release = RELEASE.read_text(encoding="utf-8")
    # The tarball's one top directory IS the package directory, under the name it unpacks to.
    assert 'stage="$(mktemp -d)/uffici-for-hermes"' in release, "the release stages another name"
    assert 'cp -R src/uffici_for_hermes/. "$stage/"' in release, "the release stages another tree"
    assert "-czf dist/uffici-for-hermes.tar.gz uffici-for-hermes" in release, "another top dir"


def test_readme_enables_the_manifests_name() -> None:
    name = _manifest_name()
    block = _install_block()
    assert f"hermes plugins enable {name}\n" in block, f"README must enable {name!r}: {block}"
