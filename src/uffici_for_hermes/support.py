"""The Hermes versions this plugin release supports, and its verdict on the running one (#2414).

The adapter reaches into Hermes's internals — the session store's routing, the transcript
reader, the gateway's processing hooks — and those move between Hermes releases. A Hermes
outside the range may load and still misbehave without an error: before ``v2026.9.24`` its
session store reset an idle session by time inside the same routing call a suspended one
uses, which is how a conversation lost its history with nothing in any log.

So the range is declared here, checked when the plugin connects, logged when the running
Hermes is outside it, and sent in the hello's ``harnessSupport`` so ``uffici doctor`` shows
it. 🔴 NEVER a refusal: an unsupported Hermes still connects and answers — the doctor says
so, nothing stops. The range is the plugin's, not the companion's: only the plugin knows
the interfaces it was written against and how Hermes numbers its releases.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["MINIMUM", "SUPPORTED_RANGE", "UPPER_EXCLUSIVE", "harness_support", "parse_version"]

#: The oldest Hermes this release is written and tested against (README, ``dev/Dockerfile``).
MINIMUM = (0, 23, 0)
#: The first Hermes minor this release has not been tested against.
UPPER_EXCLUSIVE = (0, 24, 0)
#: The range in the words the hello carries (at most 64 characters on the wire).
SUPPORTED_RANGE = ">=0.23.0,<0.24"

_VERSION = re.compile(r"^v?(\d+)\.(\d+)(?:\.(\d+))?")


def parse_version(version: str) -> tuple[int, int, int] | None:
    """Hermes's ``__version__`` as numbers, or ``None`` when it is not a version at all.

    A pre-release or local suffix (``0.21.5rc1``, ``0.21.5+dev``) reads as its numbers —
    deliberately, although PEP 440 sorts ``0.21.5rc1`` below ``0.21.5``: the verdict only
    ever warns, and a release candidate of the tested Hermes is not the case to warn about.
    """
    match = _VERSION.match(str(version or "").strip())
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


def harness_support(version: str) -> dict[str, Any]:
    """The hello's ``harnessSupport``: the range, and the verdict when the version reads.

    🔴 A version that cannot be read carries NO verdict — the wire says so with an absent
    ``supported``, and the doctor says the plugin could not tell, rather than either side
    guessing.
    """
    support: dict[str, Any] = {"range": SUPPORTED_RANGE}
    parsed = parse_version(version)
    if parsed is not None:
        support["supported"] = MINIMUM <= parsed < UPPER_EXCLUSIVE
    return support
