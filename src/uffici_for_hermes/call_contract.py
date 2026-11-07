"""The call tunnel's fixed vocabulary, and the one way a frame is encoded (#2028).

A module of its own because two others need it — the call client builds and
measures calls, the wire client sends them and bounds results — and the size a
call is MEASURED at must be the size it is SENT at, byte for byte. One encoder is
what makes that true.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: One ``call`` or ``result`` frame, encoded. The companion's decoder enforces
#: the same number; JSON Schema cannot state a byte ceiling.
MAX_CALL_PAYLOAD_BYTES = 262_144

#: The contract's closed vocabulary, in the schema's order.
CALL_ERROR_CLASSES = (
    "unknown_op",
    "not_permitted",
    "timed_out",
    "upstream_failed",
    "malformed_payload",
)

#: ``call.op``: an open vocabulary, but a NAME — the schema's pattern, which is
#: also what keeps free text out of the one log field a caller chooses.
OP_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
MAX_CALL_ID_CHARS = 64
MAX_DEADLINE_MS = 300_000

#: The sentence each class relays to the person (decision 32).
#: 🔴 The companion's own wording, pinned from the platform repository by
#: ``uffici-call-instructions-parity.test.mjs``: the person must
#: hear the same thing whichever side answered the call.
INSTRUCTIONS = {
    "unknown_op": (
        "This action is not available on this connection; tell the person it could not be done."
    ),
    "not_permitted": (
        "This action is not permitted for this agent; tell the person it could not be done."
    ),
    "timed_out": "The platform did not answer in time; tell the person, and try again shortly.",
    "upstream_failed": (
        "The platform could not complete this; tell the person, and try again shortly."
    ),
    "malformed_payload": (
        "The request was too large or could not be read; tell the person it could not be done."
    ),
}


def encode_frame(frame: dict[str, Any]) -> str:
    """The wire encoding: compact, and UTF-8 rather than ``\\uXXXX`` escapes.

    🔴 The only encoder. A frame measured with one encoder and sent with another
    can fit the bound here and break it on the wire — ``ensure_ascii`` alone
    turns one accented character into six bytes.
    """
    return json.dumps(frame, separators=(",", ":"), ensure_ascii=False)


def encoded_size(frame: dict[str, Any]) -> int:
    """The frame's size on the wire, in bytes."""
    return len(encode_frame(frame).encode("utf-8"))
