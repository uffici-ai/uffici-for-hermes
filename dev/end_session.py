"""End one conversation's Hermes session, as the fixture's ``endHarnessSession`` step asks.

The companion logs ``end the harness session`` with the conversation's id, and the run's
driver pipes this file into the RUNNING gateway's container::

    docker exec -i <hermes> /opt/hermes/.venv/bin/python - <conversation id> < end_session.py

It ends every live row Hermes keeps for that conversation in ``state.db``, through Hermes's
own ``SessionDB.end_session``, with ``session_reset`` — an explicit boundary Hermes never
reopens. That is how a session is replaced under a live gateway: its index still holds the
entry, and only the next message's routing reads the ended row and creates a new, empty
session. The plugin must then answer that brief ``fresh`` and earn the opener, which is what
the step after this one asserts.

Prints how many rows it ended and exits 1 when there were none: a step that ended nothing
would leave the next ring proving nothing about a replaced session.
"""

from __future__ import annotations

import contextlib
import sqlite3
import sys

PLATFORM = "uffici"
END_REASON = "session_reset"


def main(argv: list[str]) -> int:
    if len(argv) != 1 or not argv[0]:
        print("usage: python - <conversation id>", file=sys.stderr)
        return 2
    conversation = argv[0]

    from hermes_state import SessionDB

    db = SessionDB()
    # A read through a separate read-only connection: the selection is ours, the write is
    # Hermes's own.
    with contextlib.closing(sqlite3.connect(f"file:{db.db_path}?mode=ro", uri=True)) as reader:
        live = [
            row[0]
            for row in reader.execute(
                "SELECT id FROM sessions WHERE source = ? AND chat_id = ? AND ended_at IS NULL",
                (PLATFORM, conversation),
            )
        ]
    for session_id in live:
        db.end_session(session_id, END_REASON)
    print(f"end session: {len(live)} live session(s) of conversation {conversation} ended")
    if not live:
        print(
            f"end session: Hermes holds no live session for conversation {conversation}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
