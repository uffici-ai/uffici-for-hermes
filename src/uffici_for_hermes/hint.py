"""The platform hint Hermes injects into the system prompt.

🔴 The identity lines are QUOTED VERBATIM from the platform's
``apps/conversations-api/src/lib/channel-frame.ts`` (``AI_IDENTITY_LINES``) and never
imported: the `uffici/` fence forbids this child reading the platform, and this
file is exported to its own repository where the platform does not exist.

They are EU AI Act Article 50(1) material — a person must know they are talking
to an AI from the first message. There is deliberately no configuration path
that can remove or edit them: a setting that could switch them off would be a
setting that switches off the disclosure, and an agent able to reach it would be
authorizing itself.
"""

#: Verbatim from the platform's ``AI_IDENTITY_LINES``. Changing a character here
#: is changing the platform's disclosure, so the test pins the exact text.
AI_IDENTITY_LINES: tuple[str, ...] = (
    "You are an AI agent, and you never present yourself as a human: do not claim or imply",
    "that you are a person, and if anyone asks whether they are talking to a person or a",
    "machine, say plainly that you are an AI.",
)

#: What the platform is, for the model. Stable across a session, which is why it
#: belongs in the system prompt rather than in any per-turn slot (decision 43).
_PLATFORM_LINES: tuple[str, ...] = (
    "You are working in Uffici, an office for AI agents, through a conversation"
    " its members can read.",
    "Other people and other agents may be reading and writing in the same"
    " conversation; address an agent by its @handle when you mean it.",
)

#: The whole hint, assembled once.
PLATFORM_HINT: str = "\n".join((*_PLATFORM_LINES, *AI_IDENTITY_LINES))
