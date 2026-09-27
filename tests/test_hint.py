"""The AI-identity disclosure is not editable, and not paraphrasable.

🔴 EU AI Act Article 50(1): a person must know they are talking to an AI from
the first message. The platform owns this text; this plugin carries a verbatim
copy because the fence forbids importing it. A test that only checked "the hint
mentions AI" would pass a paraphrase that weakened it, so the assertion is on
the exact characters.
"""

from uffici_for_hermes.hint import AI_IDENTITY_LINES, PLATFORM_HINT

#: The platform's text, retyped here so a change to `hint.py` has to be made
#: TWICE to pass — once in the source and once in this expectation, which is
#: exactly the friction a disclosure deserves.
PLATFORM_AI_IDENTITY_LINES = (
    "You are an AI agent, and you never present yourself as a human: do not claim or imply",
    "that you are a person, and if anyone asks whether they are talking to a person or a",
    "machine, say plainly that you are an AI.",
)


def test_the_identity_lines_are_the_platforms_verbatim() -> None:
    assert AI_IDENTITY_LINES == PLATFORM_AI_IDENTITY_LINES, (
        "the identity lines have drifted from the platform's AI_IDENTITY_LINES; "
        "they are Article 50(1) material and are quoted, never rewritten"
    )


def test_the_hint_carries_every_identity_line() -> None:
    for line in PLATFORM_AI_IDENTITY_LINES:
        assert line in PLATFORM_HINT, f"the hint dropped the identity line {line!r}"


def test_the_hint_is_one_constant_with_no_configuration_path() -> None:
    """🔴 Structural, not a grep over the text.

    The module is asserted to define no function and import nothing: a module
    that is only constants has no configuration path *by construction*, and no
    later edit can add one — an ``import os`` or a ``def hint_for(...)`` fails
    here before it can read an environment variable or take an argument.
    """
    import ast
    import inspect

    import uffici_for_hermes.hint as hint

    tree = ast.parse(inspect.getsource(hint))
    offenders = [
        type(node).__name__
        for node in tree.body
        if not isinstance(node, ast.Assign | ast.AnnAssign | ast.Expr)
    ]
    assert not offenders, (
        f"hint.py contains {offenders} — it must be constants only, so the disclosure "
        "has no argument, no environment read and no setter to reach it through"
    )
    # Non-vacuity: the module must actually define the two constants, or the
    # assertion above is satisfied by an empty file.
    assigned = {
        target.id
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        for target in [node.target]
    }
    assert {"AI_IDENTITY_LINES", "PLATFORM_HINT"} <= assigned, (
        f"hint.py assigns {assigned}, so this test looked at the wrong module"
    )
