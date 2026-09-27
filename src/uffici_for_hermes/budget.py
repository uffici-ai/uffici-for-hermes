"""How much of Hermes's context window the platform may fill.

🔴 The plugin holds a SHARE and a MARGIN. It never holds a window.

The number in ``hello`` is what the platform sizes its reads to, and it has to
be Hermes's own answer: a plugin that shipped "32768" would be telling the
platform about a model it has never seen, and the first operator to run a
different one would get briefs that overflow the turn — or, worse, briefs far
smaller than the window, silently paying for nothing. So the window is resolved
from the harness at connect time, and when it cannot be resolved the plugin
REFUSES rather than inventing one: `hello` without a truthful budget is worse
than no connection, because the platform would trust it.
"""

from collections.abc import Callable, Iterable

#: The fraction of Hermes's window the platform may fill with transcript.
#: The rest is the harness's own: its system prompt, its tools, its reasoning
#: and the reply it is about to write. A quarter is deliberately conservative —
#: the platform bounds to what it is told, so being wrong downward costs a
#: shorter delta and being wrong upward costs a failed turn.
DEFAULT_CONTEXT_SHARE = 0.25

#: 🔴 Applied on top of the share, for TOKENIZER DRIFT. The platform estimates
#: tokens with its own estimator and the harness counts them with the model's
#: real tokenizer; the two disagree, and they disagree most on exactly the
#: content this carries — names, handles, punctuation-dense tool lines. Ten per
#: cent of headroom is cheaper than a turn that fails at the provider.
TOKENIZER_DRIFT_MARGIN = 0.9

#: The environment variable an operator uses to change the share.
SHARE_ENV = "UFFICI_CONTEXT_SHARE"

#: An operator's explicit statement of the window, for a harness whose own
#: configuration cannot be read. Not a default — a fact the operator supplies.
WINDOW_ENV = "UFFICI_CONTEXT_WINDOW_TOKENS"


class BudgetUnavailable(RuntimeError):
    """Hermes's context window could not be learned, so no honest hello exists."""


def resolve_share(get_env: Callable[[str], str | None]) -> float:
    """The configured share, refusing a value that is not a usable fraction."""
    raw = (get_env(SHARE_ENV) or "").strip()
    if not raw:
        return DEFAULT_CONTEXT_SHARE
    try:
        share = float(raw)
    except ValueError as exc:
        raise BudgetUnavailable(
            f"{SHARE_ENV} is {raw!r}, which is not a number; it is a fraction of the "
            f"harness's context window, for example 0.25"
        ) from exc
    if not 0 < share <= 1:
        raise BudgetUnavailable(
            f"{SHARE_ENV} is {share}, which is not a fraction between 0 and 1. The platform's "
            f"share of the window cannot be none of it, nor more than all of it"
        )
    return share


def resolve_window_tokens(
    get_env: Callable[[str], str | None],
    probes: Iterable[Callable[[], int | None]] = (),
) -> int:
    """Hermes's context window, from the operator or from the harness itself.

    The operator's explicit statement wins, because an operator who sets it is
    correcting something the probes got wrong. Otherwise each probe is asked in
    turn and the first positive answer is taken.

    🔴 A probe that raises is SKIPPED, not fatal. They reach into the harness's
    internals, which move between versions; one that has moved must cost this
    plugin a fallback, never a refusal to start on a harness that is working.
    """
    raw = (get_env(WINDOW_ENV) or "").strip()
    if raw:
        try:
            stated = int(raw)
        except ValueError as exc:
            raise BudgetUnavailable(
                f"{WINDOW_ENV} is {raw!r}, which is not a whole number of tokens"
            ) from exc
        if stated <= 0:
            raise BudgetUnavailable(f"{WINDOW_ENV} is {stated}, which is not a context window")
        return stated

    for probe in probes:
        try:
            answer = probe()
        except Exception:  # noqa: BLE001 — a moved internal is a fallback, never a crash
            continue
        if isinstance(answer, int) and not isinstance(answer, bool) and answer > 0:
            return answer

    raise BudgetUnavailable(
        "the harness's context window could not be read, so there is no honest budget to "
        f"declare. Set {WINDOW_ENV} to the model's context window in tokens"
    )


def context_budget_tokens(window_tokens: int, share: float) -> int:
    """The share of the window, less the drift margin, at least one token."""
    return max(1, int(window_tokens * share * TOKENIZER_DRIFT_MARGIN))
