"""The budget is the harness's answer, never the plugin's guess."""

import pytest

from uffici_for_hermes.budget import (
    DEFAULT_CONTEXT_SHARE,
    SHARE_ENV,
    TOKENIZER_DRIFT_MARGIN,
    WINDOW_ENV,
    BudgetUnavailable,
    context_budget_tokens,
    resolve_share,
    resolve_window_tokens,
)


def env(**values: str):
    return lambda name: values.get(name)


def test_the_budget_is_the_share_of_the_window_less_the_margin() -> None:
    assert context_budget_tokens(100_000, 0.25) == int(100_000 * 0.25 * TOKENIZER_DRIFT_MARGIN)


def test_a_tiny_window_still_yields_a_usable_budget() -> None:
    # The wire's schema requires at least 1; a floor of 0 would make hello invalid.
    assert context_budget_tokens(1, 0.01) == 1


def test_the_default_share_applies_when_the_operator_says_nothing() -> None:
    assert resolve_share(env()) == DEFAULT_CONTEXT_SHARE


def test_an_operator_share_is_honoured() -> None:
    assert resolve_share(env(**{SHARE_ENV: "0.5"})) == 0.5


@pytest.mark.parametrize("bad", ["half", "0", "-0.2", "1.5", "  "])
def test_a_share_that_is_not_a_fraction_is_refused_with_its_value(bad: str) -> None:
    if not bad.strip():
        assert resolve_share(env(**{SHARE_ENV: bad})) == DEFAULT_CONTEXT_SHARE
        return
    with pytest.raises(BudgetUnavailable) as caught:
        resolve_share(env(**{SHARE_ENV: bad}))
    assert SHARE_ENV in str(caught.value)


def test_the_operators_stated_window_beats_every_probe() -> None:
    """An operator who sets it is correcting a probe that got it wrong."""
    called = []
    probe = lambda: (called.append(1), 999)[1]  # noqa: E731
    assert resolve_window_tokens(env(**{WINDOW_ENV: "128000"}), [probe]) == 128_000
    assert not called, "a probe ran even though the operator stated the window"


def test_the_first_positive_probe_answers() -> None:
    probes = [lambda: None, lambda: 0, lambda: 64_000, lambda: 32_000]
    assert resolve_window_tokens(env(), probes) == 64_000


def test_a_probe_that_raises_is_skipped_rather_than_fatal() -> None:
    """🔴 The probes reach into the harness's internals, which move between
    versions. One that has moved must cost a fallback, never a refusal to start
    on a harness that is otherwise working."""

    def moved() -> int:
        raise AttributeError("module 'agent' has no attribute 'context_length'")

    assert resolve_window_tokens(env(), [moved, lambda: 48_000]) == 48_000


def test_a_boolean_is_not_a_context_window() -> None:
    # bool is an int in Python; True would silently become a one-token window.
    assert resolve_window_tokens(env(), [lambda: True, lambda: 8_000]) == 8_000


def test_no_window_anywhere_refuses_rather_than_inventing_one() -> None:
    """🔴 The whole point. A plugin that shipped a number would be telling the
    platform about a model it has never seen, and the platform would believe it.
    """
    with pytest.raises(BudgetUnavailable) as caught:
        resolve_window_tokens(env(), [lambda: None])
    assert WINDOW_ENV in str(caught.value)


@pytest.mark.parametrize("bad", ["lots", "-1", "0"])
def test_a_stated_window_that_is_not_a_window_is_refused(bad: str) -> None:
    with pytest.raises(BudgetUnavailable):
        resolve_window_tokens(env(**{WINDOW_ENV: bad}), [])


def test_the_plugin_holds_no_window_constant() -> None:
    """Non-vacuity for the refusal above, so "it refuses" cannot quietly become
    "it defaults".

    🔴 Over the AST's numeric literals, not the source text: a docstring is free
    to NAME a context window while explaining why the module must not hold one,
    and a grep cannot tell the explanation from the mistake.
    """
    import ast
    import inspect

    import uffici_for_hermes.budget as budget

    tree = ast.parse(inspect.getsource(budget))
    numbers = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ]
    # Non-vacuity: the module does contain integer literals (the bounds checks),
    # so an empty list would mean this walked the wrong tree.
    assert numbers, "no integer literals found at all; this test walked the wrong module"
    plausible_windows = [n for n in numbers if n >= 1024]
    assert not plausible_windows, (
        f"budget.py contains {plausible_windows}, which are context-window sized; "
        "the plugin holds a share and a margin, never a window"
    )
