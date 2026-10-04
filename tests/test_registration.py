"""What the plugin tells Hermes about itself at startup."""

from __future__ import annotations

from typing import Any

import pytest

from uffici_for_hermes import register
from uffici_for_hermes.hint import AI_IDENTITY_LINES


class RecordingCtx:
    """Hermes's registration context, as much of it as this plugin uses.

    🔴 ``register_platform`` in the pinned source takes ``check_fn`` as a
    REQUIRED parameter and raises ``TypeError`` on any keyword ``PlatformEntry``
    does not declare. This double enforces both, so a registration that would
    fail inside Hermes fails here instead.
    """

    #: Every keyword ``PlatformEntry`` declares at the pinned commit.
    KNOWN = {
        "name",
        "label",
        "adapter_factory",
        "check_fn",
        "validate_config",
        "ensure_deps_fn",
        "is_connected",
        "required_env",
        "install_hint",
        "setup_fn",
        "source",
        "plugin_name",
        "allowed_users_env",
        "allow_all_env",
        "max_message_length",
        "pii_safe",
        "emoji",
        "allow_update_command",
        "platform_hint",
        "env_enablement_fn",
        "apply_yaml_config_fn",
        "cron_deliver_env_var",
        "parse_target_ref_fn",
        "standalone_sender_fn",
    }

    def __init__(self) -> None:
        self.entry: dict[str, Any] = {}

    def register_platform(self, name: str, label: str, adapter_factory, check_fn, **kwargs) -> None:
        unknown = set(kwargs) - self.KNOWN
        if unknown:
            raise TypeError(f"PlatformEntry got unexpected keyword(s): {sorted(unknown)}")
        self.entry = {
            "name": name,
            "label": label,
            "adapter_factory": adapter_factory,
            "check_fn": check_fn,
            **kwargs,
        }


@pytest.fixture
def entry() -> dict[str, Any]:
    ctx = RecordingCtx()
    register(ctx)
    assert ctx.entry, "register() registered nothing"
    return ctx.entry


def test_the_platform_registers_under_its_own_name(entry: dict[str, Any]) -> None:
    assert entry["name"] == "uffici" and entry["label"] == "Uffici"


def test_a_passive_probe_is_supplied_and_answers_without_installing(entry: dict[str, Any]) -> None:
    """🔴 ``check_fn`` is required, and Hermes calls it from status displays —
    so it must answer, and must never install anything."""
    assert callable(entry["check_fn"])
    assert entry["check_fn"]() is True


def test_no_length_cap_is_imposed_on_top_of_the_platforms(entry: dict[str, Any]) -> None:
    """🔴 ``0`` means NO LIMIT in Hermes's registry.

    The platform already bounded the transcript to the budget this plugin
    declared in ``hello``; a second cap here would chop a reply the platform
    sized. Note this is a REGISTRY field — an attribute of the same name on the
    adapter would be read by nothing (the adapter-side knob is
    ``MAX_MESSAGE_LENGTH``, where ``0`` coerces to 4096).
    """
    assert entry["max_message_length"] == 0


def test_the_hint_carries_the_disclosure_verbatim(entry: dict[str, Any]) -> None:
    """One of the five named mutations: change the hint's text and this fails."""
    for line in AI_IDENTITY_LINES:
        assert line in entry["platform_hint"]


def test_neither_an_allowlist_nor_an_allow_all_switch_is_registered(
    entry: dict[str, Any],
) -> None:
    """🔴 Authorization is upstream. Registering either of these would put a way
    to widen access to conversations the platform never granted."""
    assert "allowed_users_env" not in entry
    assert "allow_all_env" not in entry


def test_the_factory_builds_an_adapter_that_declares_upstream_authorization(
    entry: dict[str, Any],
) -> None:
    from conftest import PlatformConfig

    adapter = entry["adapter_factory"](PlatformConfig(enabled=True))
    assert adapter.authorization_is_upstream is True
    assert adapter.platform.value == "uffici"


def test_an_unknown_keyword_would_be_refused() -> None:
    """Non-vacuity for the double: prove it actually rejects, so
    ``test_the_platform_registers_under_its_own_name`` passing means something."""
    ctx = RecordingCtx()
    with pytest.raises(TypeError):
        ctx.register_platform(
            name="x",
            label="X",
            adapter_factory=lambda c: None,
            check_fn=lambda: True,
            not_a_real_field=1,
        )


def test_is_connected_answers_false_without_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from conftest import PlatformConfig
    from uffici_for_hermes.adapter import is_connected

    for name in ("UFFICI_WIRE_TOKEN", "UFFICI_WIRE_TOKEN_FILE", "UFFICI_WIRE_SOCKET"):
        monkeypatch.delenv(name, raising=False)
    assert is_connected(PlatformConfig(enabled=True)) is False


def test_is_connected_answers_true_once_a_token_is_there(monkeypatch: pytest.MonkeyPatch) -> None:
    from conftest import PlatformConfig
    from uffici_for_hermes.adapter import is_connected

    monkeypatch.setenv("UFFICI_WIRE_TOKEN", "t" * 40)
    assert is_connected(PlatformConfig(enabled=True)) is True
