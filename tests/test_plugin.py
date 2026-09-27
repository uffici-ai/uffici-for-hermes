"""The plugin's own identity, asserted rather than assumed."""

from importlib.metadata import PackageNotFoundError

import pytest

import uffici_for_hermes
from uffici_for_hermes import DEV_VERSION, PLUGIN_NAME, version


def test_plugin_name_matches_the_child_directory() -> None:
    # The mirror pushes this project onto a repository of the same name
    # (epic #1994 decision 34), and Hermes loads the plugin by it.
    assert PLUGIN_NAME == "uffici-for-hermes"


def test_version_reports_the_installed_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(uffici_for_hermes, "distribution_version", lambda name: "1.2.3")
    assert version() == "1.2.3"


def test_version_falls_back_to_dev_outside_an_installed_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 🔴 The branch a source checkout takes — the one the platform's own check
    # runs. Without the fallback this raises, and `version()` is unusable
    # exactly where a developer first calls it.
    def missing(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(uffici_for_hermes, "distribution_version", missing)
    assert version() == DEV_VERSION
