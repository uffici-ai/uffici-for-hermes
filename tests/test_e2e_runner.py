"""The release e2e's judgement: a green run is a run that proved something.

`dev/e2e.py` runs the scripted conversation against a released companion; the
docker half needs a real engine and runs in `.github/workflows/e2e.yml`. What
decides pass or fail is pure, and a pass that asserted nothing is the failure the
gate exists to catch — so it is pinned here, inside this project's own check.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parent.parent / "dev" / "e2e.py"


def _load():
    spec = importlib.util.spec_from_file_location("uffici_e2e_runner", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e2e = _load()


def _log(*lines: dict) -> list[dict]:
    return e2e.log_lines("\n".join(json.dumps(line) for line in lines))


VERDICT = {"msg": e2e.SCRIPT_COMPLETE, "ringsDelivered": 7, "briefs": 3, "unmet": 0}


def test_a_real_pass_passes() -> None:
    closed = {"msg": e2e.CONNECTION_CLOSED, "reason": "shutting_down"}
    assert e2e.verdict_failures(_log(VERDICT, closed), 0) == []


def test_no_verdict_is_a_failure_even_on_exit_zero() -> None:
    assert e2e.verdict_failures(_log({"msg": e2e.PLUGIN_READY}), 0) == [
        "the companion never reported a verdict"
    ]


@pytest.mark.parametrize("count", ["ringsDelivered", "briefs"])
def test_a_verdict_over_nothing_is_a_failure(count: str) -> None:
    failures = e2e.verdict_failures(_log({**VERDICT, count: 0}), 0)
    assert failures == [
        f"the script reported {count}=0 — every expectation held over nothing at all"
    ]


def test_a_dropped_connection_fails_a_run_whose_verdict_held() -> None:
    dropped = {"msg": e2e.CONNECTION_CLOSED, "reason": "protocol_error"}
    failures = e2e.verdict_failures(_log(VERDICT, dropped), 0)
    assert failures == ["the companion dropped the plugin's connection 1 time(s): protocol_error"]


def test_a_non_zero_exit_names_the_unmet_count() -> None:
    failures = e2e.verdict_failures(_log({**VERDICT, "unmet": 2}), 1)
    assert failures == ["the scripted conversation had 2 unmet expectation(s)"]


def test_log_lines_drops_what_is_not_a_json_object() -> None:
    text = 'plain text\n[1, 2]\n{"msg": "plugin ready"}\n'
    assert e2e.log_lines(text) == [{"msg": "plugin ready"}]


def test_the_wire_token_is_redacted() -> None:
    redacted = e2e.redact("token=abc123 again abc123", "abc123")
    assert redacted == "token=«wire-token» again «wire-token»"
    assert e2e.redact("nothing to hide", "") == "nothing to hide"


@pytest.mark.parametrize("requested", ["1.2.3", "v1.2.3"])
def test_an_explicit_version_needs_no_network(requested: str) -> None:
    assert e2e.resolve_version(requested) == "1.2.3"


@pytest.mark.parametrize("requested", ["main", "1.2", "1.2.3-rc1", ""])
def test_anything_but_a_release_version_is_refused(requested: str) -> None:
    with pytest.raises(e2e.Failure):
        e2e.resolve_version(requested)


def test_latest_resolves_through_the_release_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    asked = []

    def fake_fetch(url: str) -> bytes:
        asked.append(url)
        return b'{"tag_name": "v0.3.0"}'

    monkeypatch.setattr(e2e, "fetch", fake_fetch)
    assert e2e.resolve_version("latest") == "0.3.0"
    assert asked == ["https://api.github.com/repos/uffici-ai/uffici-companion/releases/latest"]


def test_the_fixture_comes_from_the_same_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    asked = []
    monkeypatch.setattr(e2e, "fetch", lambda url: asked.append(url) or b'{"v": 1}')
    e2e.fetch_fixture("0.3.0", tmp_path)
    assert asked == [
        "https://raw.githubusercontent.com/uffici-ai/uffici-companion/v0.3.0/fixture/ci/wave1.json"
    ]
    assert json.loads((tmp_path / "ci" / "wave1.json").read_text()) == {"v": 1}


def test_a_fixture_that_is_not_json_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(e2e, "fetch", lambda url: b"<html>not found</html>")
    with pytest.raises(e2e.Failure, match="not JSON"):
        e2e.fetch_fixture("0.3.0", tmp_path)


def test_an_unreadable_release_says_this_run_takes_no_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(request, timeout):
        raise OSError("HTTP Error 404: Not Found")

    monkeypatch.setattr(e2e.urllib.request, "urlopen", refuse)
    with pytest.raises(e2e.Failure) as caught:
        e2e.fetch("https://example.invalid/x")
    assert "without a credential" in caught.value.detail


def test_the_live_follower_prints_redacted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeProcess:
        stdout = iter(['{"msg": "plugin ready", "token": "s3cret"}\n'])

        def kill(self) -> None:
            pass

    monkeypatch.setattr(e2e.subprocess, "Popen", lambda *a, **k: FakeProcess())
    follower = e2e.Follower("container", "s3cret")
    follower._read()
    printed = capsys.readouterr().out
    assert "s3cret" not in printed
    assert "«wire-token»" in printed
    # What it collects stays raw: it is read, never printed, and the readiness checks need it.
    assert "plugin ready" in follower.text


def test_a_run_whose_turns_errored_is_not_a_pass() -> None:
    """🔴 #2232: Hermes posts a failed turn's error as the reply, so every post
    expectation held over a run in which no turn reached the model."""
    errored = (
        "2026-09-29 10:55:14,464 ERROR gateway.run: Agent error in session agent:main:x\n"
        "ValueError: Model mock-model has a context window of 32,000 tokens"
    )
    assert len(e2e.harness_failures(errored, "[mock] request 1: stream=True")) == 1
    assert "served no completion" in " ".join(e2e.harness_failures("", ""))
    assert e2e.harness_failures("INFO gateway.run: ok", "[mock] request 1: stream=True") == []


def test_the_hermes_contract_probe_is_part_of_the_verdict() -> None:
    """The probe's own named line fails the run; a probe that died without one still does,
    by its exit code. Silence from a failing probe is never a pass."""
    assert e2e.contract_failures(0, "") == []
    named = "hermes contract: follow-up hooks FAILED — the request would never close"
    assert e2e.contract_failures(1, f"noise\n{named}\n") == [named]
    assert e2e.contract_failures(137, "") == ["the Hermes contract probe exited with code 137"]


def test_the_hermes_contract_probe_runs_inside_the_gateway_as_hermes(monkeypatch) -> None:
    """Inside the running gateway's container, as Hermes's user, with the probe on stdin and
    the release's own manifest as its argument."""
    seen: dict = {}

    def fake_run(args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return e2e.subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(e2e.subprocess, "run", fake_run)
    e2e.run_hermes_contract("hermes-container")
    assert seen["args"][:7] == [
        "docker", "exec", "-i", "-u", "hermes", "hermes-container", e2e.HERMES_PYTHON,
    ]
    assert json.loads(seen["args"][-1])["name"] == "uffici-for-hermes"
    assert "def follow_up_hooks_reach_the_adapter" in seen["kwargs"]["input"]
    assert seen["kwargs"]["timeout"] == e2e.CONTRACT_TIMEOUT_S, "a hung probe must end the run"
