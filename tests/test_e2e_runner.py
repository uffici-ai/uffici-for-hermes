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


# ------------------------------------------------------------- the tool tally (#2351)

TOOL_SCRIPT = {"steps": [{"step": "ring"}, {"step": "expectToolCall", "name": "list_boards"}]}


def test_a_script_expects_a_tool_call_only_when_it_says_so() -> None:
    assert e2e.fixture_expects_tool_call(TOOL_SCRIPT) is True
    for script in ({"steps": [{"step": "ring"}]}, {}, {"steps": None}, [], {"steps": ["x"]}):
        assert e2e.fixture_expects_tool_call(script) is False


def test_a_script_that_expects_a_tool_call_passes_on_a_counted_one() -> None:
    lines = _log({**VERDICT, "toolCalls": 1})
    assert e2e.verdict_failures(lines, 0, expects_tool_call=True) == []


@pytest.mark.parametrize("calls", [0, None, True, "1"])
def test_a_script_that_expects_a_tool_call_fails_without_one(calls) -> None:
    verdict = {**VERDICT, "toolCalls": calls} if calls is not None else VERDICT
    failures = e2e.verdict_failures(_log(verdict), 0, expects_tool_call=True)
    assert len(failures) == 1 and "no Uffici tool reached the platform" in failures[0]


def test_a_script_without_the_expectation_is_judged_as_before() -> None:
    """A superset of the old rule: a release whose script has no tool step is not asked."""
    assert e2e.verdict_failures(_log(VERDICT), 0) == []
    assert e2e.verdict_failures(_log({**VERDICT, "toolCalls": 0}), 0) == []


def test_the_run_reads_the_expectation_from_the_releases_own_script() -> None:
    source = RUNNER.read_text()
    assert "expects_tool_call=expects_tool_call" in source
    assert "fixture_expects_tool_call(fixture_script)" in source
    assert "fixture_script = json.loads(fixture_text)" in source


# ------------------------------------------------------------- the re-run tally (#2454)

RERUN_SCRIPT = {"steps": [{"step": "ring"}, {"step": "expectToolRerun"}]}


def test_a_script_expects_a_rerun_only_when_it_says_so() -> None:
    assert e2e.fixture_expects_tool_rerun(RERUN_SCRIPT) is True
    for script in (TOOL_SCRIPT, {}, {"steps": None}, [], {"steps": ["x"]}):
        assert e2e.fixture_expects_tool_rerun(script) is False


def test_a_script_that_expects_a_rerun_passes_on_a_counted_one() -> None:
    lines = _log({**VERDICT, "toolCalls": 2, "toolReruns": 1})
    assert e2e.verdict_failures(lines, 0, expects_tool_rerun=True) == []


@pytest.mark.parametrize("reruns", [0, None, True, "1"])
def test_a_script_that_expects_a_rerun_fails_without_one(reruns) -> None:
    """🔴 A tool call is not a re-run: `toolCalls` never stands in for `toolReruns`."""
    verdict = {**VERDICT, "toolCalls": 2}
    if reruns is not None:
        verdict["toolReruns"] = reruns
    failures = e2e.verdict_failures(_log(verdict), 0, expects_tool_rerun=True)
    assert len(failures) == 1 and "no re-run of a recorded call" in failures[0]


def test_the_run_reads_the_rerun_expectation_from_the_releases_own_script() -> None:
    source = RUNNER.read_text()
    assert "expects_tool_rerun=expects_tool_rerun" in source
    assert "fixture_expects_tool_rerun(fixture_script)" in source


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


def test_the_fixture_is_readable_by_the_companions_user(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 🔴 #3134: the runner's scratch directory is `0700` and the runner's own; the
    # companion image runs as 65532 and, on a Linux engine, could not open the
    # bind-mounted fixture — the pod died before Hermes joined it. The fixture is
    # published release content, so every user may read it.
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    monkeypatch.setattr(e2e, "fetch", lambda url: b'{"v": 1}')
    e2e.fetch_fixture("0.3.0", scratch)
    for directory in (scratch, scratch / "ci"):
        assert directory.stat().st_mode & 0o005 == 0o005, oct(directory.stat().st_mode)
    assert (scratch / "ci" / "wave1.json").stat().st_mode & 0o004, (
        "the fixture is not world-readable"
    )


def test_a_pod_that_will_not_start_shows_the_companions_own_words(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 🔴 #3134: the reason was only in the companion's log, which nobody printed.
    seen = {}

    def fake_docker(*args: str, env: dict[str, str]):
        seen["token"] = env["UFFICI_WIRE_TOKEN"]
        if args[0] == "up":
            return e2e.subprocess.CompletedProcess(args, 1, "", "runc create failed")
        if args[0] == "logs":
            return e2e.subprocess.CompletedProcess(
                args,
                0,
                f"uffici: opening the fixture script: permission denied {seen['token']}",
                "",
            )
        return e2e.subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(e2e, "docker", fake_docker)
    with pytest.raises(e2e.Failure) as caught:
        e2e.run("0.3.0", tmp_path)
    assert caught.value.reason == "the pod would not start"
    assert "runc create failed" in caught.value.detail
    assert "opening the fixture script: permission denied" in caught.value.detail
    assert seen["token"] not in caught.value.detail


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
        "docker",
        "exec",
        "-i",
        "-u",
        "hermes",
        "hermes-container",
        e2e.HERMES_PYTHON,
    ]
    assert json.loads(seen["args"][-1])["name"] == "uffici-for-hermes"
    assert "def follow_up_hooks_reach_the_adapter" in seen["kwargs"]["input"]
    assert seen["kwargs"]["timeout"] == e2e.CONTRACT_TIMEOUT_S, "a hung probe must end the run"


def test_each_restart_the_companion_asks_for_is_made_once() -> None:
    """#2235: a `restartHarness` step waits on a restart only this runner can make."""
    asked = {"msg": e2e.RESTART_HARNESS}
    other = {"msg": e2e.PLUGIN_READY}
    assert e2e.restarts_owed(_log(other), 0) == 0
    assert e2e.restarts_owed(_log(asked, other), 0) == 1
    assert e2e.restarts_owed(_log(asked, other), 1) == 0
    assert e2e.restarts_owed(_log(asked, other, asked), 1) == 1


def test_each_session_end_the_companion_asks_for_is_made_once_for_its_conversation() -> None:
    """#2414: an `endHarnessSession` step names a conversation only this runner can reach
    inside Hermes; each is ended once, in the order asked."""
    first = {"msg": e2e.END_SESSION_LINE, "conversation": "conv_ci_invoices"}
    second = {"msg": e2e.END_SESSION_LINE, "conversation": "conv_ci_ledger"}
    other = {"msg": e2e.RESTART_HARNESS}
    assert e2e.session_ends_owed(_log(other), 0) == []
    assert e2e.session_ends_owed(_log(first, other), 0) == ["conv_ci_invoices"]
    assert e2e.session_ends_owed(_log(first, other), 1) == []
    assert e2e.session_ends_owed(_log(first, other, second), 1) == ["conv_ci_ledger"]


def test_a_session_is_ended_inside_the_gateway_as_hermes(monkeypatch) -> None:
    """Inside the running gateway's container, as Hermes's user, with the helper on stdin
    and the conversation as its argument."""
    seen: dict = {}

    def fake_run(args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return e2e.subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(e2e.subprocess, "run", fake_run)
    e2e.end_harness_session("hermes-container", "conv_ci_invoices")
    assert seen["args"] == [
        "docker",
        "exec",
        "-i",
        "-u",
        "hermes",
        "hermes-container",
        e2e.HERMES_PYTHON,
        "-",
        "conv_ci_invoices",
    ]
    assert "db.end_session(session_id, END_REASON)" in seen["kwargs"]["input"]
    assert seen["kwargs"]["timeout"] == e2e.CONTRACT_TIMEOUT_S


SHIPPED = {
    "msg": e2e.LOGS_SHIPPED,
    "companion": 12,
    "plugin": 4,
    "harness": 3,
    "redactions": 6,
    "tokenLeaks": 0,
    "canaries": 3,
    "restarts": 1,
    "restartsWithLastWords": 1,
}


def test_a_run_that_shipped_all_three_sources_cleanly_passes() -> None:
    assert e2e.shipped_log_failures(_log(VERDICT, SHIPPED), 1) == []


@pytest.mark.parametrize("source", ["companion", "plugin", "harness"])
def test_a_run_missing_a_source_is_not_a_pass(source: str) -> None:
    assert e2e.shipped_log_failures(_log({**SHIPPED, source: 0}), 1) == [
        f"no {source} line was shipped"
    ]


def test_a_shipped_line_carrying_the_wire_token_fails_the_run() -> None:
    assert e2e.shipped_log_failures(_log({**SHIPPED, "tokenLeaks": 2}), 1) == [
        "2 shipped line(s) carry the wire token"
    ]


def test_a_companion_that_never_reported_what_it_shipped_fails_the_run() -> None:
    # An older companion, or a reworded line: a pass must not be read off an absence.
    assert e2e.shipped_log_failures(_log(VERDICT), 1) == [
        "the companion reported what it shipped 0 time(s), not once"
    ]
    assert e2e.shipped_log_failures(_log({**SHIPPED, "tokenLeaks": None}), 1) == [
        "an unknown number of shipped line(s) carry the wire token"
    ]


def test_a_run_whose_canaries_were_not_all_shipped_redacted_is_not_a_pass() -> None:
    assert e2e.shipped_log_failures(_log({**SHIPPED, "canaries": 2}), 1) == [
        "2 of the run's 3 canaries shipped redacted"
    ]
    assert e2e.shipped_log_failures(_log({**SHIPPED, "canaries": None}), 1) == [
        "none of the run's 3 canaries shipped redacted"
    ]
    # 🔴 A total of redactions from ordinary traffic never stands in for a canary.
    assert e2e.shipped_log_failures(_log({**SHIPPED, "redactions": 40, "canaries": 0}), 1) == [
        "0 of the run's 3 canaries shipped redacted"
    ]


def test_every_restart_must_have_shipped_the_harnesss_own_warning() -> None:
    assert e2e.shipped_log_failures(_log({**SHIPPED, "restartsWithLastWords": 0}), 1) == [
        "0 of the run's 1 restart(s) shipped the harness's own warning before it stopped"
    ]
    two = {**SHIPPED, "restarts": 2, "restartsWithLastWords": 1}
    assert e2e.shipped_log_failures(_log(two), 2) == [
        "1 of the run's 2 restart(s) shipped the harness's own warning before it stopped"
    ]
    # 🔴 An absent field is never a pass: an older companion cannot vouch for a restart.
    assert e2e.shipped_log_failures(_log({**SHIPPED, "restartsWithLastWords": None}), 1) == [
        "none of the run's 1 restart(s) shipped the harness's own warning before it stopped"
    ]


def test_the_companions_restart_count_must_match_the_restarts_made() -> None:
    none = {**SHIPPED, "restarts": 0, "restartsWithLastWords": 0}
    assert e2e.shipped_log_failures(_log(none), 1) == [
        "the companion counted 0 restart(s), the run made 1"
    ]
    assert e2e.shipped_log_failures(_log({**SHIPPED, "restarts": None}), 1) == [
        "the companion counted no restart(s), the run made 1"
    ]
    assert e2e.shipped_log_failures(_log({**SHIPPED, "restarts": True}), 1) == [
        "the companion counted True restart(s), the run made 1"
    ]
    # A run with no restart step has no last words to owe.
    assert e2e.shipped_log_failures(_log(none), 0) == []


@pytest.mark.parametrize("made", [None, True, -1, "1"])
def test_a_caller_that_forgets_the_restarts_is_refused(made: object) -> None:
    with pytest.raises(TypeError):
        e2e.shipped_log_failures(_log(SHIPPED), made)  # type: ignore[arg-type]


def _mock(fields: str) -> str:
    return f"[mock] request 1: stream=True marker=False {fields} reply_chars=10"


def test_the_context_passes_on_the_system_side() -> None:
    log = "\n".join(
        [
            _mock("system_canary=True user_canary=False system_missing=False"),
            _mock("system_canary=False user_canary=False system_missing=True"),
        ]
    )
    assert e2e.context_failures(log, True) == []


def test_the_context_fails_off_the_system_side_or_on_the_user_side() -> None:
    never = _mock("system_canary=False user_canary=False system_missing=True")
    assert "never reached" in " ".join(e2e.context_failures(never, True))
    user = _mock("system_canary=True user_canary=True system_missing=True")
    assert "USER side" in " ".join(e2e.context_failures(user, True))
    unsaid = _mock("system_canary=True user_canary=False system_missing=False")
    assert "missing never told" in " ".join(e2e.context_failures(unsaid, True))


def test_a_release_fixture_without_the_canary_has_nothing_to_prove() -> None:
    assert e2e.context_failures("", False) == []


def test_the_summary_passes_on_the_user_side() -> None:
    log = "\n".join(
        [
            _mock("summary_canary=False summary_system=False"),
            _mock("summary_canary=True summary_system=False"),
        ]
    )
    assert e2e.summary_failures(log, True) == []


def test_the_summary_fails_off_the_user_side_or_on_the_system_side() -> None:
    never = _mock("summary_canary=False summary_system=False")
    assert "never reached" in " ".join(e2e.summary_failures(never, True))
    system = _mock("summary_canary=True summary_system=True")
    assert "SYSTEM side" in " ".join(e2e.summary_failures(system, True))


def test_a_release_fixture_without_a_summary_has_nothing_to_prove() -> None:
    assert e2e.summary_failures("", False) == []


MEMORY_FIXTURE = '{"memory": {"unavailableIn": ["onboarding"]}, "text": "[remember-this]"}'


def test_memory_passes_when_recalled_on_the_user_side_and_the_outage_was_told() -> None:
    log = "\n".join(
        [
            _mock("memory_recalled=True memory_unavailable=False memory_system=False"),
            _mock("memory_recalled=False memory_unavailable=True memory_system=False"),
        ]
    )
    assert e2e.memory_failures(log, MEMORY_FIXTURE, "") == []


def test_memory_fails_never_recalled_on_the_system_side_or_the_outage_unsaid() -> None:
    never = _mock("memory_recalled=False memory_unavailable=True memory_system=False")
    assert "never came back" in " ".join(e2e.memory_failures(never, MEMORY_FIXTURE, ""))
    system = _mock("memory_recalled=True memory_unavailable=True memory_system=True")
    assert "SYSTEM message" in " ".join(e2e.memory_failures(system, MEMORY_FIXTURE, ""))
    unsaid = _mock("memory_recalled=True memory_unavailable=False memory_system=False")
    assert "memory down" in " ".join(e2e.memory_failures(unsaid, MEMORY_FIXTURE, ""))


def test_hermes_keeping_its_own_memory_fails_the_run() -> None:
    log = _mock("memory_recalled=True memory_unavailable=True memory_system=False")
    assert e2e.memory_failures(log, MEMORY_FIXTURE, "something.lock\n") == []
    kept = " ".join(e2e.memory_failures(log, MEMORY_FIXTURE, "MEMORY.md\nUSER.md\n"))
    assert "MEMORY.md, USER.md" in kept


def test_a_release_fixture_without_memory_has_nothing_to_prove() -> None:
    assert e2e.memory_failures("", "{}", "MEMORY.md") == []


FILES_FIXTURE = '{"data": "[pdf-canary]", "steps": [{"step": "interruptNextUpload"}]}'
FILES_MOCK = _mock(
    "pdf_media=True image_media=True media_written=True upload_call=True upload_result=False"
)
FILES_VERDICT = {
    **VERDICT,
    "downloads": 2,
    "uploads": 2,
    "uploadsResumed": 1,
    "attachmentsPosted": 1,
}


def test_the_files_pass_when_both_use_cases_ran_and_the_upload_resumed() -> None:
    assert e2e.file_failures(FILES_MOCK, FILES_FIXTURE, _log(FILES_VERDICT)) == []


@pytest.mark.parametrize(
    ("field", "value"),
    [("downloads", 1), ("uploads", 1), ("uploadsResumed", 0), ("attachmentsPosted", None)],
)
def test_the_files_fail_on_a_count_under_its_floor(field: str, value: object) -> None:
    verdict = {**FILES_VERDICT, field: value}
    if value is None:
        del verdict[field]
    failed = " ".join(e2e.file_failures(FILES_MOCK, FILES_FIXTURE, _log(verdict)))
    assert f"{field}=" in failed


def test_a_resume_is_owed_only_when_the_fixture_interrupts_an_upload() -> None:
    verdict = {**FILES_VERDICT, "uploadsResumed": 0}
    assert e2e.file_failures(FILES_MOCK, '{"data": "[pdf-canary]"}', _log(verdict)) == []


@pytest.mark.parametrize(
    ("flag", "reason"),
    [
        ("pdf_media", "PDF"),
        ("image_media", "image"),
        ("media_written", "made a file"),
        ("upload_call", "save a file"),
    ],
)
def test_the_files_fail_when_the_model_did_not_see_or_make_them(flag: str, reason: str) -> None:
    log = FILES_MOCK.replace(f"{flag}=True", f"{flag}=False")
    assert reason in " ".join(e2e.file_failures(log, FILES_FIXTURE, _log(FILES_VERDICT)))


def test_a_release_fixture_without_files_has_nothing_to_prove() -> None:
    assert e2e.file_failures("", "{}", []) == []
    assert "uploads=none" in " ".join(e2e.file_failures(FILES_MOCK, FILES_FIXTURE, []))
