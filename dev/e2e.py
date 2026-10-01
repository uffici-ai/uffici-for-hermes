"""The first conversation against a RELEASED companion: the gate on a release.

A real Hermes, running this plugin against the companion customers install (its
published image, by version), answers the scripted conversation that same
release ships as its fixture. The companion plays the agents host, and its exit
code is the verdict.

Every failure is LOUD and named. The run exists to catch the case that reports
success while proving nothing: a plugin that never loaded, a harness that
never attached, or a script whose expectations all held because no ring was
delivered. Each exits non-zero and says which it was. None of them is a skip.

Usage: uv run --frozen python dev/e2e.py [--companion-version X.Y.Z|latest]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

DEV = Path(__file__).resolve().parent
COMPOSE = ["docker", "compose", "-f", str(DEV / "compose.e2e.yaml")]
COMPANION_REPO = "uffici-ai/uffici-companion"
FIXTURE = "fixture/ci/wave1.json"
# What the plugin assumes about Hermes, checked inside the gateway's own container.
CONTRACT_PROBE = DEV / "hermes_contract.py"
MANIFEST = DEV.parent / "uffici-plugin.json"
HERMES_PYTHON = "/opt/hermes/.venv/bin/python"
# A probe that hangs must end the run with its name, not the job's timeout.
CONTRACT_TIMEOUT_S = 120
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")

# How long Hermes gets to load the plugin and become able to answer.
READY_TIMEOUT_S = 240
# How long the scripted conversation gets once it starts: a bound to end a stuck
# run, not a pace. Against the model mock a run takes seconds.
SCRIPT_TIMEOUT_S = 1500

# The companion's log vocabulary this runner reads. Each is a message the
# companion logs verbatim; a reworded line would otherwise retire its check
# silently, so the development repository pins every one against the
# companion's source.
PLUGIN_CONNECTED = "plugin connected"
PLUGIN_READY = "plugin ready"
SCRIPT_COMPLETE = "script complete"
CONNECTION_CLOSED = "closing the plugin connection"
# The companion's count of what an --once run shipped, by source (#2267).
LOGS_SHIPPED = "logs shipped"
# Every run ships the companion's lines, the plugin's and Hermes's.
SHIPPED_SOURCES = ("companion", "plugin", "harness")
# The companion's --once run hands each capture path a canary carrying the wire
# token; its report counts the SOURCES whose own canary shipped redacted, never a
# total of redactions that ordinary traffic could meet (#2259 review).
CANARIES = len(SHIPPED_SOURCES)
# The one close that is not a fault: the companion shutting down at the end.
SHUTTING_DOWN = "shutting_down"
# What the companion logs when a `restartHarness` step wants Hermes restarted
# mid-turn (#2235). Only this runner can: the companion shares the pod's network,
# not its processes. Each line is one restart of the Hermes container.
RESTART_HARNESS = "restart the harness"
# How long Hermes gets to shut down on a restart before Docker kills it.
RESTART_GRACE_S = 30


class Failure(Exception):
    """A named reason the run failed, with the detail a reader needs."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def log_lines(text: str) -> list[dict]:
    """The companion's structured log lines; anything that is not JSON is dropped."""
    out = []
    for line in text.splitlines():
        try:
            decoded = json.loads(line)
        except ValueError:
            continue
        if isinstance(decoded, dict):
            out.append(decoded)
    return out


def has(lines: list[dict], message: str) -> bool:
    return any(line.get("msg") == message for line in lines)


#: Hermes's own line for a turn that failed (#2232). It posts the error text AS THE
#: REPLY, so a `nonEmpty` expectation holds on a turn that never reached the model.
HERMES_TURN_ERROR = "Agent error in session"

#: The model mock's line per completion it served (`model_mock/server.py`).
MOCK_COMPLETION = "[mock] request "


def restarts_owed(lines: list[dict], done: int) -> int:
    """How many harness restarts the companion has asked for that this run has not made."""
    return max(0, sum(1 for line in lines if line.get("msg") == RESTART_HARNESS) - done)


def harness_failures(hermes_log: str, mock_log: str) -> list[str]:
    """Why a run is not a pass, read from the harness's side; empty for a pass.

    A positive and a negative: the negative alone would retire silently if Hermes
    reworded its line; the positive — the mock served at least one completion — holds.
    """
    failures = []
    errored = [line for line in hermes_log.splitlines() if HERMES_TURN_ERROR in line]
    if errored:
        failures.append(
            f"{len(errored)} Hermes turn(s) failed — their error text was posted as the reply"
        )
    if MOCK_COMPLETION not in mock_log:
        failures.append("the model mock served no completion: no turn ever reached a model")
    return failures


def verdict_failures(lines: list[dict], exit_code: int) -> list[str]:
    """Every reason the finished run is not a pass; empty only for a real pass.

    Pure, so the judgement is tested on its own: a green run that asserted
    nothing is the failure this gate exists for.
    """
    verdict = next((line for line in lines if line.get("msg") == SCRIPT_COMPLETE), None)
    if verdict is None:
        return ["the companion never reported a verdict"]
    failures = [
        f"the script reported {count}=0 — every expectation held over nothing at all"
        for count in ("ringsDelivered", "briefs")
        if not verdict.get(count)
    ]
    # A run can hold every expectation and still have dropped the wire: the plugin
    # reconnects, the requests redeliver, and the verdict reads the same.
    dropped = [
        line
        for line in lines
        if line.get("msg") == CONNECTION_CLOSED and line.get("reason") != SHUTTING_DOWN
    ]
    if dropped:
        reasons = ", ".join(str(line.get("reason")) for line in dropped)
        failures.append(
            f"the companion dropped the plugin's connection {len(dropped)} time(s): {reasons}"
        )
    if exit_code != 0:
        unmet = verdict.get("unmet", "some")
        failures.append(f"the scripted conversation had {unmet} unmet expectation(s)")
    return failures


def shipped_log_failures(lines: list[dict]) -> list[str]:
    """Whether the run shipped the machine's logs (#2267); empty for a pass.

    All three sources must reach the scripted host, and no shipped line may still
    carry the run's wire token — the one secret a fixture run holds, presented by
    the plugin in every hello, so a line carrying it is a redaction that failed.
    """
    reports = [line for line in lines if line.get("msg") == LOGS_SHIPPED]
    if len(reports) != 1:
        return [f"the companion reported what it shipped {len(reports)} time(s), not once"]
    report = reports[0]
    failures = [
        f"no {source} line was shipped"
        for source in SHIPPED_SOURCES
        if not isinstance(report.get(source), (int, float)) or report[source] < 1
    ]
    canaries = report.get("canaries")
    if not isinstance(canaries, (int, float)) or canaries != CANARIES:
        shown = canaries if canaries is not None else "none"
        failures.append(f"{shown} of the run's {CANARIES} canaries shipped redacted")
    leaks = report.get("tokenLeaks")
    if leaks != 0:
        count = leaks if isinstance(leaks, (int, float)) else "an unknown number of"
        failures.append(f"{count} shipped line(s) carry the wire token")
    return failures


def contract_failures(returncode: int, stderr: str) -> list[str]:
    """The Hermes contract probe's verdict: its own named line, or its exit code.

    The plugin's tests run against a stand-in transcribed from Hermes's source, which is an
    assertion about Hermes that nothing checks. The probe checks the same assumptions against
    the real Hermes this run is using, and each of them once failed on a real machine while
    every test passed.
    """
    if returncode == 0:
        return []
    named = [line for line in stderr.splitlines() if line.startswith("hermes contract:")]
    return [named[-1] if named else f"the Hermes contract probe exited with code {returncode}"]


def run_hermes_contract(container: str) -> subprocess.CompletedProcess[str]:
    """Run the probe inside the running gateway's container, as Hermes's own user, so Hermes
    discovers the plugin exactly as the gateway did."""
    return subprocess.run(
        [
            "docker", "exec", "-i", "-u", "hermes", container, HERMES_PYTHON, "-",
            MANIFEST.read_text(encoding="utf-8"),
        ],
        input=CONTRACT_PROBE.read_text(encoding="utf-8"),
        text=True,
        capture_output=True,
        check=False,
        timeout=CONTRACT_TIMEOUT_S,
    )


def redact(text: str, token: str) -> str:
    """The wire token is never printed, whoever echoed it."""
    return text.replace(token, "«wire-token»") if token else text


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "uffici-for-hermes-e2e"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    except OSError as error:
        raise Failure(
            f"could not fetch {url}",
            "The companion's release must be readable without a credential; this run takes"
            f" none by design. ({error})",
        ) from error


def resolve_version(requested: str) -> str:
    """`latest` resolved through the release listing; anything else must be X.Y.Z."""
    version = requested.removeprefix("v")
    if version == "latest":
        listing = fetch(f"https://api.github.com/repos/{COMPANION_REPO}/releases/latest")
        version = str(json.loads(listing).get("tag_name", "")).removeprefix("v")
    if not SEMVER.match(version):
        raise Failure(f"{requested!r} is not a companion release version (X.Y.Z or latest)")
    return version


def fetch_fixture(version: str, into: Path) -> None:
    """The scripted conversation from the SAME release as the image."""
    body = fetch(f"https://raw.githubusercontent.com/{COMPANION_REPO}/v{version}/{FIXTURE}")
    try:
        json.loads(body)
    except ValueError as error:
        raise Failure(f"the release v{version} ships a fixture that is not JSON") from error
    target = into / "ci" / "wave1.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(body)


def docker(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([*COMPOSE, *args], env=env, text=True, capture_output=True)


class Follower:
    """`docker logs -f` on one container, collected in the background.

    Printed live, and redacted before it is: the companion never logs the wire
    token, but this runner holds it, so a line that did is caught here too.
    """

    def __init__(self, container: str, token: str) -> None:
        self.text = ""
        self._token = token
        self._process = subprocess.Popen(
            ["docker", "logs", "-f", container],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            self.text += line
            print(redact(line, self._token), end="", flush=True)

    def stop(self) -> None:
        self._process.kill()


def container_logs(container: str) -> str:
    return subprocess.run(
        ["docker", "logs", container], text=True, capture_output=True
    ).stdout


def exit_code_of(container: str) -> int | None:
    """The container's exit code, or None while it is still running."""
    state = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}} {{.State.ExitCode}}", container],
        text=True,
        capture_output=True,
    ).stdout.split()
    if not state or state[0] == "true":
        return None
    return int(state[1]) if len(state) > 1 else 1


def run(version: str, fixture_dir: Path) -> None:
    token = secrets.token_urlsafe(32)
    env = {
        **os.environ,
        "UFFICI_COMPANION_VERSION": version,
        "UFFICI_E2E_FIXTURE_DIR": str(fixture_dir),
        "UFFICI_WIRE_TOKEN": token,
    }
    # This compose project holds nothing but this run, so `down -v` removes
    # exactly what it started — before the run too, so a stale session store
    # cannot turn a fresh conversation into a resumed one — and after it on EVERY
    # path: a pod that half-started, a failure, or a cancelled job (SIGTERM is
    # turned into an exit, so `finally` runs).
    docker("down", "-v", "--remove-orphans", env=env)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    follower: Follower | None = None
    try:
        up = docker("up", "-d", "--build", "model-mock", "companion", "hermes", env=env)
        if up.returncode != 0:
            raise Failure("the pod would not start", redact(up.stdout + up.stderr, token))

        def container(service: str) -> str:
            return docker("ps", "-aq", service, env=env).stdout.strip().split("\n")[0]

        companion = container("companion")
        hermes = container("hermes")
        mock = container("model-mock")
        if not companion or not hermes:
            raise Failure("the pod's containers did not come up")
        follower = Follower(companion, token)

        def the_other_side() -> str:
            hermes_log = redact(container_logs(hermes), token)
            warnings = [line for line in hermes_log.splitlines() if re.search(r"WARN|ERROR", line)]
            return "\n".join(
                [
                    "── hermes (warnings and errors) ─────────────────────────",
                    *warnings[-40:],
                    "── hermes (last lines) ──────────────────────────────────",
                    *hermes_log.splitlines()[-20:],
                    "── model mock ────────────────────────────────────────────",
                    *redact(container_logs(mock), token).splitlines()[-20:],
                ]
            )

        # The plugin must LOAD: Hermes starts perfectly well without it.
        deadline = time.monotonic() + READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if re.search(r"Failed to load plugin", container_logs(hermes), re.IGNORECASE):
                raise Failure("the plugin did not load in Hermes", the_other_side())
            if has(log_lines(follower.text), PLUGIN_READY) or exit_code_of(companion) is not None:
                break
            time.sleep(2)
        exited = exit_code_of(companion)
        # Once the companion has exited its whole log is readable at once; the live
        # follower can still be behind it, and a fast run must not read as never ready.
        lines = log_lines(container_logs(companion) if exited is not None else follower.text)
        if not has(lines, PLUGIN_READY):
            reason = (
                f"the companion exited with code {exited} before the harness attached"
                if exited is not None
                else "the plugin connected but its harness never became able to answer"
                if has(lines, PLUGIN_CONNECTED)
                else "Hermes never attached to the companion"
            )
            raise Failure(reason, the_other_side())
        print("\n✓ the harness attached and is ready", flush=True)

        try:
            probe = run_hermes_contract(hermes)
        except subprocess.TimeoutExpired as exc:
            raise Failure(
                f"the Hermes contract probe did not finish in {CONTRACT_TIMEOUT_S}s"
            ) from exc
        print(redact(probe.stdout, token), end="", flush=True)
        probe_stderr = redact(probe.stderr, token)
        failures = contract_failures(probe.returncode, probe_stderr)
        if failures:
            raise Failure("; ".join(failures), probe_stderr)

        deadline = time.monotonic() + SCRIPT_TIMEOUT_S
        restarts = 0
        exit_code = exit_code_of(companion)
        while exit_code is None and time.monotonic() < deadline:
            time.sleep(1)
            for _ in range(restarts_owed(log_lines(follower.text), restarts)):
                restarts += 1
                print(f"\n── restarting Hermes mid-turn ({restarts})", flush=True)
                restart = subprocess.run(
                    ["docker", "restart", "-t", str(RESTART_GRACE_S), hermes],
                    text=True,
                    capture_output=True,
                )
                if restart.returncode != 0:
                    raise Failure("the Hermes container would not restart", the_other_side())
            exit_code = exit_code_of(companion)
        if exit_code is None:
            raise Failure("the scripted conversation did not finish in time", the_other_side())
        # The companion's log is complete once it has exited.
        lines = log_lines(container_logs(companion))
        failures = (
            verdict_failures(lines, exit_code)
            + shipped_log_failures(lines)
            + harness_failures(container_logs(hermes), container_logs(mock))
        )
        if failures:
            raise Failure("; ".join(failures), the_other_side())
        verdict = next(line for line in lines if line.get("msg") == SCRIPT_COMPLETE)
        print(f"\n── verdict: {json.dumps(verdict)}")
        print(f"✓ the first conversation passed against companion {version}")
    finally:
        if follower is not None:
            follower.stop()
        docker("down", "-v", "--remove-orphans", env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--companion-version", default="latest")
    args = parser.parse_args(argv)
    try:
        version = resolve_version(args.companion_version)
        print(f"── companion {version}: the released image and its fixture", flush=True)
        with tempfile.TemporaryDirectory(prefix="uffici-e2e-") as scratch:
            fixture_dir = Path(scratch)
            fetch_fixture(version, fixture_dir)
            run(version, fixture_dir)
    except Failure as failure:
        print(f"\n✗ e2e: {failure.reason}", file=sys.stderr)
        if failure.detail:
            print(failure.detail, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
