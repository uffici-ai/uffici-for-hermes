"""Files in and out, in Hermes's conventions (#2674): media in, ``MEDIA:`` out, two tools.

The companion moves every byte; these tests hold what the PLUGIN decides — which files
reach Hermes as media and as what, which of Hermes's hooks turn a file into an attachment
on the reply, what the person reads when one does not make it, and when the two tools
exist at all.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from conftest import MessageType, ProcessingOutcome
from test_adapter import FakeStore, build, golden, settle
from test_platform_tools import Loop, Registry, in_turn
from uffici_for_hermes import adapter as adapter_module
from uffici_for_hermes import files as fm
from uffici_for_hermes import platform_tools as pt
from uffici_for_hermes.calls import CallResult
from uffici_for_hermes.files import (
    DOWNLOAD_TOOL,
    FEATURE_FILES,
    PREFETCH_MAX_BYTES,
    UPLOAD_TOOL,
    FileTools,
    attachment_suffix,
    display_name,
    inbound_media,
    media_type_for,
    not_attached,
    workspace_path,
    workspace_relative,
)
from uffici_for_hermes.wire import WIRE_V1
from uffici_for_hermes.wire_client import plugin_hello

CONVERSATION = "a1b2c3d4-e5f6-4789-8abc-def012345678"
REQUEST = "6f1c0f7a-2f0e-4f2e-9b1e-0d6a0b5c7e21"


def op_golden(name: str) -> dict:
    return json.loads((WIRE_V1 / "golden" / "ops" / "valid" / f"{name}.json").read_text())


ATTACHED = op_golden("files.upload.result-attached")
SAVED = op_golden("files.upload.result-saved")
DOWNLOADED = op_golden("files.download.result-attachment")


def run(coro):
    return asyncio.run(coro)


class Tunnel:
    """The call client's ``call``: records each call (and whether its local file existed
    when it was made), answers from a table."""

    def __init__(self, workspace: Path | None = None, answers: dict | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.present: list[bool] = []
        self.workspace = workspace
        self.answers = answers or {}

    async def call(self, op: str, payload: dict | None = None, *, deadline_ms: int) -> CallResult:
        payload = payload or {}
        self.calls.append((op, payload))
        assert deadline_ms == fm.TRANSFER_DEADLINE_MS
        if self.workspace is not None and "localPath" in payload:
            self.present.append((self.workspace / payload["localPath"]).is_file())
        answer = self.answers.get(op)
        if isinstance(answer, CallResult):
            return answer
        if op == fm.UPLOAD_OP:
            return CallResult(call_id="c", payload=SAVED if "to" in payload else ATTACHED)
        return CallResult(call_id="c", payload=DOWNLOADED)


def refused(error_class: str, upstream: str | None = None) -> CallResult:
    return CallResult(
        call_id="c",
        error_class=error_class,
        instruction="Tell the person it could not be done.",
        upstream={"class": upstream} if upstream else None,
    )


def with_files(workspace: Path, tunnel: Tunnel | None = None, **kwargs: Any):
    """An adapter whose companion carries ``files``, with ``workspace``, and a request open."""
    adapter, wire = build(**kwargs)
    adapter._companion_features = frozenset({FEATURE_FILES, "tools"})
    adapter._workspace = workspace
    tunnel = tunnel or Tunnel(workspace)
    adapter._files._call = tunnel.call
    adapter._open_request[CONVERSATION] = REQUEST
    adapter._conversation_of[REQUEST] = CONVERSATION
    adapter._posts[REQUEST] = 0
    return adapter, wire, tunnel


def made_file(workspace: Path, name: str = "report.pdf") -> Path:
    """A file the agent saved in the workspace, where a reply's files must be."""
    path = workspace / "out" / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"%PDF-1.7 not really")
    return path


def outside_file(tmp_path: Path, name: str = "report.pdf") -> Path:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(exist_ok=True)
    path = elsewhere / name
    path.write_bytes(b"%PDF-1.7 not really")
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


# ------------------------------------------------------------------ inbound: media


def test_a_prefetched_attachment_reaches_hermes_as_media(workspace: Path) -> None:
    """🔴 The deliverable's first half: the person's PDF is a LOCAL file on the event, typed,
    the way Hermes's other platforms hand it media — and still a reference line naming it."""

    async def scenario():
        adapter, _, _ = with_files(workspace, store=FakeStore(), opener=golden("opener"))
        adapter._open_request.clear()
        adapter._conversation_of.clear()
        await adapter._on_frame(golden("brief-with-prefetched-attachments"))
        assert len(adapter.handled) == 1
        return adapter.handled[0]

    event = run(scenario())
    local = workspace / "conversations" / CONVERSATION / "1042.0-invoice-2026-09.pdf"
    assert event.media_urls == [str(local)]
    assert event.media_types == ["application/pdf"]
    assert event.media_text_inlined == [False]
    assert event.message_type is MessageType.DOCUMENT
    assert f"invoice-2026-09.pdf — conversations/{CONVERSATION}/invoice-2026-09.pdf" in event.text
    assert f"on this machine at {local}" in event.text
    # The one over the budget is a reference line saying why — and no media.
    assert "meter-scan.tiff" in event.text
    assert "not downloaded (too large to fetch before the turn)" in event.text


def test_the_reference_line_names_uffici_download_only_while_it_is_offered(
    workspace: Path,
) -> None:
    attachment = {
        "id": "1042.1",
        "name": "x.tiff",
        "ref": "r",
        "prefetchError": {"class": "over_budget"},
    }
    assert DOWNLOAD_TOOL not in attachment_suffix(attachment, workspace, fetchable=False)
    offered = attachment_suffix(attachment, workspace, fetchable=True)
    assert f"{DOWNLOAD_TOOL} with attachment_id '1042.1' fetches it" in offered
    # A refusal a download would meet again names no tool.
    denied = {**attachment, "prefetchError": {"class": "transfer_denied"}}
    assert attachment_suffix(denied, workspace, fetchable=True) == (
        " — not downloaded (Uffici did not allow the download)"
    )
    # A class from a newer companion is shown as it came.
    newer = {**attachment, "prefetchError": {"class": "quota_reached"}}
    assert "(quota_reached)" in attachment_suffix(newer, workspace, fetchable=True)
    # An opener's attachment was never prefetched: only the hint.
    plain = {"id": "9", "name": "a.pdf", "ref": "r"}
    assert attachment_suffix(plain, workspace, fetchable=True) == (
        f" — {DOWNLOAD_TOOL} with attachment_id '9' fetches it"
    )
    assert attachment_suffix({"name": "a.pdf"}, workspace, fetchable=True) == ""


def test_an_image_is_a_photo_and_an_image_hermes_cannot_show_is_a_document(
    workspace: Path,
) -> None:
    delta = [
        {"attachments": [{"name": "meter.PNG", "localPath": "conversations/c/1-meter.PNG"}]},
        "not an entry",
        {"attachments": ["not an attachment", {"name": "scan.tiff", "localPath": "c/2.tiff"}]},
        {"attachments": [{"name": "away.pdf", "localPath": "../escape.pdf"}]},
        {"attachments": [{"name": "abs.pdf", "localPath": "/etc/passwd"}]},
    ]
    urls, types = inbound_media(delta, workspace)
    assert urls == [str(workspace / "conversations/c/1-meter.PNG"), str(workspace / "c/2.tiff")]
    assert types == ["image/png", fm.OPAQUE_TYPE]
    assert adapter_module._message_type_for(["image/png"]) is MessageType.PHOTO
    assert adapter_module._message_type_for(["image/png", "application/pdf"]) is (
        MessageType.DOCUMENT
    )
    assert adapter_module._message_type_for([]) is MessageType.TEXT
    # Without a workspace there is nothing local to hand over.
    assert inbound_media(delta, None) == ([], [])


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("photo.jpg", "image/jpeg"),
        ("photo.JPEG", "image/jpeg"),
        ("chart.webp", "image/webp"),
        ("report.pdf", "application/pdf"),
        ("notes.txt", "text/plain"),
        ("scan.heic", fm.OPAQUE_TYPE),
        ("mystery", fm.OPAQUE_TYPE),
        ("", fm.OPAQUE_TYPE),
    ],
)
def test_media_types_follow_hermess_image_table(name: str, expected: str) -> None:
    assert media_type_for(name) == expected


def test_a_workspace_path_never_leaves_the_workspace(workspace: Path) -> None:
    assert workspace_path(workspace, "a/b.pdf") == workspace / "a" / "b.pdf"
    for bad in ("../x", "/abs", "", None, 3):
        assert workspace_path(workspace, bad) is None
    assert workspace_path(None, "a/b.pdf") is None


def test_the_hello_names_the_workspace_only_beside_files() -> None:
    hello = golden("hello-companion-with-files")
    features = frozenset(hello["features"])
    assert adapter_module._workspace_of(hello, features) == Path(hello["workspace"])
    assert adapter_module._workspace_of(hello, frozenset({"tools"})) is None
    assert adapter_module._workspace_of({**hello, "workspace": "relative"}, features) is None
    assert adapter_module._workspace_of({**hello, "workspace": 7}, features) is None
    assert adapter_module._workspace_of(None, features) is None


def test_the_hello_asks_for_prefetch_with_hermess_budget() -> None:
    """🔴 Requirement 5: the plugin asks; the companion knows nothing about Hermes."""
    assert PREFETCH_MAX_BYTES == 25 * 1024 * 1024
    assert adapter_module.CAPABILITIES == {"prefetchAttachments": {"maxBytes": PREFETCH_MAX_BYTES}}
    hello = plugin_hello("t", "0.21.5", "1.0.0", 1000, capabilities=adapter_module.CAPABILITIES)
    shape = golden("hello-plugin-with-capabilities")["capabilities"]
    assert set(hello["capabilities"]) == set(shape)
    assert hello["capabilities"]["prefetchAttachments"]["maxBytes"] == PREFETCH_MAX_BYTES
    assert "capabilities" not in plugin_hello("t", "0.21.5", "1.0.0", 1000)


def test_the_hello_of_a_companion_with_files_sets_the_workspace_and_offers_the_tools(
    workspace: Path,
) -> None:
    async def scenario():
        registry = Registry()
        adapter = adapter_module.UfficiAdapter(
            adapter_module.PlatformConfig(enabled=True),
            get_env=lambda _n: None,
            plugin_ctx=type("Ctx", (), {"register_tool": staticmethod(registry.register_tool)}),
        )
        adapter._files.on_catalog(frozenset({"read_file", "write_file"}))
        hello = {**golden("hello-companion-with-files"), "workspace": str(workspace)}
        await adapter._on_companion_hello(hello)
        adapter._ready_runner.cancel()
        assert adapter._workspace == workspace
        assert set(registry.tools) >= {DOWNLOAD_TOOL, UPLOAD_TOOL}
        await adapter.disconnect()
        assert adapter._workspace is None

    run(scenario())


# ------------------------------------------------------------------ outbound: MEDIA:


def test_send_document_uploads_and_the_post_carries_the_attachment(
    workspace: Path, tmp_path: Path
) -> None:
    """🔴 THE GUARD (the issue's mutation): Hermes sends a reply's text, then its
    ``MEDIA:`` files through ``send_document``, then completes the turn. The file is
    uploaded for the open request and the turn's post carries it. Without the override,
    Hermes's base answers a failure notice and no post carries anything."""

    async def scenario():
        adapter, wire, tunnel = with_files(workspace)
        report = made_file(workspace)
        await adapter.send(CONVERSATION, "Here is the report.")
        result = await adapter.send_document(CONVERSATION, str(report))
        await adapter.on_processing_complete(
            adapter_module.MessageEvent(text="x", message_id=REQUEST),
            ProcessingOutcome.SUCCESS,
        )
        return result, wire, tunnel

    result, wire, tunnel = run(scenario())
    assert result.success
    assert [op for op, _ in tunnel.calls] == [fm.UPLOAD_OP]
    op, payload = tunnel.calls[0]
    assert payload["requestId"] == REQUEST and "to" not in payload
    # By its path relative to the workspace: the companion reads it, nothing is copied.
    assert payload["localPath"] == "out/report.pdf"
    assert tunnel.present == [True]
    posts = wire.of_type("post")
    assert len(posts) == 2
    assert "attachments" not in posts[0]
    expected = {k: ATTACHED[k] for k in ("id", "name", "ref")}
    assert posts[1]["attachments"] == [expected]
    assert posts[1]["text"] == ATTACHED["name"]


def test_a_file_delivered_before_the_text_rides_the_text(workspace: Path) -> None:
    async def scenario():
        adapter, wire, tunnel = with_files(workspace)
        inside = workspace / "out" / "summary.pdf"
        inside.parent.mkdir()
        inside.write_bytes(b"pdf")
        await adapter.send_document(CONVERSATION, str(inside), file_name="ignored.pdf")
        await adapter.send(CONVERSATION, "Here it is.")
        return wire, tunnel

    wire, tunnel = run(scenario())
    # A file already inside the workspace goes as it is: nothing copied.
    assert tunnel.calls[0][1]["localPath"] == "out/summary.pdf"
    (post,) = wire.of_type("post")
    assert post["text"] == "Here it is."
    assert post["attachments"][0]["ref"] == ATTACHED["ref"]


def test_a_refused_upload_becomes_a_note_naming_the_file(workspace: Path, tmp_path: Path) -> None:
    """🔴 A refusal reaches the person, never vanishes: the reply's note names the file and
    why — and Hermes is told SUCCESS so it adds no generic notice of its own."""

    async def scenario():
        tunnel = Tunnel(workspace, {fm.UPLOAD_OP: refused("not_permitted", "path_refused")})
        adapter, wire, _ = with_files(workspace, tunnel)
        result = await adapter.send_document(CONVERSATION, str(made_file(workspace)))
        await adapter.send(CONVERSATION, "I made the report.")
        return result, wire

    result, wire = run(scenario())
    assert result.success
    (post,) = wire.of_type("post")
    assert "attachments" not in post
    assert post["text"] == ("I made the report.\n\n" + not_attached("report.pdf", "path_refused"))
    assert "report.pdf could not be attached: it is not a file this agent may share" in post["text"]


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        (refused("upstream_failed", "transfer_denied"), "may not share files"),
        (refused("timed_out"), "did not finish in time"),
        (refused("upstream_failed"), fm.NOT_ATTACHED_DEFAULT),
        (CallResult(call_id="c", payload={"path": "/x", "size": 1}), fm.NOT_ATTACHED_DEFAULT),
    ],
)
def test_every_failure_class_reads_as_a_sentence(
    workspace: Path, tmp_path: Path, answer: CallResult, reason: str
) -> None:
    async def scenario():
        adapter, wire, _ = with_files(workspace, Tunnel(workspace, {fm.UPLOAD_OP: answer}))
        await adapter.send_image_file(CONVERSATION, str(made_file(workspace, "chart.png")))
        await adapter._flush_outbox(REQUEST)
        return wire

    (post,) = run(scenario()).of_type("post")
    assert post["text"].startswith("chart.png could not be attached:")
    assert reason in post["text"]


def test_a_companion_without_files_is_said_on_the_reply(workspace: Path, tmp_path: Path) -> None:
    async def scenario():
        adapter, wire, tunnel = with_files(workspace)
        adapter._companion_features = frozenset({"tools"})
        await adapter.send_video(CONVERSATION, str(made_file(workspace, "clip.mp4")))
        await adapter._flush_outbox(REQUEST)
        return wire, tunnel

    wire, tunnel = run(scenario())
    assert tunnel.calls == []
    (post,) = wire.of_type("post")
    assert post["text"] == not_attached("clip.mp4", "files_unavailable")
    assert "needs updating" in post["text"]


def test_a_missing_file_and_an_unreachable_workspace_are_said(
    workspace: Path, tmp_path: Path
) -> None:
    async def scenario():
        adapter, wire, tunnel = with_files(workspace)
        await adapter.send_voice(CONVERSATION, str(workspace / "gone.mp3"))
        (workspace / "folder").mkdir()
        await adapter.send_document(CONVERSATION, str(workspace / "folder"))  # not a file
        adapter._workspace = tmp_path / "not-mounted"
        await adapter.send_document(CONVERSATION, str(outside_file(tmp_path)))
        await adapter._flush_outbox(REQUEST)
        return wire, tunnel

    wire, tunnel = run(scenario())
    assert tunnel.calls == []
    (post,) = wire.of_type("post")
    assert post["text"].split("\n\n") == [
        not_attached("gone.mp3", "local_file_missing"),
        not_attached("folder", "local_file_missing"),
        not_attached("report.pdf", "workspace_unreachable"),
    ]


@pytest.mark.parametrize("hook", ["send_document", "send_image_file", "send_video", "send_voice"])
def test_a_file_outside_the_workspace_is_never_copied_never_uploaded_and_said(
    workspace: Path, tmp_path: Path, hook: str
) -> None:
    """🔴 THE SECURITY GUARD. Hermes's media policy admits roots wider than the workspace,
    so a prompt-injected ``MEDIA:~/.ssh/id_ed25519`` it approved reaches these hooks. The
    file must never be copied into the workspace, never uploaded, and the person must read
    a note naming it — through every hook, and through a link inside the workspace that
    leads out of it."""
    secret = outside_file(tmp_path, "id_ed25519")
    link = workspace / "innocent.pdf"
    os.symlink(outside_file(tmp_path, "agent-key"), link)
    before = sorted(p.relative_to(workspace).as_posix() for p in workspace.rglob("*"))

    async def scenario():
        adapter, wire, tunnel = with_files(workspace)
        send = getattr(adapter, hook)
        assert (await send(CONVERSATION, str(secret))).success
        assert (await send(CONVERSATION, str(link))).success
        await adapter._flush_outbox(REQUEST)
        return wire, tunnel

    wire, tunnel = run(scenario())
    assert tunnel.calls == [], "a file outside the workspace reached the companion"
    after = sorted(p.relative_to(workspace).as_posix() for p in workspace.rglob("*"))
    assert after == before, "something was written into the workspace"
    (post,) = wire.of_type("post")
    assert "attachments" not in post
    assert post["text"].split("\n\n") == [
        not_attached("id_ed25519", "outside_workspace"),
        not_attached("innocent.pdf", "outside_workspace"),
    ]
    assert "not in the shared workspace, so it was not attached" in post["text"]


def test_a_file_with_no_open_request_is_dropped_never_guessed(
    workspace: Path, tmp_path: Path
) -> None:
    async def scenario():
        adapter, wire, tunnel = with_files(workspace)
        result = await adapter.send_document("another-conversation", str(outside_file(tmp_path)))
        return result, wire, tunnel

    result, wire, tunnel = run(scenario())
    assert not result.success
    assert tunnel.calls == [] and wire.of_type("post") == []


def test_a_request_withdrawn_while_its_file_moved_posts_nothing(
    workspace: Path, tmp_path: Path
) -> None:
    async def scenario():
        adapter, wire, tunnel = with_files(workspace)
        real = tunnel.call

        async def cancelled_meanwhile(op: str, payload: dict | None = None, **kw: Any):
            adapter._forget(REQUEST)
            return await real(op, payload, **kw)

        adapter._files._call = cancelled_meanwhile
        result = await adapter.send_document(CONVERSATION, str(made_file(workspace)))
        adapter._collect_attachment(REQUEST, {"id": "i", "name": "n", "ref": "r"})
        await adapter._flush_outbox(REQUEST)
        return result, wire

    result, wire = run(scenario())
    assert not result.success
    assert wire.of_type("post") == []


def test_a_reply_carries_at_most_twenty_files_and_the_rest_ride_the_next_post(
    workspace: Path,
) -> None:
    async def scenario():
        adapter, wire, _ = with_files(workspace)
        for n in range(23):
            adapter._collect_attachment(REQUEST, {"id": f"i{n}", "name": "", "ref": f"r{n}"})
        await adapter._flush_outbox(REQUEST)
        return wire

    posts = run(scenario()).of_type("post")
    assert [len(p["attachments"]) for p in posts] == [20, 3]
    assert posts[1]["text"] == "file, file, file"


def test_a_post_that_did_not_go_keeps_its_files(workspace: Path) -> None:
    async def scenario():
        adapter, wire, _ = with_files(workspace)
        adapter._collect_attachment(REQUEST, {"id": "i", "name": "a.pdf", "ref": "r"})
        adapter._outbox_for(REQUEST)["notes"].append("b.pdf could not be attached: x.")
        sent = wire.send

        async def drop(frame: dict) -> bool:
            return False

        wire.send = drop
        result = await adapter.send(CONVERSATION, "text")
        assert not result.success
        await adapter._flush_outbox(REQUEST)  # the flush's own failure is logged, not looped
        wire.send = sent
        adapter._give_back_outbox(REQUEST, [{"id": "i", "name": "a.pdf", "ref": "r"}], [])
        adapter._give_back_outbox(REQUEST, [], [])
        await adapter.send(CONVERSATION, "again")
        return wire

    (post,) = run(scenario()).of_type("post")
    assert post["attachments"] == [{"id": "i", "name": "a.pdf", "ref": "r"}]


def test_a_settling_turn_still_posts_a_file_a_tool_attached(workspace: Path) -> None:
    async def scenario():
        adapter, wire, _ = with_files(workspace)
        await adapter.send(CONVERSATION, "Working on it.")
        adapter._schedule_done(REQUEST, ProcessingOutcome.SUCCESS)
        adapter._collect_attachment(REQUEST, {"id": "i", "name": "late.pdf", "ref": "r"})
        await settle(adapter)
        return wire

    wire = run(scenario())
    assert [p["text"] for p in wire.of_type("post")] == ["Working on it.", "late.pdf"]
    (done,) = wire.of_type("done")
    assert done["outcome"] == "answered"


def test_a_finished_turn_closed_by_the_disconnect_posts_its_files_first(
    workspace: Path,
) -> None:
    async def scenario():
        adapter, wire, _ = with_files(workspace)
        await adapter.send(CONVERSATION, "Done.")
        adapter._schedule_done(REQUEST, ProcessingOutcome.SUCCESS)
        adapter._collect_attachment(REQUEST, {"id": "i", "name": "end.pdf", "ref": "r"})
        await adapter._close_finished_turns()
        return wire

    wire = run(scenario())
    assert [f["type"] for f in wire.sent] == ["post", "post", "done"]
    assert wire.sent[1]["attachments"][0]["name"] == "end.pdf"


def test_the_hooks_hermes_calls_carry_every_kind_of_file(workspace: Path, tmp_path: Path) -> None:
    async def scenario():
        adapter, _, tunnel = with_files(workspace)
        for hook, name in (
            (adapter.send_document, "a.pdf"),
            (adapter.send_image_file, "b.png"),
            (adapter.send_video, "c.mp4"),
            (adapter.send_voice, "d.mp3"),
        ):
            assert (await hook(CONVERSATION, str(made_file(workspace, name)))).success
        return adapter, tunnel

    adapter, tunnel = run(scenario())
    assert [p["localPath"].rsplit("/", 1)[1] for _, p in tunnel.calls] == [
        "a.pdf",
        "b.png",
        "c.mp4",
        "d.mp3",
    ]
    assert len(adapter._outbox[REQUEST]["attachments"]) == 4


def test_the_overrides_are_the_adapters_own() -> None:
    """The mutation's other face: each hook is the adapter's, never the base's notice."""
    base = adapter_module.BasePlatformAdapter
    for name in ("send_document", "send_image_file", "send_video", "send_voice"):
        assert getattr(adapter_module.UfficiAdapter, name) is not getattr(base, name), name


# ------------------------------------------------------------------ staging


def test_only_a_file_inside_the_workspace_has_a_path_to_upload(
    workspace: Path, tmp_path: Path
) -> None:
    inside = made_file(workspace)
    assert workspace_relative(str(inside), workspace) == "out/report.pdf"
    with pytest.raises(fm.OutsideWorkspace):
        workspace_relative(str(outside_file(tmp_path)), workspace)
    with pytest.raises(FileNotFoundError):
        workspace_relative(str(workspace / "missing"), workspace)
    with pytest.raises(NotADirectoryError):
        workspace_relative(str(inside), tmp_path / "absent")


def test_a_link_into_the_workspace_is_read_where_it_leads(workspace: Path, tmp_path: Path) -> None:
    inside = workspace / "made.pdf"
    inside.write_bytes(b"x")
    link = tmp_path / "link.pdf"
    os.symlink(inside, link)
    assert workspace_relative(str(link), workspace) == "made.pdf"


def test_a_link_inside_the_workspace_that_leads_out_is_outside(
    workspace: Path, tmp_path: Path
) -> None:
    link = workspace / "looks-local.pdf"
    os.symlink(outside_file(tmp_path), link)
    with pytest.raises(fm.OutsideWorkspace):
        workspace_relative(str(link), workspace)


def test_the_plugin_never_copies_or_reads_a_file() -> None:
    """No copy, and no read of a file's bytes, anywhere in the package: the companion moves
    every byte, behind its fence."""
    root = Path(fm.__file__).resolve().parent
    sources = sorted(root.glob("*.py"))
    assert len(sources) > 5
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for forbidden in ("shutil", "copyfile", "read_bytes", "sendfile"):
            assert forbidden not in text, f"{path.name} names {forbidden}"


def test_a_display_name_is_a_name() -> None:
    assert display_name("/tmp/a/b.pdf") == "b.pdf"
    assert display_name("/tmp/a/b.pdf", "x/y\\z.pdf") == "z.pdf"
    assert display_name("") == "file"
    assert len(display_name("/t/" + "n" * 400)) == fm.MAX_NAME_CHARS


# ------------------------------------------------------------------ the tools


def files_for(
    workspace: Path | None,
    *,
    registry: Registry | None = None,
    tunnel: Tunnel | None = None,
    ledger: dict[str, str] | None = None,
    available: bool = True,
    listed: frozenset[str] = frozenset({"read_file", "write_file"}),
):
    registry = registry if registry is not None else Registry()
    tunnel = tunnel or Tunnel(workspace)
    ledger = {"conv-a": "req-a"} if ledger is None else ledger
    attached: list[tuple[str, dict]] = []
    state = {"available": available, "workspace": workspace}
    tools = FileTools(
        registry.register_tool,
        call=tunnel.call,
        request_for=ledger.get,
        available=lambda: state["available"],
        workspace=lambda: state["workspace"],
        attached=lambda request, attachment: attached.append((request, attachment)),
    )
    tools.on_catalog(listed)
    return tools, registry, tunnel, attached, state


def test_the_tools_exist_only_with_the_companions_files_and_the_catalogs_permission(
    workspace: Path,
) -> None:
    """🔴 Registered only while the companion announces ``files`` AND the catalog lists the
    platform's ``read_file`` (download) / ``write_file`` (upload)."""
    tools, registry, _, _, state = files_for(workspace, listed=frozenset())
    assert registry.tools == {}
    tools.on_catalog(frozenset({"read_file"}))
    assert set(registry.tools) == {DOWNLOAD_TOOL}
    assert tools.download_offered
    tools.on_catalog(frozenset({"read_file", "write_file", "list_boards"}))
    assert set(registry.tools) == {DOWNLOAD_TOOL, UPLOAD_TOOL}
    for entry in registry.tools.values():
        assert entry["toolset"] == pt.TOOLSET
        assert str(workspace) in entry["description"]
        assert "Uffici files" in entry["description"] and "workspace" in entry["description"]
    tools.on_catalog(frozenset({"write_file"}))
    assert set(registry.tools) == {UPLOAD_TOOL}
    state["available"] = False
    tools.sync()
    assert registry.tools == {} and tools.offered == frozenset()
    # Without a workspace the companion's files are nowhere this side can read.
    state["available"], state["workspace"] = True, None
    tools.sync()
    assert registry.tools == {}


def test_a_new_workspace_re_registers_with_its_name(workspace: Path, tmp_path: Path) -> None:
    tools, registry, _, _, state = files_for(workspace)
    first = len(registry.calls)
    other = tmp_path / "other"
    state["workspace"] = other
    tools.sync()
    assert len(registry.calls) == first + 2
    assert all(str(other) in t["description"] for t in registry.tools.values())


def test_the_catalog_read_drives_the_file_tools_through_the_platform_tools(
    workspace: Path,
) -> None:
    seen: list[frozenset[str]] = []
    platform = pt.PlatformTools(
        None,
        call=Tunnel().call,
        request_for={}.get,
        enabled=lambda: True,
        on_catalog=seen.append,
    )
    platform.apply({"read_file": {"name": "read_file"}})
    assert seen == [frozenset({"read_file"})]


def test_a_refused_or_failing_registration_offers_nothing(workspace: Path) -> None:
    class Raising:
        def register_tool(self, **_k: Any) -> Any:
            raise RuntimeError("no")

    tools, _, _, _, _ = files_for(workspace, registry=Registry(taken={DOWNLOAD_TOOL}))
    assert tools.offered == frozenset({UPLOAD_TOOL})
    raising = FileTools(
        Raising().register_tool,
        call=Tunnel().call,
        request_for={}.get,
        available=lambda: True,
        workspace=lambda: workspace,
        attached=lambda *_: None,
    )
    raising.on_catalog(frozenset({"read_file"}))
    assert raising.offered == frozenset()
    inert = FileTools(
        None,
        call=Tunnel().call,
        request_for={}.get,
        available=lambda: True,
        workspace=lambda: workspace,
        attached=lambda *_: None,
    )
    inert.on_catalog(frozenset({"read_file"}))
    assert inert.offered == frozenset()


def test_a_dispose_that_raises_still_takes_the_tool_away(workspace: Path) -> None:
    class Stubborn:
        def dispose(self) -> None:
            raise RuntimeError("no")

    tools, _, _, _, state = files_for(workspace)
    tools._handles[DOWNLOAD_TOOL] = Stubborn()
    tools._handles[UPLOAD_TOOL] = object()
    state["available"] = False
    tools.sync()
    assert tools.offered == frozenset()


def test_download_by_attachment_answers_where_it_landed_never_its_content(
    workspace: Path,
) -> None:
    tools, _, tunnel, _, _ = files_for(workspace)
    answer = json.loads(run(tools.invoke("conv-a", DOWNLOAD_TOOL, {"attachment_id": " 1042.0 "})))
    assert tunnel.calls == [(fm.DOWNLOAD_OP, {"requestId": "req-a", "attachmentId": "1042.0"})]
    assert answer == {
        "path": str(workspace / DOWNLOADED["localPath"]),
        "workspacePath": DOWNLOADED["localPath"],
        "size": DOWNLOADED["size"],
        "contentType": DOWNLOADED["contentType"],
    }
    run(tools.invoke("conv-a", DOWNLOAD_TOOL, {"path": "reports/2026-q3.pdf"}))
    assert tunnel.calls[-1][1] == {"requestId": "req-a", "path": "reports/2026-q3.pdf"}


@pytest.mark.parametrize("args", [{}, {"attachment_id": "1", "path": "p"}, {"path": "  "}, "x"])
def test_download_takes_exactly_one_of_its_two(workspace: Path, args: Any) -> None:
    tools, _, tunnel, _, _ = files_for(workspace)
    answer = json.loads(run(tools.invoke("conv-a", DOWNLOAD_TOOL, args)))
    assert answer["errorClass"] == "malformed_payload" and tunnel.calls == []


def test_a_failed_transfer_relays_its_instruction_and_class(workspace: Path) -> None:
    tunnel = Tunnel(workspace, {fm.DOWNLOAD_OP: refused("upstream_failed", "transfer_denied")})
    tools, _, _, _, _ = files_for(workspace, tunnel=tunnel)
    answer = json.loads(run(tools.invoke("conv-a", DOWNLOAD_TOOL, {"attachment_id": "1"})))
    assert answer == {
        "error": "Tell the person it could not be done.",
        "errorClass": "transfer_denied",
    }


@pytest.mark.parametrize(
    "payload", [{"localPath": "../x", "size": 1}, {"localPath": "a", "size": "1"}, {}]
)
def test_a_download_answer_outside_the_wire_is_malformed(workspace: Path, payload: dict) -> None:
    tunnel = Tunnel(workspace, {fm.DOWNLOAD_OP: CallResult(call_id="c", payload=payload)})
    tools, _, _, _, _ = files_for(workspace, tunnel=tunnel)
    answer = json.loads(run(tools.invoke("conv-a", DOWNLOAD_TOOL, {"attachment_id": "1"})))
    assert answer["errorClass"] == "malformed_payload"


def test_upload_without_to_attaches_to_the_reply(workspace: Path) -> None:
    tools, _, tunnel, attached, _ = files_for(workspace)
    answer = json.loads(run(tools.invoke("conv-a", UPLOAD_TOOL, {"local_path": "out/summary.pdf"})))
    assert tunnel.calls == [(fm.UPLOAD_OP, {"requestId": "req-a", "localPath": "out/summary.pdf"})]
    assert answer == {"attachedToReply": True, "name": ATTACHED["name"], "size": ATTACHED["size"]}
    assert attached == [("req-a", {k: ATTACHED[k] for k in ("id", "name", "ref")})]
    assert "ref" not in answer and "content" not in answer


def test_upload_with_to_saves_into_files(workspace: Path) -> None:
    tools, _, tunnel, attached, _ = files_for(workspace)
    answer = json.loads(
        run(
            tools.invoke(
                "conv-a",
                UPLOAD_TOOL,
                {"local_path": "out/chart.png", "to": "reports/chart.png", "overwrite": True},
            )
        )
    )
    assert tunnel.calls[0][1] == {
        **op_golden("files.upload.request-to-files"),
        "requestId": "req-a",
    }
    assert answer == {"savedAt": SAVED["path"], "size": SAVED["size"]}
    assert attached == []


@pytest.mark.parametrize(
    ("args", "payload"),
    [
        ({"local_path": "a"}, {"ref": "r"}),
        ({"local_path": "a", "to": "b"}, {"path": "", "size": 1}),
    ],
)
def test_an_upload_answer_outside_the_wire_is_malformed(
    workspace: Path, args: dict, payload: dict
) -> None:
    tunnel = Tunnel(workspace, {fm.UPLOAD_OP: CallResult(call_id="c", payload=payload)})
    tools, _, _, attached, _ = files_for(workspace, tunnel=tunnel)
    answer = json.loads(run(tools.invoke("conv-a", UPLOAD_TOOL, args)))
    assert answer["errorClass"] == "malformed_payload" and attached == []


def test_upload_refusals(workspace: Path) -> None:
    tunnel = Tunnel(workspace, {fm.UPLOAD_OP: refused("not_permitted", "path_refused")})
    tools, _, _, attached, _ = files_for(workspace, tunnel=tunnel)
    answer = json.loads(run(tools.invoke("conv-a", UPLOAD_TOOL, {"local_path": ".ssh/id"})))
    assert answer["errorClass"] == "path_refused" and attached == []
    missing = json.loads(run(tools.invoke("conv-a", UPLOAD_TOOL, {"local_path": " "})))
    assert missing["errorClass"] == "malformed_payload"


def test_outside_a_request_or_without_the_tool_nothing_moves(workspace: Path) -> None:
    tools, _, tunnel, _, _ = files_for(workspace, listed=frozenset({"read_file"}))
    for conversation in (None, "conv-other"):
        answer = json.loads(run(tools.invoke(conversation, DOWNLOAD_TOOL, {"attachment_id": "1"})))
        assert answer == {"error": pt.OUTSIDE_A_REQUEST}
    answer = json.loads(run(tools.invoke("conv-a", UPLOAD_TOOL, {"local_path": "a"})))
    assert answer["errorClass"] == "unknown_op"
    assert tunnel.calls == []


def test_the_registered_handler_runs_the_call_on_the_gateways_loop(workspace: Path) -> None:
    tools, registry, tunnel, _, _ = files_for(workspace)
    with Loop() as loop:
        tools.bind_loop(loop)
        handler = registry.tools[DOWNLOAD_TOOL]["handler"]
        answer = in_turn("uffici", "conv-a", lambda: handler({"attachment_id": "1"}))
    assert json.loads(answer)["size"] == DOWNLOADED["size"]
    assert tunnel.calls[0][1]["requestId"] == "req-a"


def test_the_file_tools_are_never_reported_as_hermess_own_activity() -> None:
    tools = pt.PlatformTools(
        None, call=Tunnel().call, request_for={"conv-a": "req-a"}.get, enabled=lambda: True
    )
    with Loop() as loop:
        tools.bind_loop(loop)
        for name in (DOWNLOAD_TOOL, UPLOAD_TOOL, "terminal"):
            in_turn("uffici", "conv-a", lambda n=name: tools.on_tool_called(n, "ok"))
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(timeout=5)
    assert [a["tool"] for a in tools.take_extras("req-a")["activity"]] == ["terminal"]


def test_no_path_or_name_reaches_a_log_line(
    workspace: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """🔴 Paths, file names and refs are tenant content: logged never, by any of the paths
    a file takes."""
    caplog.set_level("DEBUG")

    async def scenario():
        tunnel = Tunnel(workspace, {fm.DOWNLOAD_OP: refused("upstream_failed", "x")})
        adapter, _, _ = with_files(workspace, tunnel)
        await adapter.send_document(CONVERSATION, str(outside_file(tmp_path, "secret-plan.pdf")))
        await adapter._flush_outbox(REQUEST)
        tools, _, _, _, _ = files_for(workspace, tunnel=tunnel)
        await tools.invoke("conv-a", DOWNLOAD_TOOL, {"path": "secret-plan.pdf"})
        await tools.invoke("conv-a", UPLOAD_TOOL, {"local_path": "secret-plan.pdf"})

    run(scenario())
    assert caplog.records, "nothing was logged, so this guard asserts nothing"
    for record in caplog.records:
        line = record.getMessage() + json.dumps(record.__dict__, default=str)
        assert "secret-plan" not in line, line
        assert str(tmp_path) not in record.getMessage()
