"""Files between Uffici and Hermes, in Hermes's own conventions (#2674, epic #2667).

The companion moves every byte, the same way for every harness (``files.download``,
``files.upload``, a fenced workspace); what Hermes does with a file is decided HERE and
nowhere else. The plugin stays a format translator: no network code, no credential, and
never a file's content in a frame, a tool result or a log line — paths, sizes and refs.

- **In.** A brief's attachment the companion prefetched (``localPath``, relative to the
  workspace its hello names) becomes Hermes's inbound media: an entry in
  ``MessageEvent.media_urls`` with its MIME type in ``media_types`` — what Hermes's other
  platforms hand it after caching an image or a document — so an image reaches the model's
  vision and a document is one Hermes tells the agent to read. One that was not prefetched
  stays a reference line, with ``uffici_download`` named as the way to fetch it.
- **Out.** Hermes delivers a file it made through its own ``MEDIA:<path>`` convention: it
  validates the path against its delivery policy and calls the adapter's ``send_document``
  / ``send_image_file`` (``send_video``, ``send_voice`` for those kinds). The adapter
  uploads each through ``files.upload`` for the open request, and the request's next post
  carries every attachment collected. 🔴 ONLY a file inside the workspace is ever uploaded,
  and it is handed to the companion as a path — never copied, never read here. A file
  outside it (or a link inside it that leads out) is refused before any call, and the
  reply says so, naming the file. Hermes's media policy admits more roots than the
  workspace (its caches, an operator's list), and a prompt-injected ``MEDIA:~/.ssh/…`` it
  approved must not become an upload: the companion's workspace fence is the one decision
  about what leaves this machine (epic #2667 decision 8), and this plugin never moves a file
  across it. A file the agent wants to send is written into the workspace first.
- **Tools.** ``uffici_download`` and ``uffici_upload``, the plugin's own (like
  ``uffici_recall``), registered in the ``uffici`` toolset only while the companion
  announces ``files`` and the agent's catalog lists ``read_file`` (download) / ``write_file``
  (upload) — the platform enforces the same permissions on every transfer (#2671).

🔴 Logged: tool and op names, request ids, sizes and error classes. Never a path, a file
name or a ref: they are tenant content.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
from collections.abc import Awaitable, Callable
from pathlib import Path, PurePosixPath
from typing import Any

from .call_contract import INSTRUCTIONS, MAX_DEADLINE_MS
from .calls import CallResult
from .platform_tools import (
    DOWNLOAD_TOOL,
    OUTSIDE_A_REQUEST,
    TOOLSET,
    UPLOAD_TOOL,
    refusal,
    run_on_loop,
    session_conversation,
)

logger = logging.getLogger(__name__)

#: The companion's word for "the plugin may call files.download and files.upload, and the
#: hello names the workspace", in its hello.
FEATURE_FILES = "files"

#: What this plugin asks the companion to prefetch per brief (``capabilities``).
#:
#: 🔴 25 MiB: one conversation attachment at the platform's cap always arrives before the
#: turn, so a person's PDF or photo is already a local file when Hermes reads the message —
#: and a burst of large files never holds the brief back for long, because the companion
#: downloads them BEFORE handing the brief over. Anything past it stays a reference line the
#: agent fetches with ``uffici_download`` when it needs it. Hermes inlines an image into the
#: model's request and reads a document with its own tools, so a larger budget would mostly
#: buy waiting for files the turn may never open. The companion's own cap may be lower.
PREFETCH_MAX_BYTES = 25 * 1024 * 1024

#: The ops the two tools and the reply's files ride.
DOWNLOAD_OP = "files.download"
UPLOAD_OP = "files.upload"

#: A transfer's deadline: the wire's ceiling. The companion resumes a dropped transfer
#: inside the same call, and the call's deadline bounds all of it.
TRANSFER_DEADLINE_MS = MAX_DEADLINE_MS
TRANSFER_WAIT_SECONDS = TRANSFER_DEADLINE_MS / 1000 + 10.0

#: The most attachments one post carries (``post.schema.json``).
MAX_POST_ATTACHMENTS = 20

#: A file name the wire accepts on an attachment (``post.schema.json``).
MAX_NAME_CHARS = 255

#: Why a prefetched attachment did not arrive, as the reference line says it. A class not
#: listed here — a newer companion's — is shown as it came.
PREFETCH_REASONS = {
    "over_budget": "too large to fetch before the turn",
    "cancelled": "the download was stopped",
    "transfer_denied": "Uffici did not allow the download",
    "file_too_large": "larger than this machine accepts",
    "path_refused": "its name cannot be written on this machine",
    "checksum_mismatch": "the download did not arrive intact",
    "size_mismatch": "the download did not arrive intact",
}

#: The prefetch failures a download would meet again: no point naming the tool.
PREFETCH_FINAL = frozenset({"transfer_denied", "file_too_large", "path_refused"})

#: Why a file the agent delivered could not be attached, as the PERSON reads it on the
#: reply. 🔴 The person always learns that a file did not arrive (#1958 decision 32): the
#: class is the platform's or the companion's, the sentence is for a person.
NOT_ATTACHED_REASONS = {
    "path_refused": "it is not a file this agent may share from this machine",
    "local_file_missing": "the file was not found on this machine",
    "transfer_denied": "this agent may not share files in this conversation",
    "not_permitted": "this agent may not share files in this conversation",
    "file_too_large": "it is larger than Uffici accepts",
    "destination_refused": "Uffici would not store it",
    "checksum_mismatch": "it changed while it was being sent",
    "size_mismatch": "it changed while it was being sent",
    "timed_out": "the upload did not finish in time",
    "cancelled": "the turn was stopped before it was sent",
    "unknown_op": "the Uffici companion on this machine cannot send files and needs updating",
    "files_unavailable": (
        "the Uffici companion on this machine cannot send files and needs updating"
    ),
    "workspace_unreachable": "the companion's file workspace cannot be reached from Hermes",
    "outside_workspace": "it is not in the shared workspace, so it was not attached",
    "too_many_files": "a reply carries at most twenty files",
}
NOT_ATTACHED_DEFAULT = "the upload failed"

#: What the model reads when a tool cannot run, written TO the agent (decision 32).
WORKSPACE_UNREACHABLE = (
    "The Uffici file workspace is not reachable from this Hermes, so nothing was moved. Tell "
    "the person the file could not be transferred."
)
ONE_OF_TWO = (
    "Give exactly one of attachment_id and path, so nothing was downloaded. Tell the person if "
    "you cannot tell which file they meant."
)


#: The images Hermes hands a model's vision (``gateway/platforms/base.py``,
#: ``SUPPORTED_IMAGE_DOCUMENT_TYPES``), transcribed and pinned against the real Hermes by
#: ``dev/hermes_contract.py``.
HERMES_IMAGE_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

#: Any type Hermes cannot place: a document it tells the agent to read.
OPAQUE_TYPE = "application/octet-stream"


def media_type_for(name: str) -> str:
    """The MIME type Hermes routes an inbound file by: ``image/*`` reaches the model's
    vision, anything else is a document (audio and video get Hermes's own notes).

    🔴 By the file's name, because an entry's attachment carries no type on the wire. Only
    the images Hermes itself treats as images are ``image/*``: a TIFF or a HEIC sent to a
    vision model is a request the provider refuses, so those go as a document the agent
    opens with its own tools. An unknown name is ``application/octet-stream``."""
    text = str(name or "")
    suffix = PurePosixPath(text.replace("\\", "/")).suffix.lower()
    if suffix in HERMES_IMAGE_TYPES:
        return HERMES_IMAGE_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(text, strict=False)
    if not guessed or guessed.startswith("image/"):
        return OPAQUE_TYPE
    return guessed


def workspace_path(workspace: Path | None, local_path: Any) -> Path | None:
    """A wire ``localPath`` (relative to the workspace, forward slashes) as this machine's
    absolute path, or None when it is not one the plugin can place."""
    if workspace is None or not isinstance(local_path, str) or not local_path:
        return None
    relative = PurePosixPath(local_path)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    return workspace.joinpath(*relative.parts)


def inbound_media(entries: Any, workspace: Path | None) -> tuple[list[str], list[str]]:
    """Every prefetched attachment of a delta, as Hermes's ``media_urls`` and ``media_types``."""
    urls: list[str] = []
    types: list[str] = []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        for attachment in entry.get("attachments") or []:
            if not isinstance(attachment, dict):
                continue
            local = workspace_path(workspace, attachment.get("localPath"))
            if local is None:
                continue
            urls.append(str(local))
            types.append(media_type_for(str(attachment.get("name") or local.name)))
    return urls, types


def prefetch_error_class(attachment: dict[str, Any]) -> str | None:
    error = attachment.get("prefetchError")
    if not isinstance(error, dict):
        return None
    name = error.get("class")
    return str(name) if isinstance(name, str) and name else None


def attachment_suffix(attachment: dict[str, Any], workspace: Path | None, fetchable: bool) -> str:
    """What a reference line adds to ``name — ref``: where the file is on this machine, or
    why it is not and how to fetch it."""
    local = workspace_path(workspace, attachment.get("localPath"))
    if local is not None:
        return f" — on this machine at {local}"
    failed = prefetch_error_class(attachment)
    said = f" — not downloaded ({PREFETCH_REASONS.get(failed, failed)})" if failed else ""
    attachment_id = str(attachment.get("id") or "").strip()
    if fetchable and attachment_id and failed not in PREFETCH_FINAL:
        said += f" — {DOWNLOAD_TOOL} with attachment_id {attachment_id!r} fetches it"
    return said


def not_attached(name: str, error_class: str | None) -> str:
    """The note a reply carries for a file that did not reach it."""
    reason = NOT_ATTACHED_REASONS.get(error_class or "", NOT_ATTACHED_DEFAULT)
    return f"{name} could not be attached: {reason}."


def failure_class(answered: CallResult) -> str:
    """The most specific class a failed call carries: the platform's or the companion's
    own (``upstream.class``) when there is one, else the call's."""
    upstream = answered.upstream if isinstance(answered.upstream, dict) else {}
    specific = upstream.get("class")
    if isinstance(specific, str) and specific:
        return specific
    return answered.error_class or "upstream_failed"


def display_name(path: str, name: str | None = None) -> str:
    """The name a file is shown and uploaded under: the one Hermes gave, else its own."""
    chosen = str(name or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not chosen:
        chosen = Path(str(path or "")).name.strip()
    return (chosen or "file")[:MAX_NAME_CHARS]


class OutsideWorkspace(ValueError):
    """The file is not in the workspace: it is never uploaded, and never copied there."""


def workspace_relative(source: str, workspace: Path) -> str:
    """The workspace-relative path the companion uploads ``source`` from — only when the
    file, with every link resolved, is a regular file INSIDE the workspace.

    🔴 Nothing is copied and nothing is read here: the path is all that crosses, and the
    companion's fence decides again. A file outside the workspace, or a link inside it that
    leads out, raises ``OutsideWorkspace``; a source that is not a regular file raises
    ``FileNotFoundError``; a workspace this machine cannot reach, ``NotADirectoryError``.
    """
    root = workspace.resolve(strict=False)
    if not root.is_dir():
        raise NotADirectoryError("workspace")
    resolved = Path(source).expanduser().resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise OutsideWorkspace("source")
    if not resolved.is_file():
        raise FileNotFoundError("source")
    return resolved.relative_to(root).as_posix()


def attachment_of(payload: Any) -> dict[str, str] | None:
    """A ``files.upload`` result without ``to`` as the post's attachment, or None."""
    if not isinstance(payload, dict):
        return None
    ref, ident, name = payload.get("ref"), payload.get("id"), payload.get("name")
    if not (isinstance(ref, str) and ref and isinstance(ident, str) and ident):
        return None
    return {"id": ident, "name": name if isinstance(name, str) else "", "ref": ref}


def _schemas(workspace: Path | None) -> dict[str, dict[str, Any]]:
    """The two tools in the OpenAI function shape Hermes registers.

    🔴 Each says what it moves and between which two places — this organization's Uffici
    files and the local workspace — and that a path on this side is relative to the
    workspace, which is named, so the model knows where to read and write."""
    where = f" The workspace is {workspace}." if workspace is not None else ""
    return {
        DOWNLOAD_TOOL: {
            "name": DOWNLOAD_TOOL,
            "description": (
                "Download a file from this organization's Uffici files into the local "
                "workspace, so your own tools can open it: an attachment of this conversation "
                "by its attachment_id (as a message's attachment line names it), or a file in "
                "Uffici Files by its path. Answers where it landed — an absolute path and the "
                "path relative to the workspace — its size and its type; never its content." + where
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "attachment_id": {
                        "type": "string",
                        "description": "An attachment's id, from its attachment line.",
                    },
                    "path": {
                        "type": "string",
                        "description": "A path in this organization's Uffici Files.",
                    },
                },
            },
        },
        UPLOAD_TOOL: {
            "name": UPLOAD_TOOL,
            "description": (
                "Upload a file from the local workspace to this organization's Uffici files. "
                "Without `to` the file is attached to your reply in this conversation; with "
                "`to` it is saved in Uffici Files at that path instead. local_path is "
                "relative to the workspace (or an absolute path inside it): write the file "
                "there first. Replacing a file already at `to` needs overwrite, and a person "
                "approves it first. Answers the attachment or the saved path and the size." + where
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "local_path": {
                        "type": "string",
                        "description": "The file, relative to the workspace.",
                    },
                    "to": {
                        "type": "string",
                        "description": "A path in Uffici Files to save it at, instead.",
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Replace a file already at `to`.",
                    },
                },
                "required": ["local_path"],
            },
        },
    }


class FileTools:
    """The two file tools, and the upload behind a reply's files.

    ``register_tool`` is Hermes's ``PluginContext.register_tool`` (None outside Hermes);
    ``call`` the call tunnel's ``CallClient.call``; ``request_for`` the adapter's ledger;
    ``available`` whether the attached companion announced ``files``; ``workspace`` the
    workspace its hello named; ``attached`` how an upload without ``to`` reaches the
    request's next post.
    """

    def __init__(
        self,
        register_tool: Callable[..., Any] | None,
        *,
        call: Callable[..., Awaitable[CallResult]],
        request_for: Callable[[str], str | None],
        available: Callable[[], bool],
        workspace: Callable[[], Path | None],
        attached: Callable[[str, dict[str, str]], None],
        session_reader: Callable[[], str | None] = session_conversation,
    ) -> None:
        self._register_tool = register_tool
        self._call = call
        self._request_for = request_for
        self._available = available
        self._workspace = workspace
        self._attached = attached
        self._session_reader = session_reader
        self._loop: asyncio.AbstractEventLoop | None = None
        #: The agent's catalog as last listed: which file tools its permissions allow.
        self._listed: frozenset[str] = frozenset()
        self._handles: dict[str, Any] = {}
        #: The workspace the registered descriptions name.
        self._described: Path | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def available(self) -> bool:
        return bool(self._available()) and self._workspace() is not None

    @property
    def offered(self) -> frozenset[str]:
        return frozenset(self._handles)

    @property
    def download_offered(self) -> bool:
        """Whether a reference line may name ``uffici_download``: only while it exists."""
        return DOWNLOAD_TOOL in self._handles

    # -------------------------------------------------------------- registration

    def on_catalog(self, listed: frozenset[str]) -> None:
        """The agent's catalog was read: offer what it now allows."""
        self._listed = frozenset(listed)
        self.sync()

    def wanted(self) -> frozenset[str]:
        """The tools to offer now: none without the companion's ``files``, and each only
        while the catalog lists the platform's tool with the same permission."""
        if not self.available():
            return frozenset()
        wanted = set()
        if "read_file" in self._listed:
            wanted.add(DOWNLOAD_TOOL)
        if "write_file" in self._listed:
            wanted.add(UPLOAD_TOOL)
        return frozenset(wanted)

    def sync(self) -> None:
        if self._register_tool is None:
            return
        wanted = self.wanted()
        workspace = self._workspace()
        if workspace != self._described:
            # The descriptions name the workspace: a new one re-registers both.
            for name in list(self._handles):
                self._dispose(name)
        for name in [n for n in self._handles if n not in wanted]:
            self._dispose(name)
        schemas = _schemas(workspace)
        added = 0
        for name in sorted(wanted - set(self._handles)):
            schema = schemas[name]
            try:
                handle = self._register_tool(
                    name=name,
                    toolset=TOOLSET,
                    schema=schema,
                    handler=self._handler_for(name),
                    description=schema["description"],
                    emoji="📎",
                )
            except Exception as exc:  # noqa: BLE001 — a refused tool never fails the rest
                logger.warning(
                    "uffici: Hermes refused the file tool %s (%s)",
                    name,
                    type(exc).__name__,
                    extra={"tool": name},
                )
                continue
            if handle is None:
                logger.warning(
                    "uffici: the file tool %s was not offered — Hermes already has a tool of "
                    "that name",
                    name,
                    extra={"tool": name},
                )
                continue
            self._handles[name] = handle
            added += 1
        self._described = workspace
        if added:
            logger.info("uffici: offering %d Uffici file tool(s) to Hermes", len(self._handles))

    def _dispose(self, name: str) -> None:
        handle = self._handles.pop(name, None)
        dispose = getattr(handle, "dispose", None)
        if not callable(dispose):
            return
        try:
            dispose()
        except Exception as exc:  # noqa: BLE001 — the handler refuses a tool that left
            logger.warning(
                "uffici: Hermes would not remove the file tool %s (%s)",
                name,
                type(exc).__name__,
                extra={"tool": name},
            )

    # ------------------------------------------------------------------ the tools

    def _handler_for(self, name: str) -> Callable[[dict[str, Any]], str]:
        def handle(args: dict[str, Any]) -> str:
            # 🔴 Read HERE, in the calling thread: the turn's session is its ContextVars.
            conversation_id = self._session_reader()
            return run_on_loop(
                self._loop,
                lambda: self.invoke(conversation_id, name, args),
                wait_seconds=TRANSFER_WAIT_SECONDS,
                tool=name,
            )

        handle.__name__ = f"uffici_files_{name}"
        return handle

    async def invoke(self, conversation_id: str | None, name: str, args: Any) -> str:
        """One file tool call on the request open on ``conversation_id``."""
        request_id = self._request_for(conversation_id) if conversation_id else None
        if not request_id:
            logger.warning(
                "uffici: the file tool %s was called outside a Uffici request; refused",
                name,
                extra={"tool": name},
            )
            return refusal(OUTSIDE_A_REQUEST)
        if name not in self.wanted():
            return refusal(INSTRUCTIONS["unknown_op"], "unknown_op")
        args = args if isinstance(args, dict) else {}
        if name == DOWNLOAD_TOOL:
            return await self._download(request_id, args)
        return await self._upload(request_id, args)

    async def _download(self, request_id: str, args: dict[str, Any]) -> str:
        attachment_id, path = args.get("attachment_id"), args.get("path")
        given = [v for v in (attachment_id, path) if isinstance(v, str) and v.strip()]
        if len(given) != 1:
            return refusal(ONE_OF_TWO, "malformed_payload")
        payload: dict[str, Any] = {"requestId": request_id}
        if isinstance(attachment_id, str) and attachment_id.strip():
            payload["attachmentId"] = attachment_id.strip()
        else:
            payload["path"] = str(path).strip()
        answered = await self._call(DOWNLOAD_OP, payload, deadline_ms=TRANSFER_DEADLINE_MS)
        if not answered.ok:
            return self._failed(DOWNLOAD_TOOL, request_id, answered)
        result = answered.payload or {}
        local = workspace_path(self._workspace(), result.get("localPath"))
        if local is None or not isinstance(result.get("size"), int):
            return refusal(INSTRUCTIONS["malformed_payload"], "malformed_payload")
        logger.info(
            "uffici: %s ran on request %s (%d bytes)",
            DOWNLOAD_TOOL,
            request_id,
            result["size"],
            extra={"tool": DOWNLOAD_TOOL, "requestId": request_id, "bytes": result["size"]},
        )
        return json.dumps(
            {
                "path": str(local),
                "workspacePath": result["localPath"],
                "size": result["size"],
                "contentType": str(result.get("contentType") or ""),
            },
            ensure_ascii=False,
        )

    async def _upload(self, request_id: str, args: dict[str, Any]) -> str:
        local_path = args.get("local_path")
        if not isinstance(local_path, str) or not local_path.strip():
            return refusal(INSTRUCTIONS["malformed_payload"], "malformed_payload")
        payload: dict[str, Any] = {"requestId": request_id, "localPath": local_path.strip()}
        to = args.get("to")
        if isinstance(to, str) and to.strip():
            payload["to"] = to.strip()
            if isinstance(args.get("overwrite"), bool):
                payload["overwrite"] = args["overwrite"]
        # 🔴 The path goes as the model gave it: the COMPANION's fence decides what may be
        # read, and nothing is ever copied into the workspace for it.
        answered = await self._call(UPLOAD_OP, payload, deadline_ms=TRANSFER_DEADLINE_MS)
        if not answered.ok:
            return self._failed(UPLOAD_TOOL, request_id, answered)
        result = answered.payload or {}
        size = result.get("size")
        if "to" in payload:
            saved = result.get("path")
            if not isinstance(saved, str) or not saved or not isinstance(size, int):
                return refusal(INSTRUCTIONS["malformed_payload"], "malformed_payload")
            body: dict[str, Any] = {"savedAt": saved, "size": size}
        else:
            attachment = attachment_of(result)
            if attachment is None or not isinstance(size, int):
                return refusal(INSTRUCTIONS["malformed_payload"], "malformed_payload")
            # The request's next post carries it — or the post the turn ends with.
            self._attached(request_id, attachment)
            body = {"attachedToReply": True, "name": attachment["name"], "size": size}
        logger.info(
            "uffici: %s ran on request %s (%s, %d bytes)",
            UPLOAD_TOOL,
            request_id,
            "saved" if "to" in payload else "attached",
            size,
            extra={"tool": UPLOAD_TOOL, "requestId": request_id, "bytes": size},
        )
        return json.dumps(body, ensure_ascii=False)

    def _failed(self, tool: str, request_id: str, answered: CallResult) -> str:
        error_class = failure_class(answered)
        logger.warning(
            "uffici: %s failed on request %s (%s)",
            tool,
            request_id,
            error_class,
            extra={"tool": tool, "requestId": request_id, "class": error_class},
        )
        return refusal(answered.instruction or INSTRUCTIONS["upstream_failed"], error_class)

    # ------------------------------------------------------------------ the reply's files

    async def attach(
        self, request_id: str, source: str
    ) -> tuple[dict[str, str] | None, str | None]:
        """Upload one file Hermes delivered, for the request's reply: the attachment, or
        the class saying why there is none. Never raises.

        🔴 Only a file inside the workspace is uploaded, by its path: anything else is
        ``outside_workspace`` before any call is made, and nothing is ever copied."""
        if not self.available():
            return None, "files_unavailable"
        workspace = self._workspace()
        assert workspace is not None  # available() says so
        try:
            relative = await asyncio.to_thread(workspace_relative, source, workspace)
        except OutsideWorkspace:
            logger.info(
                "uffici: a file Hermes delivered is outside the workspace; not attached",
                extra={"requestId": request_id},
            )
            return None, "outside_workspace"
        except NotADirectoryError:
            return None, "workspace_unreachable"
        except (FileNotFoundError, IsADirectoryError):
            return None, "local_file_missing"
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "uffici: a file Hermes delivered could not be located (%s)",
                type(exc).__name__,
                extra={"requestId": request_id},
            )
            return None, "local_file_missing"
        answered = await self._call(
            UPLOAD_OP,
            {"requestId": request_id, "localPath": relative},
            deadline_ms=TRANSFER_DEADLINE_MS,
        )
        if not answered.ok:
            error_class = failure_class(answered)
            logger.warning(
                "uffici: a file Hermes delivered was not attached on request %s (%s)",
                request_id,
                error_class,
                extra={"requestId": request_id, "class": error_class},
            )
            return None, error_class
        attachment = attachment_of(answered.payload)
        if attachment is None:
            return None, "malformed_payload"
        logger.info(
            "uffici: a file Hermes delivered was uploaded for request %s",
            request_id,
            extra={"requestId": request_id},
        )
        return attachment, None
