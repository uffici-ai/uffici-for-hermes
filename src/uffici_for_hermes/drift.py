"""The harness's drift from the plugin policy, as the hello reports it (#2406).

Three kinds, each read from Hermes's own configuration through Hermes's own readers, and
sent to the companion in ``hello.drift`` so ``uffici doctor`` can show them
(#1994 decision 13, a client-reported signal for #1958 decision 27):

- ``mcpServers`` — the MCP servers Hermes has configured and enabled
  (``hermes_cli.mcp_config._get_mcp_servers`` and ``tools.mcp_tool_common.mcp_server_enabled``,
  the one reader of the ``enabled`` key every Hermes surface uses);
- ``channels`` — the platforms Hermes's gateway would connect besides Uffici
  (``gateway.config.load_gateway_config().get_connected_platforms()``, which sees an
  environment-enabled platform as well as a configured one);
- ``localMemory`` — the memory keys that keep a memory of Hermes's own
  (:func:`uffici_for_hermes.memory.local_memory_keys`).

🔴 NEVER a refusal, and never the reason a hello is refused: every name is flattened and
clipped to the wire's bound and every list cut to its length HERE, and a source that cannot
be read leaves its kind out — which the doctor says, rather than reading it as clean —
instead of raising. Names only: never a URL, a header, a path or a credential.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Any

from .memory import BUILTIN_STORE_KEYS, PROVIDER_KEY, local_memory_keys

logger = logging.getLogger(__name__)

__all__ = [
    "BUILTIN_STORE_KEYS",
    "MAX_LOCAL_MEMORY",
    "MAX_NAME_CHARS",
    "MAX_NAMES",
    "PROVIDER_KEY",
    "bounded_names",
    "connected_platforms",
    "enabled_mcp_servers",
    "local_memory_keys",
    "read_drift",
]

#: The wire's bounds (``hello.schema.json``, ``drift``).
MAX_NAME_CHARS = 64
MAX_NAMES = 32
MAX_LOCAL_MEMORY = 8


def enabled_mcp_servers(config: dict[str, Any]) -> list[str]:
    """The MCP servers Hermes would start, through Hermes's own readers."""
    from hermes_cli.mcp_config import _get_mcp_servers  # type: ignore[import-not-found]
    from tools.mcp_tool_common import mcp_server_enabled  # type: ignore[import-not-found]

    return [
        str(name)
        for name, server in _get_mcp_servers(config).items()
        if isinstance(server, dict) and mcp_server_enabled(server)
    ]


def connected_platforms() -> list[str]:
    """The platforms Hermes's gateway connects, through Hermes's own gateway config."""
    from gateway.config import load_gateway_config  # type: ignore[import-not-found]

    return [str(platform.value) for platform in load_gateway_config().get_connected_platforms()]


def _load_config() -> dict[str, Any]:
    from hermes_cli.config import load_config  # type: ignore[import-not-found]

    return load_config()


def bounded_names(names: Iterable[str], limit: int = MAX_NAMES) -> list[str]:
    """Names as the wire admits them: whitespace and control characters flattened, each
    clipped to :data:`MAX_NAME_CHARS`, empties dropped, de-duplicated, sorted, at most
    ``limit`` of them."""
    out: set[str] = set()
    for name in names:
        flat = "".join(ch if ch.isprintable() else " " for ch in str(name))
        flat = " ".join(flat.split())[:MAX_NAME_CHARS].strip()
        if flat:
            out.add(flat)
    return sorted(out)[:limit]


def read_drift(
    own_platform: str,
    *,
    load_config: Callable[[], dict[str, Any]] = _load_config,
    mcp_servers: Callable[[dict[str, Any]], list[str]] = enabled_mcp_servers,
    platforms: Callable[[], list[str]] = connected_platforms,
) -> dict[str, list[str]]:
    """The hello's ``drift``: each kind Hermes's configuration let the plugin read.

    Never raises. A kind whose source fails is left out and logged once by its error's
    class; ``own_platform`` (Uffici's own) is never a channel of drift.
    """
    drift: dict[str, list[str]] = {}

    def kind(name: str, read: Callable[[], list[str]], limit: int = MAX_NAMES) -> None:
        try:
            drift[name] = bounded_names(read(), limit)
        except Exception as exc:  # noqa: BLE001 — a source that cannot be read says so, once
            logger.info("uffici: drift kind %s could not be read (%s)", name, type(exc).__name__)

    try:
        config = load_config()
    except Exception as exc:  # noqa: BLE001
        logger.info("uffici: Hermes's configuration could not be read (%s)", type(exc).__name__)
        config = None
    if isinstance(config, dict):
        kind("mcpServers", lambda: mcp_servers(config))
        kind("localMemory", lambda: local_memory_keys(config), MAX_LOCAL_MEMORY)
    kind("channels", lambda: [name for name in platforms() if name != own_platform])

    def count(key: str) -> int | None:
        """How many names a kind holds; ``None`` when it could not be read."""
        return len(drift[key]) if key in drift else None

    # Counts only: a name may be logged (`logging.md`), but the doctor is where they are read.
    logger.info(
        "uffici: harness drift read",
        extra={
            "driftMcpServers": count("mcpServers"),
            "driftChannels": count("channels"),
            "driftLocalMemory": count("localMemory"),
        },
    )
    return drift
