"""The Uffici platform adapter for the Hermes agent harness.

Hermes loads this package through the ``hermes_agent.plugins`` entry point and
calls :func:`register`, which registers the ``uffici`` platform. The same directory is
the ``uffici-for-hermes`` memory provider (``memory.provider``): Hermes's memory loader
finds it by the words ``MemoryProvider`` in this file and calls :func:`register` again.

🔴 ``register`` is imported LAZILY, in the function, and never at module scope.
This module is imported by the platform's own checks and by tests that have no
Hermes to import; ``adapter`` imports ``gateway.*``, which exists only inside
Hermes's virtual environment. A module-level import here would make every one of
those fail with an ImportError about a package they were never meant to have.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version

#: The plugin name Hermes loads this package under.
PLUGIN_NAME = "uffici-for-hermes"

#: Reported when the package is imported from a source checkout rather than an
#: installed distribution — which is how a developer runs it.
DEV_VERSION = "0.0.0+dev"

#: The platform's name in `config.yaml`, in Hermes's registry and on the wire.
PLATFORM_NAME = "uffici"

__all__ = ["DEV_VERSION", "PLATFORM_NAME", "PLUGIN_NAME", "register", "version"]


def register(ctx: object) -> None:
    """The plugin entry point Hermes calls — at startup for the platform, and again from
    its memory loader, which hands a ``MemoryProvider`` collector instead of a plugin
    context. That second call registers ``UfficiMemoryProvider`` and nothing else: the
    platform is already registered, from the copy of this package the gateway loaded.
    """
    from .memory import build_provider, is_memory_collector

    if is_memory_collector(ctx):
        ctx.register_memory_provider(build_provider())  # type: ignore[attr-defined]
        return
    from .adapter import register as _register

    _register(ctx)


def version() -> str:
    """Return the installed distribution version, or :data:`DEV_VERSION`."""
    try:
        return distribution_version(PLUGIN_NAME)
    except PackageNotFoundError:
        return DEV_VERSION
