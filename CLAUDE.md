# CLAUDE.md — Uffici for Hermes

The Uffici plugin for the Hermes agent harness: a translator between Hermes's
native hooks and the Uffici companion, with no network code, no credential and no
schema of its own. [README.md](README.md) is the document to read first: it says
what the plugin does, which Hermes it targets, and how to install and configure it.

## This repository receives releases; it is not where the work happens

Uffici.AI develops the plugin in its own platform repository and publishes each
release here as a single squashed commit, tagged — no history, nothing
force-pushed. There are no contribution rules, review workflows or plans here, and
a pull request opened here has no reader. To report a problem or ask for a change,
contact Uffici.AI through your account or [uffici.ai](https://uffici.ai).

## Check

One entry, in this project's own toolchain. It is what CI runs here:

```sh
uv run --frozen ruff check . && uv run --frozen pytest
```

Requires [uv](https://docs.astral.sh/uv/) and the Python pinned in
[pyproject.toml](pyproject.toml). The Markdown link guard runs inside `pytest` —
this project depends on no other toolchain, by design.

## Layout

- `src/uffici_for_hermes/` — the plugin package Hermes loads; `wire/v1/` inside it is a byte-identical copy of the companion's local-wire contract, held equal at the source.
- `tests/` — the test suite, including the Markdown link guard, run against a stand-in transcribed from the pinned Hermes source.
- `dev/` — a container recipe for the pinned Hermes and a deterministic model stand-in, for exercising the plugin end to end without an account or a provider.

## Dependencies

Confined to what Hermes itself ships (`websockets`, `httpx`). The plugin runs
inside Hermes's sealed virtual environment, so a dependency it does not ship is
unavailable at runtime however well it resolves here.

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).
