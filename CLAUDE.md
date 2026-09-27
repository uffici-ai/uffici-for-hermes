# CLAUDE.md — Uffici for Hermes

The Uffici plugin for the Hermes agent harness: a format translator between
Hermes's native seams and the Uffici companion, with no network code, no
credential and no schema of its own. [README.md](README.md) is the design.

## This repository receives releases; it is not where the work happens

Development happens in the Uffici platform repository, in the `uffici/` tree.
A release dispatched there exports this project's subtree here as a single squashed
commit and tags it — no history, nothing force-pushed. So there are no rules, skills,
agents or review workflows here, and a pull request opened here has no reader:
file the issue and the change in the platform repository instead.

## Check

One entry, in this project's own toolchain. It is what CI runs and what the
platform's `pnpm check:uffici` delegates to:

```sh
uv run --frozen ruff check . && uv run --frozen pytest
```

Requires [uv](https://docs.astral.sh/uv/) and the Python pinned in
[pyproject.toml](pyproject.toml). The Markdown link guard runs inside `pytest` —
this project depends on no other toolchain, by design.

## Layout

- `src/uffici_for_hermes/` — the plugin package Hermes loads.
- `tests/` — the test suite, including the Markdown link guard.

## Dependencies

🔴 Confined to what Hermes itself ships (`websockets`, `httpx`). The plugin runs
inside Hermes's sealed virtual environment, so a dependency it does not ship is
unavailable at runtime however well it resolves here.

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).
