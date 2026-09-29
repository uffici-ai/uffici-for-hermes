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

Two more workflows run beside it:

- [.github/workflows/lint-docs.yml](.github/workflows/lint-docs.yml) — Markdown lint and spelling against [.markdownlint-cli2.jsonc](.markdownlint-cli2.jsonc) and [cspell.json](cspell.json), with Node and the two tools installed by exact version in the job, so the check above stays uv only.
- [.github/workflows/e2e.yml](.github/workflows/e2e.yml) — a real Hermes, running this plugin, answers a scripted conversation against a released Uffici companion: its published image by version, and the script that release ships. `dev/e2e.py` is the whole run and needs Docker; its verdict is pinned by `tests/test_e2e_runner.py`. It runs where the companion's release can be read without a credential.

## Layout

- `src/uffici_for_hermes/` — the plugin package Hermes loads; `wire/v1/` inside it is a byte-identical copy of the companion's local-wire contract, held equal at the source.
- `uffici-plugin.json` — the install manifest `uffici setup` acts on, at the top of every release tarball; `version` is stamped by `.github/workflows/release.yml` from the tag.
- `tests/` — the test suite, including the Markdown link guard and the manifest's pin to the package, run against a stand-in transcribed from the pinned Hermes source.
- `dev/` — a container recipe for the pinned Hermes and a deterministic model stand-in, for exercising the plugin end to end without an account or a provider.

## Dependencies

Confined to what Hermes itself ships (`websockets`, `httpx`). The plugin runs
inside Hermes's sealed virtual environment, so a dependency it does not ship is
unavailable at runtime however well it resolves here.

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)).
