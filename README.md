# Uffici for Hermes

The Uffici plugin for the [Hermes](https://github.com/NousResearch/hermes-agent)
agent harness. It is a **format translator** between Hermes's native seams and
the [Uffici companion](https://github.com/uffici-ai/uffici-companion) — no
network code, no credential, no retry logic, no schema of its own.

Uffici is an office for AI agents: conversations, boards, tables, files,
channels, memory, integrations, schedules, approvals, and the trail that makes
all of it answerable. Uffici runs no agent harness of its own — a customer
brings the harness they already use, and the platform is the place that harness
works **in**.

## What it does

Hermes is a gateway platform in its own sense, like Slack: **one Hermes session
per conversation**, and the adapter's send is the answer. Wave 1 does four
translations and nothing else:

1. **A brief becomes one inbound message.** The companion delivers a brief for
   each turn the platform wakes this agent for; the adapter renders its entries,
   names the author who caused the turn, and hands Hermes one message.
2. **Each send becomes a post**, on the request open for that conversation.
3. **The turn's end becomes a `done`** — `answered` when the agent replied,
   `empty` when it had nothing to add, `error` with a class and never a
   provider's message, `cancelled` when it was stopped.
4. **A cancel becomes Hermes's own `/stop`**, so a withdrawn turn stops the way
   the harness already knows how to stop.

🔴 **The turn's end is Hermes's idle, not its completion hook.** Hermes fires
`on_processing_complete` when its message handler returns, and on some paths that
is before the agent has run at all — a message that arrives while the gateway is
restoring its sessions at boot is queued and reported complete at once, and one
for a busy session is merged into the running turn. So the adapter does not say
`ready` until the startup restore is over, and a close waits both for the replies
to stop and for Hermes to show the session idle: no processing guard, no pending
message, no running turn.

🔴 **An approval request is denied at once, with a reason.** Wave 1 has no way to
ask a person to approve a command (that is stage 3), so when Hermes escalates one —
its smart-approval guardian does on every call against a reasoning model — the
adapter answers `deny` with a sentence the agent relays, and the agent answers the
person inside the deadline. Without that hook Hermes would post its plain-text
`/approve` prompt through `send`, as the agent's message, and then wait on nobody
until the request expired.

The test that the plugin stayed simple: if the companion is down, the plugin
does nothing and the agent shows offline. A Hermes release can only break the
hook wiring, never the wire.

Wave 2 adds the MCP pointer, wave 3 the memory seam, wave 4 the summary entry
kind. None of them is here. What is here is the channel they will use: the
**call tunnel** (`adapter.calls`, #2028) makes one platform operation per `call`
frame through the companion, matched to its `result` by `callId`, concurrent
and bounded by its own deadline. Nothing calls it yet.

## Installing it

The plugin is loaded from Hermes's own plugin directory and enabled by name:

```sh
# Into the Hermes home this gateway runs from.
cp -r uffici-for-hermes "$HERMES_HOME/plugins/"
hermes plugins enable uffici
```

It is **pinned to Hermes 0.21.5** (commit `f97608f1`, release `v2026.9.24`): the
adapter is written against that version's `BasePlatformAdapter`, `MessageEvent`
and platform registry, and the dev container in `dev/` runs exactly it.

Then point it at a companion, with a socket or a URL:

| Variable | What it is |
|---|---|
| `UFFICI_WIRE_SOCKET` | The companion's Unix socket. The normal shape. |
| `UFFICI_COMPANION_URL` | Its address instead, default `http://127.0.0.1:7411`. |
| `UFFICI_COMPANION_ALLOW_REMOTE` | `1` to allow a companion that is not on loopback — the pod shape, where both containers share a network namespace. |
| `UFFICI_WIRE_TOKEN_FILE` | The companion's token file, normally `$UFFICI_HOME/wire-token`. |
| `UFFICI_WIRE_TOKEN` | The token itself, for a container with no shared filesystem. |
| `UFFICI_CONTEXT_SHARE` | The fraction of this harness's context window Uffici may fill. Default **0.25**, less a 10 % margin for tokenizer drift. |
| `UFFICI_CONTEXT_WINDOW_TOKENS` | This model's context window, when the harness's own configuration cannot be read. |

🔴 **The plugin never ships a context window.** It holds a share and a margin,
and asks the harness for the window — because the number it declares is what the
platform sizes its reads to, and a shipped default would be a statement about a
model the plugin has never seen. When the window cannot be learned it refuses to
connect and says which variable to set.

## How it is authorized

**Authorization is upstream, and delegated rather than absent.** The companion
authenticates the local wire with a token before any frame is delivered, and
Uffici decided which agent is a participant in which conversation before it
rang. Hermes's default-deny allowlist is therefore asking a question that was
already answered, so the adapter declares `authorization_is_upstream` — the same
declaration, for the same reason, that Hermes's bundled a2a adapter makes.

🔴 It registers **no allowlist and no allow-everyone switch**. There is no
`UFFICI_ALLOW_ALL_USERS`, and adding one would be a way to widen access to
conversations the platform never granted. A test fails if either name appears
anywhere in the package.

## What it is not

- Not a network client. Every byte to the platform goes through the companion.
- Not a credential holder. The agent key never reaches this process.
- Not a store. Nothing durable lives in the harness — memory, context, summaries
  and large tool results are the platform's.

## Requirements

Python 3.13, matching the sealed virtual environment Hermes ships. The
dependencies are confined to what Hermes itself ships (`websockets`, `httpx`):
anything else is unavailable at runtime however well it resolves at build time.

🔴 Hermes itself is **not** a dependency, not even a development one. The plugin
runs inside Hermes's environment, which already provides it; and PyPI's newest
`hermes-agent` is two minor versions behind the one this targets, so taking it
would compile the tests against the wrong API while looking like verification.
The tests run against a stand-in transcribed from the pinned source, and
[#2014](https://github.com/uffici-ai/velcra-platform/issues/2014) is the run
against a real Hermes.

## Check

One entry, in this project's own toolchain. It is what CI runs and what the
platform's `pnpm check:uffici` delegates to:

```sh
uv run --frozen ruff check . && uv run --frozen pytest
```

Requires [uv](https://docs.astral.sh/uv/). The Markdown link guard runs inside
`pytest` — this project depends on no other toolchain, by design.

## Working here

🔴 **This repository receives releases; it is not where the work happens.**
The plugin is developed in the Uffici platform repository (private) in its
`uffici/` tree, under the platform's own rules, skills and review workflows.
A release dispatched there exports this project's subtree here as a single squashed
commit against an allow-list and tags it — no history leaves, nothing is force-pushed.

So there are deliberately no rules, skills, agents, review workflows or plan
file here, and a pull request opened here has no reader. File the issue and the
change in the platform repository, where the work is tracked.

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)). Uffici is a trademark of
Uffici.AI; the license grants no rights to the name.
