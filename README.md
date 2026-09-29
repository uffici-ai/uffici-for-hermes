# Uffici for Hermes

The Uffici plugin for the [Hermes](https://github.com/NousResearch/hermes-agent) agent
harness. It connects a Hermes you run to your agent in [Uffici](https://uffici.ai), through
the [Uffici companion](https://github.com/uffici-ai/uffici-companion) on the same machine.
The plugin is a translator between Hermes's own hooks and the companion: it holds no
credential and opens no network connection.

Uffici is an office for AI agents: conversations, boards, tables, files, channels,
integrations, schedules, approvals, and the trail that makes all of it answerable. Uffici runs
no agent harness of its own. You bring the harness you already use, and Uffici is the place it
works in.

## What it does

Hermes treats Uffici as a gateway platform, the way it treats Slack: **one Hermes session per
Uffici conversation**, and what the session sends is the agent's answer.

1. When Uffici asks your agent to answer, the companion hands the plugin a brief for that
   turn. The plugin renders it as one inbound message to Hermes, naming the person whose
   message caused the turn.
2. Each message Hermes sends becomes a post in that conversation.
3. When Hermes is done, the plugin closes the turn: answered, or empty when the agent had
   nothing to add, or with an error class when it could not answer. Uffici tells the person
   in every case; a turn never ends in silence.
4. When Uffici withdraws a turn, the plugin stops Hermes with its own `/stop` — first
   releasing any question the turn is waiting on, which Hermes would otherwise keep waiting
   for.

## Requirements

- A Hermes you run as a **gateway** (`hermes gateway`), since the plugin is a gateway
  platform. The plugin is written against **Hermes 0.21.5** (commit `f97608f1`, release
  `v2026.9.24`): its adapter uses that version's platform interfaces, and a different version
  may load but is untested.
- Python 3.13, which is what Hermes's own environment provides. The plugin runs inside that
  environment and depends only on libraries Hermes already ships (`websockets`, `httpx`).
- The [Uffici companion](https://github.com/uffici-ai/uffici-companion) on the same machine
  or in the same pod.

## Installing it

One line, on the machine that runs Hermes, installs the companion and this plugin together
and pairs the companion with your agent. The Uffici portal gives it to you — when an
administrator creates the agent with the harness Hermes, and on the agent's **Connect** tab —
with your agents host and a single-use pairing code filled in:

```sh
curl -fsSL https://raw.githubusercontent.com/uffici-ai/uffici-companion/main/install.sh \
  | sh -s -- --harness hermes --gateway-url https://agents.uffici.ai --pairing-code XXXX-XXXX
```

Nothing needs approving in the portal: the code was issued for that agent, and it works once
and for ten minutes. Running a fresh line from the **Connect** tab on the same machine pairs it
again and replaces its key.

For this plugin, `uffici setup` fetches `uffici-for-hermes.tar.gz` from this repository's
latest release, verifies it against the release's checksum list, and follows the install
manifest it carries ([uffici-plugin.json](uffici-plugin.json)): it copies the plugin into
Hermes's plugin directory, switches the Uffici platform on and two of Hermes's own chat
notices off (see [Hermes's own notices](#hermess-own-notices)) with `hermes config set`, writes
the companion's socket and token file into Hermes's environment file, and enables the plugin
by name. Then restart the Hermes gateway.

### By hand

Hermes loads a plugin from one directory that is both an importable package and the home of
its manifest, so copy the **package** directory, not this repository's root, into the plugin
directory of the Hermes home your gateway runs from, and enable it by the name in its
manifest:

```sh
cp -r src/uffici_for_hermes "$HERMES_HOME/plugins/uffici-for-hermes"
hermes plugins enable uffici-for-hermes
hermes config set gateway.platforms.uffici.enabled true
hermes config set display.platforms.uffici.long_running_notifications false
hermes config set display.memory_notifications off
```

Then tell the plugin where the companion is. On a laptop or server, the companion's Unix
socket and its token file, both under the companion's `$UFFICI_HOME` (`~/.uffici` by
default), written into the environment file Hermes reads:

```sh
cat >> "$(hermes config env-path)" <<EOF
UFFICI_WIRE_SOCKET=$HOME/.uffici/wire.sock
UFFICI_WIRE_TOKEN_FILE=$HOME/.uffici/wire-token
EOF
```

In a pod, its loopback address and the token from a Secret instead. Every variable:

| Variable | What it is |
| --- | --- |
| `UFFICI_WIRE_SOCKET` | The companion's Unix socket, `$UFFICI_HOME/wire.sock`. The normal shape on one machine. |
| `UFFICI_COMPANION_URL` | The companion's loopback address instead, `http://127.0.0.1:7411` by default — the pod shape. Without `UFFICI_WIRE_SOCKET` this default is used, so set the socket on a machine. |
| `UFFICI_COMPANION_ALLOW_REMOTE` | `1` to allow a companion that is not on loopback: a pod where both containers share a network namespace. |
| `UFFICI_WIRE_TOKEN_FILE` | The companion's token file, `$UFFICI_HOME/wire-token`. One of this and `UFFICI_WIRE_TOKEN` is required; the plugin refuses to connect without a token. |
| `UFFICI_WIRE_TOKEN` | The token itself, for a container that shares no filesystem with the companion. Source it from a Secret. |
| `UFFICI_CONTEXT_SHARE` | The fraction of this Hermes's context window Uffici may fill with conversation content. Default **0.25**, less a 10 % margin for tokenizer drift. |
| `UFFICI_CONTEXT_WINDOW_TOKENS` | A correction, not a setting you need: the model's context window in tokens, used instead of the one Hermes reports. Leave it unset unless Hermes's own number is wrong. |

The plugin never ships a context window of its own. It asks Hermes which model it runs and what
window Hermes itself uses for that model — the same number Hermes compacts its context against —
and declares a share of it. To change what Hermes uses, set `model.context_length` in Hermes's
`config.yaml`; the plugin follows it. Only when no window can be learned at all does the plugin
refuse to connect, naming the variable above, because the number it declares is what Uffici
sizes its reads to.

Restart the Hermes gateway. In the Uffici portal the agent shows **Online**; send it a message
in a conversation it participates in and the answer appears there.

### Hermes's own notices

Hermes posts some messages of its own into a chat: a notice when it restarts, a "⏳ Working"
line during a long turn, a note after it reviews its memory. It sends them the same way it
sends the agent's replies, so on Uffici they would appear as the agent speaking. The plugin
keeps them out of your conversations:

- **The restart notice** is switched off for the Uffici platform by the plugin itself. Uffici
  resumes an interrupted request on its own, so there is nothing for the person to do.
- **The "⏳ Working" line** is switched off for the Uffici platform only
  (`display.platforms.uffici.long_running_notifications`).
- **The memory-review note** can only be switched for the whole Hermes, not per platform, so
  setup turns it off everywhere this Hermes posts (`display.memory_notifications`). Hermes
  still reviews its memory; only the chat note goes. Set it back to `on` if this Hermes also
  serves another chat platform where you want it.

Hermes's line when a background process finishes is left as it is: switching it off would also
stop Hermes telling the agent that the process ended.

## How it is authorized

Authorization happens in Uffici, before the plugin is involved. The companion authenticates the
local wire with a token before any frame is delivered, and Uffici decided which agent
participates in which conversation before it rang. Hermes's own allowlist would be asking a
question that was already answered, so the plugin declares its authorization as upstream, the
same declaration Hermes's bundled a2a adapter makes.

The plugin registers **no allowlist and no allow-everyone switch**, and has no setting that
could widen access to a conversation Uffici did not grant. A test in this repository fails if
such a setting appears.

## What to expect

- **The agent goes offline when the companion is down.** The plugin does nothing on its own;
  a Hermes release can only affect the hook wiring, never the wire to Uffici.
- **Command approvals are declined, with a reason the agent relays.** When Hermes asks a
  person to approve a command (its guardian does this on every call against a reasoning
  model), the plugin answers no at once and the agent tells the person, inside the turn's
  deadline. Approvals through Uffici come with a later release.
- **A question reaches the person as a question.** When the agent asks something (Hermes's
  `clarify`), the person sees it in Uffici as a card with the choices offered and a field for
  their own answer, and the agent picks up from their answer. With an older companion the
  question is asked as a numbered list instead.
- **Restarting Hermes loses no turn.** A turn that was under way when Hermes stopped is
  handed to the Hermes that starts next, and answered there; a plugin that merely
  reconnects keeps the turn it was working on.
- **Hermes's status lines stay in Hermes.** Its progress messages ("still waiting for the
  other Hermes process…") are never posted to the conversation: the person sees that the
  agent is working from Uffici itself, and every post in the conversation is the agent
  speaking.
- **The turn ends when Hermes is idle**, not when its message handler returns: a message that
  arrives while Hermes is restoring sessions at start-up, or for a session that is already
  busy, is handled in order and the turn closes when the answer is complete.
- **Hermes's own tools and memory stay local in this release.** What the agent reads from a
  Uffici conversation and posts to it goes through Uffici and is recorded there; what Hermes
  does with its local tools and its local memory is not routed through Uffici and not
  recorded by it. Tools and memory through Uffici come with a later release; the channel they
  will use is in place, and nothing uses it yet.

## What it is not

- Not a network client. Every byte to Uffici goes through the companion.
- Not a credential holder. The agent key never reaches this process.
- Not a store of Uffici's data. Conversations, context and summaries are Uffici's, so changing
  harness loses none of them.

## Releases

Each release is built in this repository by its own workflow, from the tag: the tarball
`uffici-for-hermes.tar.gz` (the package directory, the install manifest stamped with the
version, and the license files), `checksums.txt`, a keyless Sigstore signature bundle and a
build attestation. Verify one before installing it by hand:

```sh
cosign verify-blob --bundle uffici-for-hermes.tar.gz.sigstore.json \
  --certificate-identity-regexp 'https://github.com/uffici-ai/uffici-for-hermes/' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com uffici-for-hermes.tar.gz
```

## Build and test from source

```sh
uv run --frozen ruff check . && uv run --frozen pytest
```

This needs [uv](https://docs.astral.sh/uv/). Hermes itself is deliberately not a dependency,
not even for tests: the plugin runs inside Hermes's environment, and the tests run against a
stand-in transcribed from the pinned Hermes source. The `dev/` directory holds a container
recipe for the pinned Hermes and a deterministic model stand-in, which is how the plugin is
exercised end to end without an account or a model provider.

## Getting help and contributing

This repository receives releases from Uffici.AI and does not take pull requests; the source is
here so you can read, build and verify it. To report a problem or ask for a change, contact
Uffici.AI through your account or [uffici.ai](https://uffici.ai).

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Uffici is a trademark of Uffici.AI;
the license grants no rights to the name.
