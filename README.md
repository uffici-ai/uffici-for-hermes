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
5. The agent's Uffici tools — its boards, tables, files and connected apps, as your
   organization has allowed them — appear in Hermes as Hermes tools. When the agent uses one,
   the call goes through the companion to Uffici, on the turn it was made in, and its answer
   comes back to the agent.
6. The agent's memory is your organization's Uffici memory, not Hermes's. Before each turn the
   plugin recalls what was remembered where the conversation sits — the organization, its
   folders, the conversation — and Hermes reads it ahead of the person's message. The agent
   can look for more, remember a fact, mark one it remembered as no longer true, and see what
   a fact replaced, all in Uffici, where people see, edit and delete what it keeps.
7. Files cross in both directions. A file a person attaches reaches Hermes as a local file,
   the way Hermes's other platforms deliver media — an image its model can see, a document
   it can read — and a file the agent sends the way Hermes always does (a `MEDIA:` path in
   its reply) arrives in Uffici as an attachment on the agent's reply. See [Files](#files).

## Requirements

- A Hermes you run as a **gateway** (`hermes gateway`), since the plugin is a gateway
  platform. The plugin is written against **Hermes 0.23.0** (commit `2bd378ed`, release
  `v0.23.0`) and supports `>=0.23.0,<0.24`: its adapter uses those versions' platform
  interfaces. A Hermes outside the range still connects and answers, but may misbehave without
  an error, so the plugin logs a warning when it connects and `uffici doctor` shows the
  harness as not supported.
- Python 3.14, which is what Hermes's own environment provides. The plugin runs inside that
  environment and depends only on libraries Hermes already ships (`websockets`, `httpx`).
- Hermes's `plugins.isolation` left at its default, `in_process`. The plugin registers a
  gateway platform and a memory provider, and Hermes refuses both to a plugin it runs in a
  separate host process.
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
notices off (see [Hermes's own notices](#hermess-own-notices)) with `hermes config set`, makes
the files workspace (`uffici-workspace` in the Hermes home) a directory Hermes delivers files
from (see [Files](#files)), writes the companion's socket and token file and the workspace's
path into Hermes's environment file, and enables the plugin by name. Then restart the Hermes
gateway.

To upgrade later, run `uffici upgrade` on the same machine: it moves the companion and this
plugin to their latest releases and sets Hermes's configuration again. When Hermes runs as its
own service (`hermes gateway install`, for Hermes's default home), it restarts it with
`hermes gateway restart`. Otherwise it tells you to restart the gateway, because for a gateway
started by hand that command would start a new one in the foreground.

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
hermes config set memory.provider uffici-for-hermes
hermes config set memory.memory_enabled false
hermes config set memory.user_profile_enabled false
hermes config set gateway.media_delivery_allow_dirs '["${UFFICI_HERMES_MEDIA_DIR}"]'
```

Then tell the plugin where the companion is. On a laptop or server, the companion's Unix
socket and its token file, both under the companion's `$UFFICI_HOME` (`~/.uffici` by
default), written into the environment file Hermes reads:

```sh
cat >> "$(hermes config env-path)" <<EOF
UFFICI_WIRE_SOCKET=$HOME/.uffici/wire.sock
UFFICI_WIRE_TOKEN_FILE=$HOME/.uffici/wire-token
UFFICI_HERMES_MEDIA_DIR=${HERMES_HOME:-$HOME/.hermes}/uffici-workspace
EOF
```

`UFFICI_HERMES_MEDIA_DIR` is the [files workspace](#files): give the companion the same
directory as its `UFFICI_WORKSPACE` (`uffici setup` records it for the companion on its own).

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
| `UFFICI_LOG_SHIPPING` | `off` keeps Hermes's log lines and the plugin's on this machine. Default on — see [Hermes's log lines](#hermess-log-lines). |
| `UFFICI_HERMES_MEDIA_DIR` | The files workspace, read by Hermes (not the plugin) through `gateway.media_delivery_allow_dirs`, so a file the agent saves there is one Hermes delivers. Setup writes `$HERMES_HOME/uffici-workspace`. |

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

### Hermes's log lines

The plugin hands the companion Hermes's own log lines and its own, and the companion sends them
to Uffici, where your organization's administrators and owners can read them on the agent's
Logs tab for at most 30 days. It is how a machine is diagnosed without asking you for a log.

- **What is captured**: every record Hermes and the plugin emit through Python `logging`, by one
  handler on the root logger. **Hermes's lines at Warning and above** by default, and the
  plugin's own at Info and above. Output printed without `logging`, and the WebSocket
  library's frame dumps, are not captured.
- **Hermes still logs locally exactly as before**: its own files and levels are untouched, and
  the plugin never changes the root logger's level. So an administrator raising the level to
  Info or Debug in Uffici has **no effect below Hermes's own configured level**
  (`logging.level` in `config.yaml`, Info by default).
- **From the moment Hermes loads the plugin**, so Hermes's own startup warnings are captured
  too, not only what follows the connection. They are held on this machine until the plugin
  connects to the companion; a Hermes that never connects to Uffici sends none of them.
- **Nothing waits on it.** A line is only appended to a bounded queue — 5,000 lines, 2 MiB,
  the oldest dropped first — and handed over every ten seconds, only while no platform call is
  in flight. The companion redacts every line before it leaves the machine, and Uffici redacts
  it again.
- **A stop hands over what is left.** When Hermes stops or restarts, the plugin pushes what it
  has queued once more before it disconnects — the lines that say why it stopped — and gives
  that at most two seconds, so a restart is never held up by its logs.
- **Harness lines can quote a conversation.** Hermes's warnings can include what a person or a
  tool said, which is why only Warning and above leave by default.

To keep them on this machine, set `UFFICI_LOG_SHIPPING=off` in Hermes's environment — or switch
it off for the whole machine with `uffici setup --log-shipping=off`.

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
- **The agent is told what the organization tells it, on every turn.** Each brief carries
  the context Uffici composes for that conversation: who the agent is and that it is an AI,
  then what the organization and the folders the conversation sits in have written for every
  agent there. The plugin places it on Hermes's system side, never in the conversation's
  history, on every turn — so an edit in Uffici applies to the agent's next answer. When part
  of it could not be included, the agent is told so and tells the person. With an older
  companion no context arrives, and Hermes runs on the plugin's fixed platform note alone.
- **A long conversation opens with its summary.** When Hermes starts a fresh session for a
  conversation that has grown long, Uffici sends a summary of the earlier messages ahead of
  the recent ones. The plugin passes it on exactly as Uffici wrote it — a first line saying
  what it covers and how the agent can read the messages behind it, then the summary — as
  part of the conversation, never on Hermes's system side. The agent reads the earlier
  messages themselves with its Uffici conversation tools.
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
- **The agent can act in Uffici with the tools your organization gave it.** The plugin asks
  Uffici for the agent's tools when it connects and when a turn starts (at most once a minute)
  and offers them to Hermes in a toolset named `uffici`, with the names, descriptions and
  parameters Uffici gives them; a tool your organization takes away stops being offered. Each
  call runs on the turn of the conversation it was made in, is checked and recorded by Uffici,
  and is shown with the answer that relied on it. A tool that needs a person's approval says
  so to the agent, and the first call asks that person in the conversation. Outside a Uffici
  turn — from another chat platform this Hermes serves, or a scheduled job — a Uffici tool
  does nothing and tells the agent why.
- **A call can be run again.** While the companion carries it, the plugin also offers one tool
  of its own in the `uffici` toolset, `uffici_rerun`: given a call's `tool_call` URN — one the
  agent sees in a summary, a search result or a trace — Uffici runs that call again, now, as a
  new call with the arguments it had, and a call that needs approval asks again. A call whose
  arguments were not kept cannot be re-run, and the agent is told so.
- **Hermes's tool search applies to them too.** Hermes keeps plugin tools behind its own tool
  search by default, so the agent finds a Uffici tool there rather than in its first list.
  The plugin writes no toolset setting: Hermes enables a plugin's toolset on its own, and
  `hermes tools` is where you can switch the `uffici` toolset off for a platform.
- **Hermes's own tools stay local.** What Hermes does with its local tools — its terminal,
  its files, a web search — runs on this machine. The answer the agent posts names which of
  those tools it used, never what it sent them or what they returned.

### Memory

The agent remembers in Uffici, where your organization can see, correct and delete what it
keeps — never in files on this machine.

- **Recall comes first, on every turn.** Before Hermes calls its model, the plugin asks Uffici
  for what was remembered where the conversation sits, using the turn's own words, and Hermes
  places it beside the person's message as recalled memory. Each line names its level — the
  conversation, a folder, the organization — and the fact's URN; when two levels say the same
  thing, the closest wins. It never holds a turn up for more than a few seconds. When memory cannot be reached — Uffici
  is down, the companion is not connected, or the companion is older than this plugin — the
  agent reads a sentence saying so and tells the person it could not check, rather than acting
  as if nothing was remembered.
- **Four tools, in the `uffici` toolset**, offered while the companion carries memory:
  `uffici_recall` looks for something else than the turn's words; `uffici_remember` records
  ONE fact about the work in this conversation — never an appraisal, rating or judgement of a
  person; `uffici_forget` marks a fact the agent remembered in this conversation as no longer
  true (it is never deleted, and facts from other levels or written by people cannot be
  forgotten by the agent); `uffici_unfold` shows what a fact or a context statement replaced.
  Outside a Uffici turn they do nothing and tell the agent why.
- **People decide what lasts.** A remembered fact stays on its conversation until a person
  promotes it to a folder or the organization, edits it, or deletes it, in the conversation's
  Memory in the Uffici portal.
- **Hermes's own memory is off — for this whole Hermes.** Setup selects the plugin as Hermes's
  memory provider (`memory.provider`) and switches off Hermes's `MEMORY.md` and `USER.md`
  (`memory.memory_enabled`, `memory.user_profile_enabled`). These are Hermes-wide settings,
  not per platform: if this Hermes also serves another chat platform, its turns there recall
  nothing and keep nothing. The plugin logs a warning at every connect when another provider
  is selected or Hermes's own memory is back on; run setup again to restore them.
- **`uffici doctor` shows what reaches past Uffici.** At every connect the plugin tells the
  companion, by name only, the other MCP servers this Hermes would start, the other platforms
  its gateway connects, and the memory settings above that no longer hold — read through
  Hermes's own configuration readers. The doctor prints them as `drift` lines. They are a
  report, never a block: nothing about them stops a connection or a turn.
- **`uffici doctor` shows whether this Hermes is one the plugin supports.** At every connect
  the plugin also sends the Hermes versions it supports and its verdict on the running one;
  the doctor prints them on its `harness` line. An unsupported Hermes still connects and
  answers, and the plugin logs a warning when it connects.

### Files

The companion moves every file, in both directions, between your organization's Uffici
files and one directory on this machine: the **workspace**, `uffici-workspace` inside the
Hermes home. A transfer cut halfway resumes where it stopped, and no file ever passes
through the model, a tool's answer or a log line — only paths, sizes and references do.

- **A person's file arrives as a local file.** The plugin asks the companion to download a
  turn's attachments, up to 25 MiB per turn, before Hermes sees the message. Each one reaches
  Hermes the way its other platforms deliver media: an image (JPEG, PNG, WebP, GIF) the model
  can see, and any other file as a document Hermes tells the agent to open with its own tools.
  The message still names every attachment, with where it is on this machine. One that was
  not downloaded — over the budget, or from earlier in the conversation — is named with the
  reason, and the agent fetches it with `uffici_download` when it needs it.
- **A file the agent makes arrives as an attachment.** When the agent puts a `MEDIA:` path in
  its reply, as it would on any Hermes platform, the plugin uploads the file and the agent's
  reply in Uffici carries it as an attachment. **Only a file inside the workspace is ever
  sent**, and nothing is ever copied into it: a file anywhere else — including one Hermes's
  own delivery policy allows, and a link in the workspace that leads out of it — is not
  attached, and the reply says the file is not in the shared workspace. The agent saves what
  it wants to send in the workspace first. Files Hermes's own tools write into Hermes's caches
  (a generated image, for instance) are outside it too, and Hermes has no setting that moves
  those caches; the agent saves a copy in the workspace. When a file cannot be attached —
  outside the workspace, refused by Uffici, missing, too large — the reply says so, naming
  the file and why: it never disappears.
- **Two tools, in the `uffici` toolset**, while the companion carries files and your
  organization lets the agent read files (`uffici_download`) or write them (`uffici_upload`):
  `uffici_download` brings an attachment (by its id) or a file in Uffici Files (by its path)
  into the workspace and answers where it landed; `uffici_upload` takes a file from the
  workspace and attaches it to the agent's reply, or saves it in Uffici Files at a path the
  agent chooses — replacing a file there asks a person first. Their paths on this side are
  relative to the workspace. Uffici checks the agent's file permissions on every transfer.
- **The workspace is a directory Hermes delivers files from — for this whole Hermes.** Setup
  adds it to `gateway.media_delivery_allow_dirs`, so even Hermes's strict delivery policy
  sends a file the agent saved there. That setting is Hermes-wide and setup replaces its list:
  add your own directories after the workspace's entry.

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
  --certificate-identity-regexp '^https://github\.com/uffici-ai/uffici-for-hermes/\.github/workflows/release\.yml@refs/tags/v' \
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
