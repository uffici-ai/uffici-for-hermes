# The pinned Hermes, for the local loop.
#
# 🔴 No plugin is baked in. The plugin directory is BIND-MOUNTED read-only, so
# an edit is live on the next gateway restart and the image never carries a
# version of the plugin that has drifted from the working tree. An image with
# the plugin inside would have to be rebuilt to test a one-line change, and the
# first person in a hurry would test the old one.
#
# The tag is Hermes 0.23.0 (commit 2bd378ed, 2026-11-05) — moved from 0.21.5
# (v2026.9.24, f97608f1) on 2026-11-08, delta read that day (#3124):
# BasePlatformAdapter and MessageEvent additive (reply_expected, edit_revision,
# media_filenames; the abstract set and every overridden hook's signature
# unchanged; ExecApprovalPrompt gains request_id); ProcessingOutcome, Platform
# and build_session_key unchanged; MemoryProvider's four abstract members
# unchanged; clarify_gateway adds get_pending, a CANCELLED sentinel and a
# "skip" answer, and its tool takes a batch of questions. What moved under the
# plugin: every inbound message is admitted to a durable FIFO in state.db,
# which refuses an event carrying adapter metadata (the brief carries none,
# its request rides in message_id), runs a brief admitted while busy without
# the processing hooks (a close scheduled at admission waits for the queue),
# replays queued input after a restart and pauses a conversation whose turn
# was cut (the plugin discards the lost turn and never hands a held brief over
# twice), and posts its own queue notices through send (never posted); the
# runtime is Python 3.14 with PyYAML gone; plugins.isolation must stay
# in_process. dev/hermes_contract.py pins each of those. Moving it is a
# decision: the adapter reads Hermes's own BasePlatformAdapter, MessageEvent,
# session authority and platform registry.
FROM nousresearch/hermes-agent:v0.23.0

# 🔴 The gateway is NOT this image's default command; `hermes` alone starts the
# interactive agent. Stated here so the compose file does not have to know it.
CMD ["gateway", "run"]
