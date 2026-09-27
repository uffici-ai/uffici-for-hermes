# The pinned Hermes, for the local loop.
#
# 🔴 No plugin is baked in. The plugin directory is BIND-MOUNTED read-only, so
# an edit is live on the next gateway restart and the image never carries a
# version of the plugin that has drifted from the working tree. An image with
# the plugin inside would have to be rebuilt to test a one-line change, and the
# first person in a hurry would test the old one.
#
# The tag is Hermes 0.21.5 — the version this plugin's adapter is written
# against (commit f97608f1), and the latest stable release at the time it was
# pinned (2026-09-25, from 0.21.4, delta read that day: MessageEvent,
# MessageType and ProcessingOutcome unchanged; BasePlatformAdapter additive —
# rewire_plugin_handlers, _release_turn_marker (a crash-recovery marker the
# base releases after delivery; it touches none of the plugin's request state),
# a macOS proxy cache — with no abstract method added and no overridden hook's
# signature changed; Platform/PlatformConfig and build_session_key unchanged;
# the approval path the plugin answers and the busy state it reads unchanged).
# Moving it is a decision: the adapter reads Hermes's own BasePlatformAdapter,
# MessageEvent and platform registry.
FROM nousresearch/hermes-agent:v2026.9.24

# 🔴 The gateway is NOT this image's default command; `hermes` alone starts the
# interactive agent. Stated here so the compose file does not have to know it.
CMD ["gateway", "run"]
