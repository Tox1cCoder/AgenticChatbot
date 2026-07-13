---
name: binary-probe
description: Invokes an installed command directly to prove the binary skill runtime executes end to end.
---

# Binary Probe

A minimal, provider-neutral fixture skill used to exercise the generic skill
runtime's `binary` execution path.

Call the `probe` capability. The skill invokes an already-installed command
on `PATH` directly, with no shell and no wrapper script, and returns its
stdout. It requires no secrets and no network access.
