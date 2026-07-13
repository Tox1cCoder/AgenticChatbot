---
name: echo-python
description: Echoes a message back as JSON to prove the python_script skill runtime executes end to end.
---

# Echo Python

A minimal, provider-neutral fixture skill used to exercise the generic skill
runtime's `python_script` execution path.

Call the `echo` capability with a `message` string argument. The skill runs a
small stdlib-only Python script that prints the message back as a JSON object.
It requires no secrets, no network access, and no third-party dependencies.
