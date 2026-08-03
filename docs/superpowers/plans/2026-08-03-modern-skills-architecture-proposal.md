# Modern Skills Architecture — Proposal

**Status:** proposal, awaiting direction. Nothing here is implemented.

**Goal:** make the sidecar's skill system work with skills as they are actually
written and distributed today (the Agent Skills / Claude Code convention), while
keeping the device-local execution, approval, and secret model that this system
already has and Claude Code does not.

## What "modern skills" actually look like

Measured from `superpowers-main.zip` (obra/superpowers v6.2.0, 14 skills), which
is a representative current library rather than a guess:

| Property | What the ecosystem does |
|---|---|
| Frontmatter | `name` and `description` only. The description is a *trigger*: "You MUST use this before any creative work…" |
| Body | A short router, not the whole method |
| Companion files | Every non-trivial skill ships them: `root-cause-tracing.md`, `defense-in-depth.md`, `condition-based-waiting.md`, `visual-companion.md`, example `.ts`, `find-polluter.sh` |
| Distribution | One git repo = one **collection** of many skills, with `.claude-plugin/plugin.json` (name, version, author, license, homepage) |
| Multi-harness | The same repo carries `.claude-plugin`, `.codex-plugin`, `.cursor-plugin`, `.pi`, `.opencode`, `.agents` |
| Cross-references | Skills name each other: `superpowers:brainstorming` |

The load-bearing idea is **progressive disclosure**: `SKILL.md` stays small and
says *when* to read `references/x.md`; the agent reads the rest on demand. That is
what keeps a 14-skill library from consuming the context window.

## Where this system stands

Verified against the code, not assumed:

| # | Gap | Evidence | Consequence |
|---|---|---|---|
| G1 | **No progressive disclosure.** `activate_skill` returns the entire `SKILL.md` body and nothing else can be read | `app/ai/skills_tool.py` returns `content`; no reader for bundled files exists anywhere in `app/` or `client_backend/` | A skill's companion files are dead weight — shipped, never readable. Authors must inline everything into `SKILL.md`, which is exactly what the convention avoids |
| G2 | **One archive installs one skill** | `install.py::_discover_source` requires exactly one `SKILL.md` | Every real collection is rejected. This is the error you hit |
| G3 | **No provenance, version, or update path** | `install.json` records `bundle_name`, `source_hash`, `source` | Cannot answer "is this current?" or "update my superpowers skills" |
| G4 | **Flat namespace** | `LocalSkillsRegistry.skills` is keyed by bare name | Two collections owning a `brainstorming` skill silently collide |
| G5 | **Frontmatter narrower than the ecosystem's** | `shared/skills/front_matter.py` parses `name`, `description`, `category`, `tags` | `allowed-tools` is ignored, so a skill's own tool restrictions are not honored |
| G6 | **All skill summaries go in every system prompt** | `get_available_skill_summaries` | Fine at 14 skills; a few hundred is a context problem with no search path |

Two things this system already does that Claude Code does **not**, and which the
proposal must not regress: device-scoped execution with per-skill approval and
encrypted secrets, and hash-bound atomic installation.

## Proposal

### Phase 1 — make real skill libraries usable (the unblock)

**1.1 `read_skill_resource(skill, path)`** — the keystone.

A device-scoped tool that returns one text file from inside an installed bundle.
Confinement mirrors `run_skill_command`: resolve only under that skill's bundle
root, refuse links and escapes, cap the size, text only. Without it, every
companion file in every modern skill is unreachable.

Activation changes shape with it: `activate_skill` returns the router body plus a
manifest of readable resources, and the model pulls what it needs.

**1.2 Collection install.**

Accept an archive containing many skills. Preview lists every skill found, with
per-skill existing-collision state; approval is one decision for the set;
installation is atomic across the set (all or none), recording a shared
`collection_id`. Per-skill enable/disable afterwards, as today.

This is the difference between "zip 14 folders one at a time" and "install
superpowers".

### Phase 2 — identity and lifecycle

**2.1 Namespaced identity.** `collection:skill`, with a bare name resolving when
unambiguous and erroring when not. Required before two libraries can coexist.

**2.2 Provenance and versioning.** Record collection name, version, and origin
(URL or filename) in `install.json`; surface them in the catalog; add
`GET /skills/updates` comparing installed versions against a re-fetched source.

**2.3 Update as a first-class operation.** Reuse the guarded-replacement machinery
already built: preview → hash-bound confirm → atomic swap, per skill or per
collection.

### Phase 3 — fidelity and scale

**3.1 Honor `allowed-tools`.** Parse it, and enforce it when a skill is active.
Currently a skill can declare restrictions that nothing applies.

**3.2 Forward-compatible frontmatter.** Preserve unknown keys instead of dropping
them, so a skill written for a newer convention still installs.

**3.3 Skill-to-skill activation.** Let an active skill name another
(`superpowers:brainstorming`); resolve through the same allowlist a custom agent
already has.

**3.4 Search over injection, past a threshold.** Beyond ~50 skills, inject names
and one-line descriptions only and add a lookup tool.

## Sequencing rationale

Phase 1 is what makes the system usable with libraries that exist today, and both
items are additive — no existing contract changes. Phase 2 is what makes it
maintainable once more than one library is installed; it touches the registry key,
so it wants to land before people accumulate skills. Phase 3 is fidelity work with
no user-visible blocker behind it.

## Cost and risk

| Item | Rough size | Main risk |
|---|---|---|
| 1.1 read_skill_resource | small–medium | Another file-read surface on the device; confinement must match the command runtime's |
| 1.2 Collection install | medium | Atomicity across N skills, and a preview UI that stays comprehensible at N=14 |
| 2.1 Namespacing | medium | Touches registry keys, catalog projection, resolver, and the model-facing tool names |
| 2.2/2.3 Versioning + update | medium | Re-fetching a source implies network access from the sidecar |
| 3.x | small each | Enforcement changes behavior for already-installed skills |

## Open questions

1. Should a collection install as one unit that upgrades together, or as N
   independent skills that merely share provenance?
2. Should the sidecar ever fetch from a URL (install/update from a GitHub repo
   directly), or stay upload-only? Upload-only is a meaningful part of the current
   security story.
3. Does the AI SDK frontend need the same collection UI as Streamlit in the first
   pass, or is Streamlit enough to validate the shape?
