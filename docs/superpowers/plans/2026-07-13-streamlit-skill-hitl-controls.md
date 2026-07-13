# Streamlit Skill HITL Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MCP-style Inherit, Require, and Skip HITL controls for every skill command in the Streamlit Skills tab.

**Architecture:** Reuse the existing per-user `/hitl/settings` API and its tool-scoped rules. `render_skills_tab()` will translate each skill name into the existing qualified command ID `skill::<name>::run_skill_command`, render the same tri-state control used for MCP tools, and persist changes through the existing `set_hitl_setting()` and `clear_hitl_setting()` helpers.

**Tech Stack:** Python, Streamlit, FastAPI HITL settings API, pytest, Ruff

---

### Task 1: Specify the Streamlit skill HITL contract

**Files:**
- Modify: `tests/test_hitl_demo_panel.py`

- [ ] **Step 1: Write the failing static regression test**

Add a focused test that requires the Skills tab to construct the exact command ID, render the three approval modes, use a skill-specific widget key, and call the existing tool-scoped helpers:

```python
def test_demo_renders_per_skill_command_hitl_controls():
    src = _demo_source()
    assert 'f"skill::{skill_name}::run_skill_command"' in src
    assert 'key=f"hitl_skill_mode_{skill_name}"' in src
    assert 'modes = ["Inherit", "Require", "Skip"]' in src
    assert 'clear_hitl_setting("tool", skill_qualified_id)' in src
    assert (
        'set_hitl_setting("tool", skill_qualified_id, chosen == "Require")'
        in src
    )
    assert "approval rules below are inactive until it is enabled" in src
```

- [ ] **Step 2: Run the test and verify the expected failure**

Run:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_hitl_demo_panel.py::test_demo_renders_per_skill_command_hitl_controls
```

Expected: FAIL because `render_skills_tab()` does not yet construct the skill qualified ID or render `hitl_skill_mode_...`.

- [ ] **Step 3: Commit the red test**

```powershell
git add -- tests/test_hitl_demo_panel.py
git commit -m "test(hitl): specify Streamlit skill approval controls"
```

### Task 2: Render and persist per-skill approval modes

**Files:**
- Modify: `demo.py`
- Test: `tests/test_hitl_demo_panel.py`

- [ ] **Step 1: Load the policy once in the Skills tab**

After the skill list is validated, load the HITL settings and build the exact tool-rule index:

```python
hitl_settings = get_hitl_settings()
skill_tool_rules: dict[str, bool] = {}
if hitl_settings is None:
    st.warning(
        "Human approval settings are unavailable. Skill approval controls are disabled."
    )
else:
    hitl_master = bool(hitl_settings.get("masterEnabled", True))
    skill_tool_rules = {
        item["scopeValue"]: item["requireApproval"]
        for item in hitl_settings.get("tools", [])
    }
    if not hitl_master:
        st.caption(
            ":material/info: Human-in-the-loop is globally disabled (admin setting); "
            "approval rules below are inactive until it is enabled."
        )
```

- [ ] **Step 2: Add the MCP-style control to each skill card**

Inside each skill expander, render a control only when settings loaded:

```python
if hitl_settings is not None:
    skill_qualified_id = f"skill::{skill_name}::run_skill_command"
    if skill_qualified_id in skill_tool_rules:
        current_mode = "Require" if skill_tool_rules[skill_qualified_id] else "Skip"
    else:
        current_mode = "Inherit"

    st.markdown("**Human approval**")
    modes = ["Inherit", "Require", "Skip"]
    chosen = st.radio(
        "Approval mode for this skill command",
        modes,
        index=modes.index(current_mode),
        key=f"hitl_skill_mode_{skill_name}",
        horizontal=True,
        help=(
            "Inherit = use the safe mutation default; Require = always prompt; "
            "Skip = preapprove this skill command"
        ),
    )
    if chosen != current_mode:
        with st.spinner("Updating skill approval..."):
            if chosen == "Inherit":
                result = clear_hitl_setting("tool", skill_qualified_id)
            else:
                result = set_hitl_setting(
                    "tool", skill_qualified_id, chosen == "Require"
                )
            if result is not None:
                st.rerun()
            else:
                st.error(_last_api_error_message("Failed to update skill approval"))
```

- [ ] **Step 3: Run the focused Streamlit tests**

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_hitl_demo_panel.py tests/test_streamlit_width_deprecation.py
```

Expected: PASS.

- [ ] **Step 4: Run the skill/HITL policy regression tests**

```powershell
.venv\Scripts\python.exe -m pytest -q tests/client_backend/test_skill_hitl.py tests/test_hitl_policy.py tests/test_hitl_gate_policy.py tests/test_hitl_client_and_deferred.py tests/client_backend/test_hitl_proxy.py
```

Expected: PASS; exact Skip rules continue to override default mutation gating.

- [ ] **Step 5: Run static quality checks**

```powershell
.venv\Scripts\python.exe -m ruff check demo.py tests/test_hitl_demo_panel.py
git diff --check
```

Expected: no lint or whitespace errors.

- [ ] **Step 6: Commit the implementation**

```powershell
git add -- demo.py tests/test_hitl_demo_panel.py
git commit -m "feat(hitl): configure skill approvals in Streamlit"
```

### Task 3: Verify the completed feature

**Files:**
- Verify: `demo.py`
- Verify: `tests/test_hitl_demo_panel.py`

- [ ] **Step 1: Run the complete focused HITL and skill matrix**

```powershell
$skillFiles = @(Get-ChildItem tests/client_backend -File | Where-Object {
    $_.Name -like 'test_skill*.py' -or $_.Name -like 'test_skills*.py'
} | ForEach-Object FullName)
$hitlFiles = @(Get-ChildItem tests -File -Filter 'test_hitl*.py' | ForEach-Object FullName)
.venv\Scripts\python.exe -m pytest -q @skillFiles @hitlFiles
```

Expected: PASS, with only platform-dependent tests skipped.

- [ ] **Step 2: Inspect the final diff and repository state**

```powershell
git diff --check
git status --short
git log -3 --oneline
```

Expected: the feature commits contain only the design, test, Streamlit UI, and plan files; unrelated pre-existing runtime changes remain preserved.
