"""Static guard: the demo MCP panel manages HITL approval via the sidecar API."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests


def _demo_source() -> str:
    return Path("demo.py").read_text(encoding="utf-8")


def test_demo_defines_hitl_helpers_and_calls_settings_endpoint():
    src = _demo_source()
    assert "def get_hitl_settings(" in src
    assert "def set_hitl_setting(" in src
    assert "def clear_hitl_setting(" in src
    assert 'make_api_request("GET", "/hitl/settings")' in src
    assert 'make_api_request("POST", "/hitl/settings"' in src


def test_demo_renders_per_server_and_per_tool_controls():
    src = _demo_source()
    assert "Approval: ON" in src  # per-server toggle label
    assert "hitl_tool_mode_" in src  # per-tool tri-state widget key prefix
    assert "qualified_tool_options" in src  # duplicate tool names select by server::tool


def test_demo_renders_per_skill_command_hitl_controls():
    src = _demo_source()
    assert 'f"skill::{skill_name}::run_skill_command"' in src
    assert 'skill.get("commandCapable", False)' in src
    assert 'widget_key = f"hitl_skill_mode_{skill_hitl_scope}_{skill_name}"' in src
    assert 'modes = ["Inherit", "Require", "Skip"]' in src
    assert 'clear_hitl_setting("tool", skill_qualified_id)' in src
    assert 'set_hitl_setting("tool", skill_qualified_id, chosen == "Require")' in src
    assert "approval rules below are inactive until it is enabled" in src


def test_demo_skill_hitl_state_is_callback_driven_and_cleared_on_logout():
    src = _demo_source()
    assert "def _persist_skill_hitl_mode(" in src
    assert "on_change=_persist_skill_hitl_mode" in src
    assert "st.session_state[widget_key] = current_mode" in src
    assert "def _clear_skill_hitl_session_state(" in src
    assert "_clear_skill_hitl_session_state()" in src


def test_demo_uses_shared_hitl_decision_builder():
    src = _demo_source()
    assert "from app.ui.hitl_decisions import" in src
    assert "build_interrupt_decision(" in src
    assert "interrupt_request_target_ids(" in src


def test_demo_has_no_replay_terminal_approval_recovery_seams():
    src = _demo_source()
    for helper in (
        "extract_error_code",
        "is_recoverable_resume_conflict",
        "reconciliation_action",
        "should_suppress_pending_interrupt",
    ):
        assert helper in src

    tree = ast.parse(src)
    lifecycle_helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_hitl_interrupt_state"
    )
    lifecycle_call = next(
        node
        for node in ast.walk(lifecycle_helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "make_api_request"
    )
    assert ast.unparse(lifecycle_call) == (
        "make_api_request('GET', f'/hitl/interrupts/{interrupt_id}', use_cache=False)"
    )

    assert "hitl_resume_inflight_" in src
    assert '"failed"' in src
    assert '"status_code"' in src
    assert '"error_code"' in src


def test_streaming_http_error_keeps_status_and_canonical_code(monkeypatch):
    import demo

    response = requests.Response()
    response.status_code = 409
    response._content = b'{"code":"INTERRUPT_CONFLICT","message":"Approval claimed elsewhere."}'

    def raise_http_error(*_args, **_kwargs):
        raise requests.exceptions.HTTPError(response=response)

    monkeypatch.setattr(
        demo,
        "st",
        SimpleNamespace(session_state={"auth_token": None}, toast=lambda *_args, **_kwargs: None),
    )
    monkeypatch.setattr(
        demo,
        "get_http_session",
        lambda: SimpleNamespace(post=raise_http_error),
    )

    assert list(demo.make_streaming_request("/messages/resume-interrupt")) == [
        {
            "type": "error",
            "error": "Approval claimed elsewhere.",
            "status_code": 409,
            "error_code": "INTERRUPT_CONFLICT",
        }
    ]


def test_terminal_resume_error_reruns_after_clearing_the_approval(monkeypatch):
    import demo

    class SessionState(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

    class Status:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def update(self, **_kwargs):
            pass

    session_state = SessionState(
        pending_interrupt={"interrupt_id": "interrupt-1"},
        interrupt_conversation_id="conversation-1",
        pending_decisions_interrupt_1={"tool-1": {"type": "approve"}},
    )
    session_state[demo._hitl_resume_lock_key("interrupt-1")] = True
    errors: list[str] = []
    reruns: list[bool] = []
    monkeypatch.setattr(
        demo,
        "st",
        SimpleNamespace(
            session_state=session_state,
            status=lambda *_args, **_kwargs: Status(),
            empty=lambda: object(),
            error=errors.append,
            rerun=lambda: reruns.append(True),
        ),
    )
    monkeypatch.setattr(demo, "_StreamingRichResponseRenderer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(demo, "_reset_stream_trace_state", lambda **_kwargs: None)
    monkeypatch.setattr(demo, "_clear_inflight_state", lambda: None)
    monkeypatch.setattr(
        demo,
        "make_streaming_request",
        lambda *_args, **_kwargs: iter(
            [{"type": "error", "error": "Continuation failed", "error_code": "INTERRUPT_FAILED"}]
        ),
    )

    demo._submit_interrupt_decisions(
        "thread-1",
        "interrupt-1",
        [{"task_id": "tool-1"}],
        "pending_decisions_interrupt_1",
    )

    assert errors == ["Continuation failed"]
    assert reruns == [True]
    assert "pending_interrupt" not in session_state
    assert session_state[demo._hitl_reconciliation_key()] == "interrupt-1"
    assert session_state["hitl_reconciliation_notice"] == "Continuation failed"
    assert demo._hitl_resume_lock_key("interrupt-1") not in session_state


def test_terminal_reconciliation_notice_is_rendered_after_the_rerun():
    src = _demo_source()
    assert 'st.session_state["hitl_reconciliation_notice"] = resume_error' in src
    assert 'reconciliation_notice = st.session_state.pop("hitl_reconciliation_notice", None)' in src
    assert "st.info(reconciliation_notice)" in src


def test_nonduplicate_resume_error_does_not_reconcile_or_replay(monkeypatch):
    import demo

    class SessionState(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

    class Status:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def update(self, **_kwargs):
            pass

    session_state = SessionState(
        pending_interrupt={"interrupt_id": "interrupt-1"},
        interrupt_conversation_id="conversation-1",
        pending_decisions_interrupt_1={"tool-1": {"type": "approve"}},
    )
    session_state[demo._hitl_resume_lock_key("interrupt-1")] = True
    errors: list[str] = []
    reconciled: list[str] = []
    monkeypatch.setattr(
        demo,
        "st",
        SimpleNamespace(
            session_state=session_state,
            status=lambda *_args, **_kwargs: Status(),
            empty=lambda: object(),
            error=errors.append,
        ),
    )
    monkeypatch.setattr(demo, "_StreamingRichResponseRenderer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(demo, "_reset_stream_trace_state", lambda **_kwargs: None)
    monkeypatch.setattr(demo, "_clear_inflight_state", lambda: None)
    monkeypatch.setattr(
        demo,
        "make_streaming_request",
        lambda *_args, **_kwargs: iter(
            [
                {
                    "type": "error",
                    "error": "Device does not match this approval.",
                    "error_code": "INTERRUPT_DEVICE_MISMATCH",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        demo,
        "_reconcile_interrupt",
        lambda interrupt_id: reconciled.append(interrupt_id) or "restore_form",
    )

    demo._submit_interrupt_decisions(
        "thread-1",
        "interrupt-1",
        [{"task_id": "tool-1"}],
        "pending_decisions_interrupt_1",
    )

    assert reconciled == []
    assert errors == ["Device does not match this approval."]
    assert session_state["pending_interrupt"] == {"interrupt_id": "interrupt-1"}
    assert demo._hitl_resume_lock_key("interrupt-1") not in session_state


def test_expired_resume_error_clears_stale_approval_without_reconciliation(monkeypatch):
    import demo

    class SessionState(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

    class Status:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def update(self, **_kwargs):
            pass

    session_state = SessionState(
        pending_interrupt={"interrupt_id": "interrupt-1"},
        interrupt_conversation_id="conversation-1",
        pending_decisions_interrupt_1={"tool-1": {"type": "approve"}},
    )
    session_state[demo._hitl_resume_lock_key("interrupt-1")] = True
    errors: list[str] = []
    reconciled: list[str] = []
    reruns: list[bool] = []
    monkeypatch.setattr(
        demo,
        "st",
        SimpleNamespace(
            session_state=session_state,
            status=lambda *_args, **_kwargs: Status(),
            empty=lambda: object(),
            error=errors.append,
            rerun=lambda: reruns.append(True),
        ),
    )
    monkeypatch.setattr(demo, "_StreamingRichResponseRenderer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(demo, "_reset_stream_trace_state", lambda **_kwargs: None)
    monkeypatch.setattr(demo, "_clear_inflight_state", lambda: None)
    monkeypatch.setattr(
        demo,
        "make_streaming_request",
        lambda *_args, **_kwargs: iter(
            [
                {
                    "type": "error",
                    "error": "This approval has expired.",
                    "error_code": "INTERRUPT_EXPIRED",
                    "status_code": 410,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        demo,
        "_reconcile_interrupt",
        lambda interrupt_id: reconciled.append(interrupt_id) or "restore_form",
    )

    demo._submit_interrupt_decisions(
        "thread-1",
        "interrupt-1",
        [{"task_id": "tool-1"}],
        "pending_decisions_interrupt_1",
    )

    assert reconciled == []
    assert errors == ["This approval has expired."]
    assert reruns == [True]
    assert "pending_interrupt" not in session_state
    assert "pending_decisions_interrupt_1" not in session_state
    assert session_state[demo._hitl_reconciliation_key()] == "interrupt-1"
    assert session_state["hitl_reconciliation_notice"] == "This approval has expired."


def test_reset_conversation_clears_all_interrupt_session_state(monkeypatch):
    import demo

    class SessionState(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

    session_state = SessionState(
        pending_interrupt={"interrupt_id": "interrupt-1"},
        interrupt_conversation_id="conversation-1",
        pending_decisions_interrupt_1={"tool-1": {"type": "approve"}},
        editing_tool_0=True,
        hitl_resume_inflight_interrupt_1=True,
        hitl_resume_inflight_interrupt_2=True,
        hitl_reconciling_interrupt_id="interrupt-1",
        hitl_reconciliation_notice="Approval failed",
    )
    monkeypatch.setattr(demo, "st", SimpleNamespace(session_state=session_state))
    monkeypatch.setattr(demo, "_clear_inflight_state", lambda: None)

    demo.reset_conversation_state()

    for key in (
        "pending_interrupt",
        "interrupt_conversation_id",
        "pending_decisions_interrupt_1",
        "editing_tool_0",
        "hitl_resume_inflight_interrupt_1",
        "hitl_resume_inflight_interrupt_2",
        "hitl_reconciling_interrupt_id",
        "hitl_reconciliation_notice",
    ):
        assert key not in session_state


def test_approval_controls_are_disabled_before_per_tool_actions_when_resume_is_inflight():
    tree = ast.parse(_demo_source())
    approval_ui = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "render_interrupt_approval_ui"
    )
    resume_assignment = next(
        node
        for node in ast.walk(approval_ui)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "resume_inflight"
            for target in node.targets
        )
    )
    first_tool_loop = next(node for node in ast.walk(approval_ui) if isinstance(node, ast.For))
    assert resume_assignment.lineno < first_tool_loop.lineno

    button_disabled = {}
    for node in ast.walk(approval_ui):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"button", "form_submit_button"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            continue
        disabled = next((item.value for item in node.keywords if item.arg == "disabled"), None)
        button_disabled[(node.func.attr, node.args[0].value)] = (
            ast.unparse(disabled) if disabled else None
        )

    for button_kind, label in (
        ("button", "Change decision"),
        ("button", "Approve"),
        ("button", "Edit Args"),
        ("button", "Reject"),
        ("form_submit_button", "Save & Approve"),
        ("form_submit_button", "Cancel"),
        ("button", "Submit Decisions"),
        ("button", "Approve All"),
        ("button", "Cancel All"),
    ):
        assert "resume_inflight" in str(button_disabled[(button_kind, label)])

    arguments_field = next(
        node
        for node in ast.walk(approval_ui)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "text_area"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "Arguments (JSON format)"
    )
    disabled = next(
        (item.value for item in arguments_field.keywords if item.arg == "disabled"), None
    )
    assert disabled is not None
    assert ast.unparse(disabled) == "resume_inflight"


@pytest.mark.parametrize(
    ("lifecycle_state", "expected_action", "expected_notice"),
    [
        ("pending", "restore_form", None),
        ("resolving", "show_processing", None),
        ("resolved", "refresh_history", "Approval completed elsewhere; conversation refreshed."),
        (
            "failed",
            "require_new_message",
            "This approval can no longer be resumed. Send a new message.",
        ),
        (
            "expired",
            "require_new_message",
            "This approval can no longer be resumed. Send a new message.",
        ),
        (
            None,
            "require_new_message",
            "This approval can no longer be resumed. Send a new message.",
        ),
    ],
)
def test_duplicate_resume_reconciles_once_without_replay_for_every_lifecycle_outcome(
    monkeypatch, lifecycle_state, expected_action, expected_notice
):
    import demo

    class SessionState(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

    class Status:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def update(self, **_kwargs):
            pass

    interrupt_id = "interrupt-1"
    session_state = SessionState(
        pending_interrupt={"interrupt_id": interrupt_id},
        interrupt_conversation_id="conversation-1",
        pending_decisions_interrupt_1={"tool-1": {"type": "approve"}},
        editing_tool_0=True,
        hitl_reconciling_interrupt_id=interrupt_id,
    )
    session_state[demo._hitl_resume_lock_key(interrupt_id)] = True
    reruns: list[bool] = []
    resume_requests: list[str] = []
    state_reads: list[str] = []
    monkeypatch.setattr(
        demo,
        "st",
        SimpleNamespace(
            session_state=session_state,
            status=lambda *_args, **_kwargs: Status(),
            empty=lambda: object(),
            rerun=lambda: reruns.append(True),
            error=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setattr(demo, "_StreamingRichResponseRenderer", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(demo, "_reset_stream_trace_state", lambda **_kwargs: None)
    monkeypatch.setattr(demo, "_clear_inflight_state", lambda: None)
    monkeypatch.setattr(
        demo,
        "make_streaming_request",
        lambda endpoint, *_args, **_kwargs: (
            resume_requests.append(endpoint)
            or iter(
                [{"type": "error", "error": "Already resolved", "error_code": "INTERRUPT_CONFLICT"}]
            )
        ),
    )
    monkeypatch.setattr(
        demo,
        "get_hitl_interrupt_state",
        lambda requested_id: (
            state_reads.append(requested_id)
            or ({"status": lifecycle_state} if lifecycle_state is not None else None)
        ),
    )

    demo._submit_interrupt_decisions(
        "thread-1",
        interrupt_id,
        [{"task_id": "tool-1"}],
        "pending_decisions_interrupt_1",
    )

    assert state_reads == [interrupt_id]
    assert resume_requests == ["/messages/resume-interrupt"]
    assert reruns == [True]

    lock_key = demo._hitl_resume_lock_key(interrupt_id)
    marker_key = demo._hitl_reconciliation_key()
    if expected_action == "restore_form":
        assert "pending_interrupt" in session_state
        assert "pending_decisions_interrupt_1" in session_state
        assert "editing_tool_0" in session_state
        assert lock_key not in session_state
        assert marker_key not in session_state
        assert "hitl_reconciliation_notice" not in session_state
    elif expected_action == "show_processing":
        assert "pending_interrupt" in session_state
        assert "pending_decisions_interrupt_1" not in session_state
        assert "editing_tool_0" not in session_state
        assert session_state[lock_key] is True
        assert session_state[marker_key] == interrupt_id
        assert "hitl_reconciliation_notice" not in session_state
    else:
        assert "pending_interrupt" not in session_state
        assert "interrupt_conversation_id" not in session_state
        assert "pending_decisions_interrupt_1" not in session_state
        assert "editing_tool_0" not in session_state
        assert lock_key not in session_state
        assert session_state[marker_key] == interrupt_id
        assert session_state["hitl_reconciliation_notice"] == expected_notice
