from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest
import requests

from shared.integrations import attendance_api, take100_api

BASE = "https://abcd-1-2-3-4.ngrok-free.app"
AttendanceClient = attendance_api.AttendanceClient


def _record(day: str, check_in: str, check_out: str | None, code: str = "VT9999") -> dict:
    return {
        "maNhanVien": code,
        "tenNhanVien": "Nguyen Van A",
        "phongBan": "P. IT",
        "attendanceDate": day,
        "checkIn": check_in,
        "checkOut": check_out,
        "tenMay": "MCC00001",
    }


def _status(day: str, applications: list) -> dict:
    period = {"from": "2026-09-21", "to": "2026-10-20", "month": "this_month"}
    return take100_api.summarize_day(day, period, applications)


def _late_app(day: str, until: str) -> dict:
    return {
        "application_type_name": "paid leave",
        "reason_id": take100_api.REASON_ID_HOUR_LEAVE,
        "from_date": day,
        "to_date": day,
        "from_time": "08:00:00",
        "to_time": f"{until}:00",
        "status": take100_api.APP_STATUS_FINISHED,
        "created_at": "2026-09-30 10:00:00",
    }


def _ot_app(day: str, until: str) -> dict:
    return {
        "application_type_name": "over time",
        "reason_id": take100_api.REASON_ID_OVERTIME,
        "from_date": day,
        "to_date": day,
        "from_time": "17:00:00",
        "to_time": f"{until}:00",
        "status": take100_api.APP_STATUS_IN_PROGRESS,
        "created_at": "2026-09-30 10:00:00",
    }


class FakeResponse:
    def __init__(self, *, status_code=200, json_data=None, text="", headers=None):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text or (json.dumps(json_data) if json_data is not None else "")
        self.headers = headers or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


class FakePanel:
    """Stands in for the Next.js panel: NextAuth endpoints plus /api/attendance."""

    def __init__(self, *, rows=None, accept_login=True, attendance_statuses=None):
        self.cookies = requests.cookies.RequestsCookieJar()
        self.headers = {}
        self.rows = rows or []
        self.accept_login = accept_login
        self.attendance_statuses = list(attendance_statuses or [])
        self.login_posts: list[dict] = []
        self.attendance_params: list[dict] = []

    def get(self, url, **kwargs):
        path = url.removeprefix(BASE)
        if path == "/api/auth/csrf":
            return FakeResponse(json_data={"csrfToken": "csrf-1"})
        if path == "/api/auth/session":
            logged_in = self.cookies.get("__Secure-next-auth.session-token")
            return FakeResponse(json_data={"user": {"role": "admin"}} if logged_in else {})
        if path == "/api/attendance":
            self.attendance_params.append(kwargs["params"])
            status = self.attendance_statuses.pop(0) if self.attendance_statuses else 200
            if status == 401:
                return FakeResponse(status_code=401, json_data={"error": "Unauthorized"})
            return FakeResponse(json_data={"data": self.rows})
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url, **kwargs):
        assert url == f"{BASE}/api/auth/callback/credentials"
        assert kwargs["allow_redirects"] is False
        self.login_posts.append(kwargs["data"])
        if self.accept_login:
            self.cookies.set("__Secure-next-auth.session-token", "session-1")
        return FakeResponse(json_data={"url": "https://stale-tunnel.ngrok-free.app"})


def _client(tmp_path: Path, panel: FakePanel) -> AttendanceClient:
    client = AttendanceClient(BASE, "admin@example.com", "secret", tmp_path / "session.json")
    client.session = panel
    return client


# --- URL and date range -----------------------------------------------------------


def test_normalize_base_url_strips_a_pasted_login_path():
    assert attendance_api.normalize_base_url(f"{BASE}/login") == BASE
    assert attendance_api.normalize_base_url(f" {BASE}/ ") == BASE


def test_normalize_base_url_rejects_non_http_values():
    with pytest.raises(ValueError):
        attendance_api.normalize_base_url("abcd-1-2-3-4.ngrok-free.app")


@pytest.mark.parametrize(
    ("today", "expected_start"),
    [
        (date(2026, 10, 12), date(2026, 9, 21)),
        (date(2026, 10, 21), date(2026, 10, 21)),
        (date(2026, 10, 31), date(2026, 10, 21)),
        (date(2026, 1, 5), date(2025, 12, 21)),
    ],
)
def test_payroll_cycle_starts_on_the_21st_and_ends_today(today, expected_start):
    assert attendance_api.payroll_cycle(today) == (expected_start, today)


# --- Per-day analysis ---------------------------------------------------------------


def test_filed_late_application_excuses_the_late_arrival():
    day = attendance_api.analyze_day(
        _record("2026-09-21", "08:47:30", "17:00:10"),
        _status("2026-09-21", [_late_app("2026-09-21", "08:48")]),
    )

    assert day["late_minutes"] == 47
    assert day["unexcused_late_minutes"] == 0
    assert day["filed"]["late_until"] == "08:48"


def test_unfiled_late_arrival_is_reported_as_unexcused():
    day = attendance_api.analyze_day(
        _record("2026-09-30", "09:23:35", "17:05:00"), _status("2026-09-30", [])
    )

    assert day["late_minutes"] == 83
    assert day["unexcused_late_minutes"] == 83


def test_overtime_beyond_the_filed_window_is_unclaimed():
    day = attendance_api.analyze_day(
        _record("2026-09-23", "07:53:20", "19:10:00"),
        _status("2026-09-23", [_ot_app("2026-09-23", "18:30")]),
    )

    assert day["overtime_minutes"] == 130
    assert day["unclaimed_overtime_minutes"] == 40


def test_overtime_without_any_application_is_fully_unclaimed():
    day = attendance_api.analyze_day(
        _record("2026-09-24", "07:52:32", "18:00:00"), _status("2026-09-24", [])
    )

    assert day["unclaimed_overtime_minutes"] == 60


def test_leaving_before_the_filed_overtime_end_is_noted():
    day = attendance_api.analyze_day(
        _record("2026-09-28", "07:53:53", "20:30:00"),
        _status("2026-09-28", [_ot_app("2026-09-28", "21:00")]),
    )

    assert day["unclaimed_overtime_minutes"] == 0
    assert any("filed until 21:00" in note for note in day["notes"])


def test_missing_check_out_leaves_checkout_figures_unknown():
    day = attendance_api.analyze_day(_record("2026-10-12", "07:56:26", None), None)

    assert day["overtime_minutes"] is None
    assert day["unclaimed_overtime_minutes"] is None
    assert any("No check-out" in note for note in day["notes"])


def test_unchecked_applications_never_claim_anything_is_unexcused():
    day = attendance_api.analyze_day(_record("2026-09-30", "09:00:00", "19:00:00"), None)

    assert day["late_minutes"] == 60
    assert day["overtime_minutes"] == 120
    assert day["unexcused_late_minutes"] is None
    assert day["unclaimed_overtime_minutes"] is None
    assert day["filed"] is None


def test_missing_workdays_skip_weekends_and_report_filed_leave():
    leave = {
        "application_type_name": "paid leave",
        "reason_id": take100_api.REASON_ID_ALL_DAY_LEAVE,
        "from_date": "2026-10-07",
        "to_date": "2026-10-07",
        "status": take100_api.APP_STATUS_FINISHED,
    }
    records = [_record("2026-10-05", "08:00:00", "17:00:00")]
    statuses = {"2026-10-07": _status("2026-10-07", [leave])}

    missing = attendance_api.missing_workdays(
        records, date(2026, 10, 3), date(2026, 10, 7), statuses
    )

    assert missing == [
        {"date": "2026-10-06", "weekday": "Tue", "all_day_leave_filed": None},
        {"date": "2026-10-07", "weekday": "Wed", "all_day_leave_filed": True},
    ]


def test_build_report_totals_and_unchecked_flag():
    records = [
        _record("2026-10-05", "08:10:00", "18:00:00"),
        _record("2026-10-06", "07:55:00", "17:00:00"),
    ]

    report = attendance_api.build_report(records, date(2026, 10, 5), date(2026, 10, 6), None)

    assert report["applications_checked"] is False
    assert report["totals"]["days_late"] == 1
    assert report["totals"]["late_minutes"] == 10
    assert report["totals"]["overtime_minutes"] == 60
    assert report["totals"]["unclaimed_overtime_minutes"] is None


def test_day_statuses_only_cover_dates_inside_a_payroll_period():
    periods = [({"from": "2026-09-21", "to": "2026-10-20", "month": "this_month"}, [])]

    statuses = attendance_api.day_statuses(periods, date(2026, 10, 19), date(2026, 10, 22))

    assert sorted(statuses) == ["2026-10-19", "2026-10-20"]


# --- HTTP client --------------------------------------------------------------------


def test_fetch_logs_in_once_on_401_and_persists_the_session(tmp_path):
    panel = FakePanel(rows=[_record("2026-10-05", "08:00:00", "17:00:00")],
                      attendance_statuses=[401, 200])
    client = _client(tmp_path, panel)

    rows = client.fetch_attendance(date(2026, 10, 5), date(2026, 10, 5), "VT9999")

    assert len(rows) == 1
    assert len(panel.login_posts) == 1
    assert panel.login_posts[0]["identifier"] == "admin@example.com"
    assert panel.attendance_params[0]["maNhanVien"] == "VT9999"
    saved = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    assert saved["base_url"] == BASE
    assert "secret" not in json.dumps(saved)


def test_fetch_never_returns_another_employees_rows(tmp_path):
    panel = FakePanel(rows=[
        _record("2026-10-06", "08:00:00", "17:00:00", code="VT0001"),
        _record("2026-10-05", "08:00:00", "17:00:00"),
    ])

    rows = _client(tmp_path, panel).fetch_attendance(date(2026, 10, 5), date(2026, 10, 6),
                                                     "VT9999")

    assert [row["maNhanVien"] for row in rows] == ["VT9999"]


def test_rejected_credentials_raise_a_login_error(tmp_path):
    panel = FakePanel(accept_login=False, attendance_statuses=[401])

    with pytest.raises(attendance_api.AttendanceLoginError):
        _client(tmp_path, panel).fetch_attendance(date(2026, 10, 5), date(2026, 10, 5),
                                                  "VT9999")
    assert not (tmp_path / "session.json").exists()


def test_offline_tunnel_raises_an_actionable_error(tmp_path):
    client = _client(tmp_path, FakePanel())
    offline = FakeResponse(status_code=404, text="<html>ERR_NGROK_3200</html>",
                           headers={"ngrok-error-code": "ERR_NGROK_3200"})
    client.session.get = lambda url, **kwargs: offline

    with pytest.raises(attendance_api.AttendanceUnavailableError, match="--base-url"):
        client.fetch_attendance(date(2026, 10, 5), date(2026, 10, 5), "VT9999")


def test_cached_cookies_from_another_tunnel_are_not_restored(tmp_path):
    cache = tmp_path / "session.json"
    cache.write_text(json.dumps({
        "base_url": "https://old-tunnel.ngrok-free.app",
        "username": "admin@example.com",
        "cookies": [{"name": "__Secure-next-auth.session-token", "value": "old",
                     "domain": "old-tunnel.ngrok-free.app", "path": "/", "secure": True,
                     "expires": None}],
    }), encoding="utf-8")

    client = AttendanceClient(BASE, "admin@example.com", "secret", cache)

    assert len(client.session.cookies) == 0
    assert attendance_api.read_cached_base_url(cache) == "https://old-tunnel.ngrok-free.app"


def test_client_verifies_tls_certificates(tmp_path):
    client = AttendanceClient(BASE, "admin@example.com", "secret", tmp_path / "session.json")
    assert client.session.verify is True


# --- CLI ----------------------------------------------------------------------------


def test_main_without_credentials_fails_without_network(monkeypatch, capsys):
    monkeypatch.delenv("ATTENDANCE_USERNAME", raising=False)
    monkeypatch.delenv("ATTENDANCE_PASSWORD", raising=False)
    monkeypatch.setattr(sys, "argv", ["attendance_api.py"])

    with pytest.raises(SystemExit):
        attendance_api.main()

    assert "ATTENDANCE_USERNAME" in json.loads(capsys.readouterr().out)["error"]


def test_other_employee_is_reported_without_the_application_comparison(monkeypatch):
    class StubTake100:
        def __init__(self, email, password):
            pass

        def get_employee_code(self):
            return "VT9999"

        def fetch_periods(self):
            pytest.fail("another employee's report must not read this account's applications")

    monkeypatch.setenv("TAKE100_EMAIL", "user@example.com")
    monkeypatch.setenv("TAKE100_PASSWORD", "secret")
    monkeypatch.setattr(take100_api, "Take100Client", StubTake100)
    warnings: list[str] = []

    employee, periods = attendance_api._take100_context("VT0001", warnings)

    assert (employee, periods) == ("VT0001", None)
    assert "not checked" in warnings[0]


def test_take100_employee_code_is_read_from_the_user_setting_form(monkeypatch, tmp_path):
    client = take100_api.Take100Client("user@example.com", "secret", tmp_path / "s.json")
    html = '<input name="id" id="id" type="text" readonly="true" value="VT9999">'
    monkeypatch.setattr(client, "_request", lambda *a, **k: FakeResponse(text=html))

    assert client.get_employee_code() == "VT9999"


def test_take100_employee_code_is_none_when_the_field_is_missing(monkeypatch, tmp_path):
    client = take100_api.Take100Client("user@example.com", "secret", tmp_path / "s.json")
    monkeypatch.setattr(client, "_request", lambda *a, **k: FakeResponse(text="<form></form>"))

    assert client.get_employee_code() is None


# --- suggested applications ---------------------------------------------------------


def test_late_and_overtime_without_applications_are_suggested():
    record = _record("2026-10-14", "08:51:03", "19:36:38")

    suggestions = attendance_api.suggest_applications(record, _status("2026-10-14", []))

    assert suggestions == [
        {"kind": "hour-leave", "date": "2026-10-14", "from": "08:00", "to": "08:51",
         "reason": None},
        {"kind": "overtime", "date": "2026-10-14", "from": "17:00", "to": "19:35",
         "reason": None},
    ]


def test_filed_late_and_overtime_are_not_suggested_again():
    day = "2026-10-01"
    status = _status(day, [_late_app(day, "08:57"), _ot_app(day, "20:05")])

    assert attendance_api.suggest_applications(_record(day, "08:57:59", "20:06:52"), status) == []


def test_checkout_inside_the_rounding_step_is_not_overtime():
    record = _record("2026-10-02", "07:56:58", "17:04:59")

    assert attendance_api.suggest_applications(record, _status("2026-10-02", [])) == []


def test_no_suggestions_without_applications_weekend_or_missing_checkout():
    assert attendance_api.suggest_applications(_record("2026-10-14", "09:00:00", None), None) == []
    weekend = _record("2026-10-17", "09:00:00", "18:00:00")
    assert attendance_api.suggest_applications(weekend, _status("2026-10-17", [])) == []
    still_at_work = _record("2026-10-16", "07:50:00", None)
    assert attendance_api.suggest_applications(still_at_work, _status("2026-10-16", [])) == []


def test_underfiled_applications_become_notes_not_suggestions():
    day = "2026-10-12"
    status = _status(day, [_late_app(day, "08:30"), _ot_app(day, "19:00")])

    report = attendance_api.analyze_day(_record(day, "08:45:10", "19:36:38"), status)

    assert any("filed until 08:30 but check-in was 08:45" in note for note in report["notes"])
    assert any("would allow 19:35" in note for note in report["notes"])
    assert attendance_api.suggest_applications(_record(day, "08:45:10", "19:36:38"), status) == []


def test_report_lists_suggestions_across_days():
    records = [_record("2026-10-13", "07:50:00", "18:31:00"),
               _record("2026-10-14", "08:05:00", "17:00:30")]
    statuses = {day: _status(day, []) for day in ("2026-10-13", "2026-10-14")}

    report = attendance_api.build_report(records, date(2026, 10, 13), date(2026, 10, 14), statuses)

    suggested = report["suggested_applications"]
    assert [(item["kind"], item["date"], item["to"]) for item in suggested] == [
        ("overtime", "2026-10-13", "18:30"),
        ("hour-leave", "2026-10-14", "08:05"),
    ]
