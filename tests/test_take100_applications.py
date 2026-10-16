from __future__ import annotations

import json

import pytest
import requests

from shared.integrations import take100_applications as apps
from shared.integrations.take100_api import CommandRefused

# One row exactly as /other-applications renders it (2026-10-16), trimmed of the
# progress bar markup.
ROW_HTML = """
<table><tbody>
<tr class="row-item cursor-pointer" data-id="36676" ondblclick="get('36676')">
  <th class="text-sm">1</th>
  <th class="text-sm text-left align-middle">100-36676</th>
  <th class="text-sm text-left align-middle"><i class="fa"></i> Normal</th>
  <th class="text-sm text-left align-middle"><span class="badge">
      In progress </span></th>
  <th class="text-sm text-left align-middle capitalize">paid leave - Hour </th>
  <th class="text-sm text-left align-middle py-0"><div> 10-16 (Fri) </div></th>
  <th class="text-sm text-left align-middle">2026-10-16</th>
  <th class="text-sm text-left align-middle"> 0 /
      2 <div class="progress"></div></th>
  <th class="text-sm text-left align-middle">Lý do sức khỏe</th>
  <th class="text-sm text-left align-middle"></th>
  <th></th>
</tr>
</tbody></table>
"""

# The create page's embedded values, as served on 2026-10-16.
CREATE_HTML = """
<form id="newOtherApplicationForm"><input type="hidden" name="_token" value="tok123"></form>
<script>
    const submittedOtherApps = [];
    const shiftWork = {"id":"C001","from":"08:00","to":"17:00"};
    const remainingPaidLeaveDay = 1;
    const paidLeaveHoursLeft = "3:21";
    const remainingPaidLeaveHour = 39.35;
    const lockedDate = "2026-07-21";
</script>
"""

DETAIL_HTML = """
<form action="https://take100dot.com/api/other-applications/36700" method="POST"
      id="viewInDetailOtherApplicationForm">
  <input type="hidden" name="_method" value="PUT">
  <input type="hidden" name="_token" value="tok123">
  <input name="reason_id" type="radio" value="1">
  <input name="reason_id" type="radio" value="5" checked>
  <input id="fromDate" value=" 2026-10-16 " name="from_date" type="text">
  <input id="toDate" value=" 2026-10-16 " name="to_date" type="text">
  <input value=" 2026-10-16 " name="date" type="text">
  <input id="fromTime" name="from_time" type="time" value="08:00">
  <input id="toTime" name="to_time" type="time" value="08:51">
  <input type="checkbox" name="priority" value="1">
  <textarea name="reason">Lý do sức khỏe</textarea>
  <textarea name="message"></textarea>
  <input type="file" id="attachedFile">
</form>
"""


def _spec(kind="hour-leave", day="2026-10-16", start="08:00", end="08:51", **extra) -> dict:
    raw = {"kind": kind, "date": day, "from": start, "to": end, "reason": "Đi trễ", **extra}
    return apps.parse_specs(json.dumps([raw]))[0]


def _context(**overrides) -> dict:
    return {**apps.parse_create_context(CREATE_HTML), **overrides}


def _row(kind="overtime", day="2026-10-14", status="in_progress", no="100-36609") -> dict:
    return {"id": no[4:], "no": no, "status": status, "type": kind, "kind": kind,
            "apply_date": day, "date_from": day, "date_to": day}


class StubApi:
    def __init__(self, rows=None, fail_on=None):
        self.rows = rows or []
        self.fail_on = fail_on
        self.created: list[dict] = []
        self.submitted: list[tuple] = []

    def create_context(self):
        return _context()

    def list_rows(self, date_from=None, date_to=None, status="all", app_type="all"):
        return [row for row in self.rows if status == "all" or row["status"] == status]

    def create(self, form):
        if form.get("date") == self.fail_on or form.get("from_date") == self.fail_on:
            response = requests.Response()
            response.status_code = 422
            response._content = json.dumps({"user_res_msg": {"to_time": ["bad"]}}).encode()
            raise requests.HTTPError(response=response)
        self.created.append(form)
        return {"id": len(self.created)}

    def submit_draft(self, application_id, kind):
        self.submitted.append((application_id, kind))
        return {"ok": True}


# --- time rules -------------------------------------------------------------


@pytest.mark.parametrize(
    ("check_out", "expected"),
    [("19:36:38", "19:35"), ("20:00:06", "20:00"), ("17:31:00", "17:30"), ("17:04", "17:00")],
)
def test_floor_to_step_matches_filed_overtime(check_out, expected):
    assert apps.floor_to_step(check_out) == expected


@pytest.mark.parametrize(
    ("start", "end", "minutes"),
    [
        ("08:00", "08:51", 51),
        ("08:00", "13:30", 270),  # spans lunch: 330 - 60
        ("12:30", "14:00", 60),  # starts on lunch: counted from 13:00
        ("10:00", "12:30", 120),  # ends on lunch: counted to 12:00
        ("12:10", "12:50", 0),
    ],
)
def test_leave_minutes_excludes_lunch_like_the_site(start, end, minutes):
    assert apps.leave_minutes(start, end) == minutes


# --- spec validation ----------------------------------------------------------


def test_parse_specs_rejects_bad_json_and_empty_lists():
    with pytest.raises(CommandRefused):
        apps.parse_specs("not json")
    with pytest.raises(CommandRefused):
        apps.parse_specs("[]")
    with pytest.raises(CommandRefused):
        apps.parse_specs(None)


def test_parse_specs_defaults_to_date_and_accepts_a_single_object():
    spec = apps.parse_specs(json.dumps({"kind": "day-leave", "date": "2026-10-07",
                                         "reason": " Ốm "}))[0]
    assert spec["to_date"] == "2026-10-07"
    assert spec["reason"] == "Ốm"


def test_missing_reason_and_bad_times_are_reported():
    spec = _spec(start="8:00", reason="")
    problems = apps.plan_problems([spec], _context(), [])
    [messages] = problems.values()
    assert any("from must be HH:MM" in message for message in messages)
    assert any("reason is required" in message for message in messages)


def test_unknown_kind_is_reported():
    problems = apps.plan_problems([_spec(kind="late")], _context(), [])
    assert "kind must be one of" in next(iter(problems.values()))[0]


def test_hour_leave_must_stay_inside_the_shift():
    problems = apps.plan_problems([_spec(start="07:30", end="08:30")], _context(), [])
    assert "within 08:00-17:00" in next(iter(problems.values()))[0]


@pytest.mark.parametrize(
    ("start", "end"),
    [("16:00", "18:00"), ("07:00", "09:00"), ("08:00", "17:00"), ("18:00", "17:30")],
)
def test_weekday_overtime_outside_the_shift_only(start, end):
    problems = apps.plan_problems([_spec("overtime", "2026-10-14", start, end)], _context(), [])
    assert problems


def test_weekend_overtime_may_cover_the_day():
    spec = _spec("overtime", "2026-10-17", "08:00", "17:00")  # a Saturday
    assert apps.plan_problems([spec], _context(), []) == {}


def test_dates_before_the_locked_date_are_refused():
    problems = apps.plan_problems([_spec(day="2026-07-20")], _context(), [])
    assert "locked date 2026-07-21" in next(iter(problems.values()))[-1]


def test_same_kind_on_file_is_a_duplicate_unless_allowed():
    spec = _spec("overtime", "2026-10-14", "17:00", "18:30")
    rows = [_row("overtime", "2026-10-14")]
    assert "already covers" in next(iter(apps.plan_problems([spec], _context(), rows).values()))[0]
    assert apps.plan_problems([spec], _context(), rows, allow_duplicate=True) == {}


def test_denied_and_other_kinds_are_not_duplicates():
    spec = _spec("overtime", "2026-10-14", "17:00", "18:30")
    rows = [_row("overtime", "2026-10-14", status="denied"), _row("hour-leave", "2026-10-14")]
    assert apps.plan_problems([spec], _context(), rows) == {}


def test_the_same_kind_twice_in_one_batch_is_refused():
    specs = [_spec(), _spec(start="08:00", end="08:30")]
    problems = apps.plan_problems(specs, _context(), [])
    assert "twice" in problems["2 hour-leave 2026-10-16"][0]


def test_hourly_balance_is_checked_over_the_whole_batch():
    context = _context()
    context["balance"] = {**context["balance"], "remaining_hourly_minutes": 60}
    specs = [_spec(day="2026-10-15", end="08:40"), _spec(day="2026-10-16", end="08:40")]
    problems = apps.plan_problems(specs, context, [])
    assert "remaining hourly allowance" in problems["balance"][0]


def test_day_leave_counts_eight_hours_per_weekday_against_the_balance():
    # 1 day + 3:21 left = 681 minutes; two weekdays need 960.
    spec = _spec("day-leave", "2026-10-15", None, None, to_date="2026-10-16")
    problems = apps.plan_problems([spec], _context(), [])
    assert "remaining paid leave (681 min)" in problems["balance"][0]


# --- page parsing ---------------------------------------------------------------


def test_create_context_reads_token_lock_and_balance():
    context = apps.parse_create_context(CREATE_HTML)
    assert context["token"] == "tok123"
    assert context["locked_date"] == "2026-07-21"
    assert context["balance"]["remaining_leave_minutes"] == 8 * 60 + 3 * 60 + 21
    assert context["balance"]["remaining_hourly_minutes"] == 2361


def test_list_rows_parse_like_the_site_renders_them():
    [row] = apps.parse_application_rows(ROW_HTML)
    assert row["id"] == "36676"
    assert row["no"] == "100-36676"
    assert row["status"] == "in_progress"
    assert row["kind"] == "hour-leave"
    assert (row["date_from"], row["date_to"]) == ("2026-10-16", "2026-10-16")
    assert row["progress"] == "0/2"
    assert row["reason"] == "Lý do sức khỏe"


def test_apply_date_year_steps_back_across_new_year():
    assert apps._apply_dates("12-30 (Wed)", "2027-01-04") == ("2026-12-30", "2026-12-30")


def test_detail_form_serialises_like_form_data():
    fields = apps.parse_detail_form(DETAIL_HTML)
    assert fields["reason_id"] == "5"
    assert fields["date"] == "2026-10-16"
    assert fields["reason"] == "Lý do sức khỏe"
    assert "priority" not in fields


# --- request bodies -------------------------------------------------------------


def test_hour_leave_form_matches_the_create_page():
    form = apps.build_form(_spec(), "save", "tok")
    assert form == {
        "_token": "tok", "reason_id": "5", "from_date": "", "to_date": "",
        "date": "2026-10-16", "from_time": "08:00", "to_time": "08:51",
        "reason": "Đi trễ", "message": "", "application_type_id": "3", "action_type": "save",
    }


def test_overtime_form_sends_no_reason_id_and_uses_the_date_range():
    form = apps.build_form(_spec("overtime", "2026-10-14", "17:00", "18:00", priority=True),
                           "submit", "tok")
    assert "reason_id" not in form
    assert (form["from_date"], form["to_date"], form["date"]) == ("2026-10-14", "2026-10-14", "")
    assert form["application_type_id"] == "2"
    assert form["priority"] == "1"
    assert form["action_type"] == "submit"


def test_day_leave_form_has_no_times():
    form = apps.build_form(_spec("day-leave", "2026-10-07", None, None), "save", "tok")
    assert form["reason_id"] == "1"
    assert (form["from_time"], form["to_time"]) == ("", "")


# --- commands --------------------------------------------------------------------


def test_file_applications_sends_nothing_when_any_spec_is_wrong():
    api = StubApi()
    with pytest.raises(CommandRefused) as refused:
        apps.file_applications(api, [_spec(), _spec("overtime", "2026-10-14", "16:00", "18:00")],
                               submit=False, dry_run=False)
    assert api.created == []
    assert "2 overtime 2026-10-14" in refused.value.payload["problems"]


def test_file_applications_dry_run_sends_nothing():
    api = StubApi()
    result = apps.file_applications(api, [_spec()], submit=True, dry_run=True)
    assert result["dry_run"] is True
    assert result["action_type"] == "submit"
    assert api.created == []


def test_file_applications_saves_drafts_by_default_and_reports_each_result():
    api = StubApi(fail_on="2026-10-14")
    specs = [_spec(), _spec("overtime", "2026-10-14", "17:00", "18:00")]
    result = apps.file_applications(api, specs, submit=False, dry_run=False)
    assert [form["action_type"] for form in api.created] == ["save"]
    assert result["success"] is False
    assert result["results"][0]["success"] is True
    assert result["results"][1]["details"] == {"to_time": ["bad"]}


def test_submit_drafts_refuses_ids_that_are_not_drafts():
    api = StubApi(rows=[_row(status="draft", no="100-36700")])
    with pytest.raises(CommandRefused) as refused:
        apps.submit_application_drafts(api, ["36700", "36609"], None, dry_run=False)
    assert refused.value.payload["ids"] == ["36609"]
    assert api.submitted == []


def test_submit_drafts_puts_each_draft_with_its_kind():
    api = StubApi(rows=[_row("hour-leave", status="draft", no="100-36700"),
                        _row(status="in_progress")])
    result = apps.submit_application_drafts(api, None, ("2026-09-21", "2026-10-20"),
                                            dry_run=False)
    assert result["success"] is True
    assert api.submitted == [("36700", "hour-leave")]


def test_submit_drafts_needs_ids_or_all_saved():
    with pytest.raises(CommandRefused):
        apps.submit_application_drafts(StubApi(), None, None, dry_run=False)
