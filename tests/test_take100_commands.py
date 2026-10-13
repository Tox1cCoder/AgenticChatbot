from __future__ import annotations

import argparse
import json

import pytest

from shared.integrations import take100_api
from shared.integrations.take100_api import CommandRefused

PERIOD = {"from": "2026-09-21", "to": "2026-10-20", "month": "this_month"}

# One row exactly as /wt-applications renders it (Saved tab, 2026-10-13).
ROW_HTML = """
<table id="products-list"><tbody>
<tr> <td class="text-sm">1</td> <td class="text-sm">100-36531</td>
<td class="text-sm">Oct 8, 2026 ~ Oct 8, 2026</td> <td class="text-sm"></td>
<td class="text-sm">0/1</td> <td class="text-sm"></td>
<td> <a href=" https://take100dot.com/wt-applications/36531 "><i></i></a> </td> </tr>
</tbody></table>
"""


def _entries(time_in="08:00", time_out="17:00"):
    return [
        {"from": time_in, "to": "12:00", "project_id": "F131", "work_item_id": 128,
         "work_content": "Phân tích yêu cầu"},
        {"from": "13:00", "to": time_out, "project_id": "F131", "work_item_id": 128,
         "work_content": "Task"},
    ]


def _row(day: str, status: str = "saved", no: str = "100-1") -> dict:
    return {"id": no[4:], "no": no, "status": status, "date_from": day, "date_to": day,
            "requested_time": None, "process": "0/1"}


def _draft(application_id=36519, day="2026-09-21", status=0, time_in="08:03") -> dict:
    """A saved application as its detail page embeds it (trimmed to the fields used)."""
    return {
        "id": application_id, "no": f"100-{application_id}", "status": status, "message": None,
        "working_days": [{
            "time_in": f"{time_in}:00", "time_out": "17:00:00", "shift_work_id": "C001",
            "date": f"{day}T00:00:00.000000Z",
            "working_day_projects": [
                {"project_id": "K202201", "work_content": "Tập thể dục, báo cáo buổi sáng",
                 "work_item_id": 122, "from": f"{time_in}:00", "to": "12:00:00"},
                {"project_id": "F131", "work_content": "Implement 'Project' Feature",
                 "work_item_id": 128, "from": "13:00:00", "to": "17:00:00"},
            ],
        }],
    }


class StubClient:
    def __init__(self, rows=None, applications=None, drafts=None, delete_fails=False):
        self.rows = rows or []
        self.applications = applications or []
        self.drafts = {draft["id"]: draft for draft in drafts or []}
        self.delete_fails = delete_fails
        self.saved: list[tuple] = []
        self.submitted: list[list[dict]] = []
        self.deleted: list[int] = []
        self.calls: list[str] = []

    def get_application(self, application_id):
        return self.drafts[application_id]

    def submit_working_days(self, working_days, message=""):
        self.calls.append("submit")
        self.submitted.append(working_days)
        return {"id": 99}

    def delete_application(self, application_id):
        self.calls.append("delete")
        if self.delete_fails:
            response = type("R", (), {"status_code": 500, "text": "boom"})()
            raise take100_api.requests.HTTPError(response=response)
        self.deleted.append(application_id)
        return {"status": "deleted"}

    def fetch_periods(self):
        return [(PERIOD, self.applications)]

    def get_day_status(self, date):
        return take100_api.summarize_day(date, PERIOD, self.applications)

    def list_timesheets(self, statuses, date_from=None):
        return [row for row in self.rows if row["status"] in statuses]

    def save_timesheet(self, date, entries, message="", time_in="08:00", time_out="17:00"):
        self.saved.append((date, time_in, time_out, entries))
        return {"id": len(self.saved)}

    submit_timesheet = save_timesheet


def _args(**overrides) -> argparse.Namespace:
    defaults = {
        "action": "save", "date": None, "entries": None, "entries_file": None, "days": None,
        "status": "all", "date_from": None, "date_to": None, "application_id": None,
        "message": "", "time_in": None, "time_out": None, "no_auto": False,
        "skip_validation": False, "submit": False, "ids": None, "all_saved": False,
        "dry_run": False,
    }
    return argparse.Namespace(**{**defaults, **overrides})


# --- list ---------------------------------------------------------------------------


def test_timesheet_rows_parse_dates_ids_and_status():
    rows = take100_api.parse_timesheet_rows(ROW_HTML, "saved")

    assert rows == [{
        "id": "36531", "no": "100-36531", "status": "saved", "date_from": "2026-10-08",
        "date_to": "2026-10-08", "requested_time": None, "process": "0/1",
    }]


def test_list_reads_each_tab_until_an_empty_page(monkeypatch, tmp_path):
    client = take100_api.Take100Client("u@example.com", "pw", tmp_path / "s.json")
    pages = {(0, 1): ROW_HTML, (0, 2): ""}
    calls = []

    class Response:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

    def fake_request(method, url, params=None, **kwargs):
        calls.append((params["status"], params["page"]))
        return Response(pages.get((params["status"], params["page"]), ""))

    monkeypatch.setattr(client, "_request", fake_request)

    rows = client.list_timesheets(["saved"])

    assert [row["no"] for row in rows] == ["100-36531"]
    assert calls == [(0, 1), (0, 2)]


def test_list_stops_paging_once_a_page_is_older_than_the_range(monkeypatch, tmp_path):
    client = take100_api.Take100Client("u@example.com", "pw", tmp_path / "s.json")
    calls = []

    class Response:
        text = ROW_HTML

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        client, "_request", lambda method, url, params=None, **k: calls.append(params) or Response()
    )

    client.list_timesheets(["finished"], date_from="2026-10-09")

    assert len(calls) == 1


def test_list_reports_weekdays_without_a_timesheet_and_leave_days(monkeypatch):
    leave = {"application_type_name": "paid leave", "reason_id": 1, "from_date": "2026-09-23",
             "to_date": "2026-09-23", "status": 2}
    client = StubClient(rows=[_row("2026-09-21"), _row("2026-09-22", status="denied")],
                        applications=[leave])

    result = take100_api._cmd_list(client, _args(action="list", date_from="2026-09-21",
                                                 date_to="2026-09-23"))

    assert result["weekdays_without_timesheet"] == [
        {"date": "2026-09-22", "all_day_leave": False},
        {"date": "2026-09-23", "all_day_leave": True},
    ]


# --- save / submit ------------------------------------------------------------------


def test_save_refuses_a_day_that_already_has_a_timesheet():
    client = StubClient(rows=[_row("2026-10-08", no="100-36531")])

    with pytest.raises(CommandRefused) as refused:
        take100_api._cmd_save(client, _args(date="2026-10-08", entries=json.dumps(_entries())))

    assert refused.value.payload["on_file"]["2026-10-08"][0]["no"] == "100-36531"
    assert client.saved == []


def test_denied_timesheets_do_not_block_refiling():
    client = StubClient(rows=[_row("2026-10-08", status="denied")])

    result = take100_api._cmd_save(client, _args(date="2026-10-08",
                                                 entries=json.dumps(_entries())))

    assert result["success"] is True
    assert client.saved[0][0] == "2026-10-08"


def test_inline_unicode_entries_reach_the_site_unchanged():
    client = StubClient()

    take100_api._cmd_save(client, _args(date="2026-10-08", entries=json.dumps(_entries(),
                                                                              ensure_ascii=False)))

    assert client.saved[0][3][0]["work_content"] == "Phân tích yêu cầu"


# --- save-many ----------------------------------------------------------------------


def test_save_many_files_every_day_with_its_own_window():
    late = {"application_type_name": "paid leave", "reason_id": 5, "from_date": "2026-09-30",
            "to_date": "2026-09-30", "from_time": "08:00", "to_time": "09:23", "status": 1,
            "created_at": "2026-10-13"}
    client = StubClient(applications=[late])
    days = [
        {"date": "2026-09-29", "entries": _entries()},
        {"date": "2026-09-30", "entries": _entries(time_in="09:24")},
    ]

    result = take100_api._cmd_save_many(client, _args(action="save-many", days=json.dumps(days)))

    assert result["success"] is True
    assert [(day, time_in) for day, time_in, _, _ in client.saved] == [
        ("2026-09-29", "08:00"), ("2026-09-30", "09:24"),
    ]


def test_save_many_sends_nothing_when_any_day_is_invalid():
    client = StubClient(rows=[_row("2026-10-02")])
    days = [
        {"date": "2026-10-01", "entries": _entries()},
        {"date": "2026-10-02", "entries": _entries()},
        {"date": "2026-10-05", "entries": _entries(time_out="16:00")},
    ]

    with pytest.raises(CommandRefused) as refused:
        take100_api._cmd_save_many(client, _args(action="save-many", days=json.dumps(days)))

    assert sorted(refused.value.payload["problems"]) == ["2026-10-02", "2026-10-05"]
    assert client.saved == []


def test_save_many_rejects_repeated_dates():
    days = [{"date": "2026-10-01", "entries": []}, {"date": "2026-10-01", "entries": []}]

    with pytest.raises(CommandRefused, match="more than once"):
        take100_api._cmd_save_many(StubClient(), _args(action="save-many", days=json.dumps(days)))


def test_save_many_with_submit_posts_for_approval():
    client = StubClient()
    client.submit_timesheet = lambda date, entries, message="", **kw: client.calls.append(
        f"submit {date}") or {"id": 1}
    days = [{"date": "2026-09-29", "entries": _entries()}]

    result = take100_api._cmd_save_many(
        client, _args(action="save-many", days=json.dumps(days), submit=True)
    )

    assert result["submitted"] is True
    assert client.calls == ["submit 2026-09-29"]
    assert client.saved == []


# --- submit-drafts ------------------------------------------------------------------


def test_draft_days_rebuild_entries_from_the_embedded_application():
    days = take100_api.draft_days(_draft())

    assert days == [{
        "date": "2026-09-21", "time_in": "08:03", "time_out": "17:00", "shift_work_id": "C001",
        "entries": [
            {"from": "08:03", "to": "12:00", "project_id": "K202201", "work_item_id": 122,
             "work_content": "Tập thể dục, báo cáo buổi sáng"},
            {"from": "13:00", "to": "17:00", "project_id": "F131", "work_item_id": 128,
             "work_content": "Implement 'Project' Feature"},
        ],
    }]


def test_get_application_reads_the_object_the_detail_page_embeds(monkeypatch, tmp_path):
    client = take100_api.Take100Client("u@example.com", "pw", tmp_path / "s.json")
    html = f"<script>\n let application = {json.dumps(_draft())};\n let x = 1;</script>"

    class Response:
        text = html

        def raise_for_status(self):
            pass

    monkeypatch.setattr(client, "_request", lambda *a, **k: Response())

    assert client.get_application(36519)["working_days"][0]["shift_work_id"] == "C001"


def test_delete_sends_the_same_site_referer_the_api_requires(monkeypatch, tmp_path):
    client = take100_api.Take100Client("u@example.com", "pw", tmp_path / "s.json")
    seen = {}

    class Response:
        text = ""

        def raise_for_status(self):
            pass

    def fake_request(method, url, headers=None, **kwargs):
        seen.update(method=method, url=url, headers=headers)
        return Response()

    monkeypatch.setattr(client, "_request", fake_request)

    client.delete_application(36519)

    assert seen["method"] == "DELETE"
    assert seen["headers"]["Referer"].endswith("/wt-applications/36519")


def test_submit_drafts_submits_the_draft_content_then_deletes_the_draft():
    client = StubClient(drafts=[_draft()])

    result = take100_api._cmd_submit_drafts(client, _args(action="submit-drafts", ids="36519"))

    assert result["success"] is True
    assert client.calls == ["submit", "delete"]
    day = client.submitted[0][0]
    assert (day["date"], day["time_in"], day["time_out"]) == ("2026-09-21T00:00:00.000Z",
                                                              "08:03", "17:00")
    assert day["working_day_projects"][0]["work_content"] == "Tập thể dục, báo cáo buổi sáng"
    assert client.deleted == [36519]


def test_submit_drafts_dry_run_sends_nothing():
    client = StubClient(drafts=[_draft()])

    result = take100_api._cmd_submit_drafts(
        client, _args(action="submit-drafts", ids="36519", dry_run=True)
    )

    assert result["dry_run"] is True
    assert result["drafts"][0]["days"][0]["date"] == "2026-09-21"
    assert client.calls == []


def test_submit_drafts_refuses_everything_if_one_draft_is_not_saved():
    client = StubClient(drafts=[_draft(), _draft(application_id=36520, day="2026-09-22",
                                                 status=1)])

    with pytest.raises(CommandRefused) as refused:
        take100_api._cmd_submit_drafts(client, _args(action="submit-drafts", ids="36519,36520"))

    assert list(refused.value.payload["problems"]) == ["100-36520"]
    assert client.calls == []


def test_submit_drafts_refuses_a_day_that_is_already_submitted():
    client = StubClient(drafts=[_draft()], rows=[_row("2026-09-21", status="pending",
                                                       no="100-40000")])

    with pytest.raises(CommandRefused) as refused:
        take100_api._cmd_submit_drafts(client, _args(action="submit-drafts", ids="36519"))

    assert "100-40000" in refused.value.payload["problems"]["100-36519"][0]
    assert client.calls == []


def test_submit_drafts_reports_a_draft_it_could_not_delete():
    client = StubClient(drafts=[_draft()], delete_fails=True)

    result = take100_api._cmd_submit_drafts(client, _args(action="submit-drafts", ids="36519"))

    assert result["success"] is True
    assert result["results"][0]["draft_deleted"] is False
    assert "100-36519" in result["results"][0]["warning"]


def test_submit_drafts_all_saved_uses_the_saved_tab_in_range():
    client = StubClient(drafts=[_draft()], rows=[_row("2026-09-21", no="100-36519"),
                                                 _row("2026-08-01", no="100-30000")])

    result = take100_api._cmd_submit_drafts(
        client, _args(action="submit-drafts", all_saved=True, dry_run=True)
    )

    assert [draft["draft"] for draft in result["drafts"]] == ["100-36519"]
