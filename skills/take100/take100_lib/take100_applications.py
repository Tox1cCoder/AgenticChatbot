"""
take100_applications.py — leave and overtime applications on take100dot.com.

take100 calls these "other applications": overtime (type 2) and paid leave (type 3,
by the hour or by the day). A late arrival is an hourly paid leave starting at 08:00.
The requests mirror what the site's own pages send
(js/other-application-management/create.js):

    create        POST   /api/other-applications          multipart form, as the
                                                          create page's FormData
    submit draft  PUT    /api/other-applications/{id}     JSON, as the detail page
    delete        DELETE /api/other-applications/{id}
    list          GET    /other-applications?fromDate=&toDate=&statusId=&applicationTypeId=

The create page's own checks (time windows, locked dates, paid-leave balance) run
before anything is sent, because the site only enforces some of them in the browser.

Usage — through take100_api's CLI:
    --action list-applications [--from --to --app-status --app-type]
    --action file-applications --applications '[{...}, ...]' [--submit] [--dry-run]
    --action submit-application-drafts --ids 36700,36701 | --all-saved [--dry-run]
    --action delete-application --application-id 36700

Application JSON format:
    {"kind": "hour-leave", "date": "2026-10-16", "from": "08:00", "to": "08:51",
     "reason": "Lý do sức khỏe"}
    {"kind": "overtime", "date": "2026-10-14", "from": "17:00", "to": "18:00",
     "reason": "Fix Metadata Streaming Backport"}
    {"kind": "day-leave", "date": "2026-10-07", "to_date": "2026-10-07", "reason": "..."}
Optional keys: "message" (text), "priority" (true = High).
"""

import argparse
import json
import re
from datetime import date, datetime, timedelta

import requests

from take100_lib.take100_api import (
    BASE_URL,
    DEFAULT_TIME_IN,
    DEFAULT_TIME_OUT,
    LIST_PAGE_LIMIT,
    LUNCH_END,
    LUNCH_START,
    REASON_ID_ALL_DAY_LEAVE,
    REASON_ID_HOUR_LEAVE,
    CommandRefused,
    Take100Client,
    _api_referer,
    _list_range,
    from_minutes,
    to_minutes,
)

CREATE_PAGE_URL = f"{BASE_URL}/other-applications/create"
API_URL = f"{BASE_URL}/api/other-applications"
LIST_URL = f"{BASE_URL}/other-applications"

# The application type <select> on the create page.
APP_TYPE_IDS = {"overtime": 2, "paid-leave": 3}
KIND_APP_TYPE = {"overtime": 2, "hour-leave": 3, "day-leave": 3}
KINDS = tuple(KIND_APP_TYPE)
# The status filter on /other-applications.
APP_STATUS_IDS = {"draft": 0, "in_progress": 1, "approved": 2, "denied": -1}
STATUS_LABELS = {
    "draft": "draft",
    "in progress": "in_progress",
    "approved": "approved",
    "be denied": "denied",
    "denied": "denied",
}

WEEKEND_DAYS = (5, 6)
HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

# Observed in every OT application on file: the filed end is the check-out rounded
# down to 5 minutes (check-out 19:36 -> OT until 19:35). Late leave runs to the
# check-in minute itself (check-in 08:51:03 -> leave 08:00-08:51).
OT_ROUNDING_MINUTES = 5


def floor_to_step(hhmm: str, step: int = OT_ROUNDING_MINUTES) -> str:
    """'19:36:38' -> '19:35' for a 5-minute step."""
    minutes = to_minutes(hhmm)
    return from_minutes(minutes - minutes % step)


def leave_minutes(from_time: str, to_time: str) -> int:
    """
    Paid-leave minutes an hourly leave consumes, lunch excluded.

    A port of calTimeForPaidLeaveHourADay in the site's create.js, which is how the
    site sizes an hourly leave against the remaining balance.
    """
    start, end = to_minutes(from_time), to_minutes(to_time)
    lunch_start, lunch_end = to_minutes(LUNCH_START), to_minutes(LUNCH_END)

    def on_lunch(value: int) -> bool:
        return lunch_start <= value <= lunch_end

    if on_lunch(start):
        return 0 if on_lunch(end) else end - lunch_end
    if on_lunch(end):
        return lunch_start - start
    if start < lunch_start and end > lunch_end:
        return end - start - (lunch_end - lunch_start)
    return end - start


def _weekdays(first: str, last: str) -> int:
    day, end, count = date.fromisoformat(first), date.fromisoformat(last), 0
    while day <= end:
        if day.weekday() not in WEEKEND_DAYS:
            count += 1
        day += timedelta(days=1)
    return count


def _is_weekend_range(first: str, last: str) -> bool:
    return _weekdays(first, last) == 0


def parse_specs(raw: str | None) -> list[dict]:
    """Normalise the --applications JSON into complete application specs."""
    if not raw:
        raise CommandRefused({"error": "--applications is required for file-applications"})
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CommandRefused({"error": f"--applications is not valid JSON: {exc}"}) from None
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list) or not items:
        raise CommandRefused({"error": "--applications must be a non-empty JSON array"})

    specs = []
    for item in items:
        if not isinstance(item, dict):
            raise CommandRefused({"error": "Each application must be a JSON object.", "item": item})
        specs.append(
            {
                "kind": item.get("kind"),
                "date": item.get("date"),
                "to_date": item.get("to_date") or item.get("date"),
                "from": item.get("from"),
                "to": item.get("to"),
                "reason": (item.get("reason") or "").strip(),
                "message": (item.get("message") or "").strip(),
                "priority": bool(item.get("priority")),
            }
        )
    return specs


def _shape_problems(spec: dict) -> list[str]:
    """Problems with a spec's own fields; empty when it is well formed."""
    if spec["kind"] not in KINDS:
        return [f"kind must be one of {', '.join(KINDS)}, not {spec['kind']!r}."]
    problems = []
    for field in ("date", "to_date"):
        try:
            datetime.strptime(str(spec[field]), "%Y-%m-%d")
        except ValueError:
            problems.append(f"{field} must be YYYY-MM-DD, not {spec[field]!r}.")
    if spec["kind"] != "day-leave":
        for field in ("from", "to"):
            if not HHMM.match(str(spec[field] or "")):
                problems.append(f"{field} must be HH:MM, not {spec[field]!r}.")
    if not spec["reason"]:
        problems.append("reason is required; ask the user for it.")
    return problems


def _window_problems(spec: dict) -> list[str]:
    """The create page's time-window rules for one well-formed spec."""
    if spec["kind"] == "day-leave":
        if spec["to_date"] < spec["date"]:
            return [f"to_date {spec['to_date']} is before date {spec['date']}."]
        return []

    start, end = to_minutes(spec["from"]), to_minutes(spec["to"])
    shift_start, shift_end = to_minutes(DEFAULT_TIME_IN), to_minutes(DEFAULT_TIME_OUT)
    if end <= start:
        return [f"to {spec['to']} must be after from {spec['from']}."]

    if spec["kind"] == "hour-leave":
        if not (shift_start <= start <= shift_end and shift_start <= end <= shift_end):
            return [f"Hourly leave must lie within {DEFAULT_TIME_IN}-{DEFAULT_TIME_OUT}."]
        return []

    if _is_weekend_range(spec["date"], spec["to_date"]):
        return []
    problems = []
    if shift_start < start < shift_end:
        problems.append(f"Weekday OT from {spec['from']} falls inside the shift.")
    if shift_start < end < shift_end:
        problems.append(f"Weekday OT to {spec['to']} falls inside the shift.")
    if start <= shift_start and end >= shift_end:
        problems.append("Weekday OT cannot span the whole shift.")
    return problems


def spec_minutes(spec: dict) -> int:
    """Paid-leave minutes a spec consumes; 8 hours per weekday for a day leave."""
    if spec["kind"] == "hour-leave":
        return leave_minutes(spec["from"], spec["to"])
    if spec["kind"] == "day-leave":
        return _weekdays(spec["date"], spec["to_date"]) * 8 * 60
    return 0


def _balance_problems(specs: list[dict], balance: dict) -> list[str]:
    """The create page's paid-leave balance checks, summed over the whole batch."""
    hourly = sum(spec_minutes(spec) for spec in specs if spec["kind"] == "hour-leave")
    total = sum(spec_minutes(spec) for spec in specs)
    problems = []
    if balance.get("remaining_hourly_minutes") is not None and hourly > balance[
        "remaining_hourly_minutes"
    ]:
        problems.append(
            f"Hourly leave in this batch ({hourly} min) exceeds the remaining hourly "
            f"allowance ({balance['remaining_hourly_minutes']} min)."
        )
    if balance.get("remaining_leave_minutes") is not None and total > balance[
        "remaining_leave_minutes"
    ]:
        problems.append(
            f"Paid leave in this batch ({total} min) exceeds the remaining paid leave "
            f"({balance['remaining_leave_minutes']} min)."
        )
    return problems


def _overlaps(spec: dict, row: dict) -> bool:
    return bool(row["date_from"]) and row["date_from"] <= spec["to_date"] and (
        row["date_to"] or row["date_from"]
    ) >= spec["date"]


def _duplicate_problems(spec: dict, rows: list[dict]) -> list[str]:
    """Non-denied applications of the same kind already covering the spec's dates."""
    clashes = [
        row
        for row in rows
        if row["kind"] == spec["kind"] and row["status"] != "denied" and _overlaps(spec, row)
    ]
    return [
        f"Application {row['no']} ({row['type']}, {row['status'].replace('_', ' ')}) "
        f"already covers {row['apply_date']}."
        for row in clashes
    ]


def plan_problems(specs: list[dict], context: dict, rows: list[dict],
                  allow_duplicate: bool = False) -> dict[str, list[str]]:
    """Every problem in the batch, keyed '<index> <kind> <date>'; empty means all clear."""
    problems: dict[str, list[str]] = {}
    seen: set = set()
    for index, spec in enumerate(specs, start=1):
        key = f"{index} {spec['kind']} {spec['date']}"
        found = _shape_problems(spec)
        if not found:
            found = _window_problems(spec)
            locked = context.get("locked_date")
            if locked and spec["date"] < locked:
                found.append(f"{spec['date']} is before the locked date {locked}.")
            if not allow_duplicate:
                found.extend(_duplicate_problems(spec, rows))
                if (spec["kind"], spec["date"]) in seen:
                    found.append("The batch lists this kind for the same date twice.")
            seen.add((spec["kind"], spec["date"]))
        if found:
            problems[key] = found

    well_formed = [spec for spec in specs if not _shape_problems(spec)]
    balance = _balance_problems(well_formed, context.get("balance") or {})
    if balance:
        problems["balance"] = balance
    return problems


def build_form(spec: dict, action_type: str, token: str) -> dict[str, str]:
    """
    The fields the create page's FormData posts for this spec.

    The browser sends every enabled input, hidden or not: the date-range inputs go
    empty for an hourly leave and the single 'date' input goes empty otherwise. The
    reason radios are left unchecked for overtime, and an unchecked 'priority'
    checkbox is not sent at all.
    """
    kind = spec["kind"]
    fields = {"_token": token}
    if kind == "hour-leave":
        fields["reason_id"] = str(REASON_ID_HOUR_LEAVE)
    elif kind == "day-leave":
        fields["reason_id"] = str(REASON_ID_ALL_DAY_LEAVE)

    is_hourly = kind == "hour-leave"
    has_times = kind != "day-leave"
    fields.update(
        {
            "from_date": "" if is_hourly else spec["date"],
            "to_date": "" if is_hourly else spec["to_date"],
            "date": spec["date"] if is_hourly else "",
            "from_time": spec["from"] if has_times else "",
            "to_time": spec["to"] if has_times else "",
        }
    )
    if spec["priority"]:
        fields["priority"] = "1"
    fields.update(
        {
            "reason": spec["reason"],
            "message": spec["message"],
            "application_type_id": str(KIND_APP_TYPE[kind]),
            "action_type": action_type,
        }
    )
    return fields


def _js_value(html: str, name: str):
    """A ``const name = <json>;`` value embedded in a page, or None."""
    match = re.search(rf"(?:const|let|var)\s+{name}\s*=\s*", html)
    if not match:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(html, match.end())
    except json.JSONDecodeError:
        return None
    return value


def parse_create_context(html: str) -> dict:
    """CSRF token, locked date and paid-leave balance from the create page."""
    token = re.search(r'name="_token" value="([^"]+)"', html)
    hours_left = _js_value(html, "paidLeaveHoursLeft")
    remaining_days = _js_value(html, "remainingPaidLeaveDay")
    remaining_hours = _js_value(html, "remainingPaidLeaveHour")

    leave_minutes_left = None
    if remaining_days is not None:
        extra = to_minutes(hours_left) if isinstance(hours_left, str) and ":" in hours_left else 0
        leave_minutes_left = int(remaining_days * 8 * 60) + extra
    return {
        "token": token.group(1) if token else None,
        "locked_date": _js_value(html, "lockedDate"),
        "balance": {
            "remaining_leave_days": remaining_days,
            "remaining_leave_hours_extra": hours_left,
            "remaining_leave_minutes": leave_minutes_left,
            "remaining_hourly_minutes": (
                round(remaining_hours * 60) if remaining_hours is not None else None
            ),
        },
    }


def _kind_from_type(type_text: str) -> str:
    text = type_text.strip().lower()
    if text == "over time":
        return "overtime"
    if text.endswith("hour"):
        return "hour-leave"
    if text.endswith("all day"):
        return "day-leave"
    return "other-leave"


def _apply_dates(text: str, created: str | None) -> tuple[str | None, str | None]:
    """
    '10-16 (Fri)' -> ('2026-10-16', '2026-10-16'). The list omits the year, so it is
    taken from the created date, stepping back a year when that lands too far ahead.
    """
    found = re.findall(r"(\d{2})-(\d{2})", text)
    if not found:
        return None, None
    year = int(created[:4]) if created and created[:4].isdigit() else date.today().year
    anchor = date.fromisoformat(created) if created else date.today()

    def resolve(month: str, day: str) -> str:
        candidate = date(year, int(month), int(day))
        if candidate - anchor > timedelta(days=183):
            candidate = candidate.replace(year=year - 1)
        return candidate.isoformat()

    return resolve(*found[0]), resolve(*found[-1])


def parse_application_rows(html: str) -> list[dict]:
    """Rows of the /other-applications table, keyed by the application id."""
    from bs4 import BeautifulSoup

    rows = []
    for row in BeautifulSoup(html, "html.parser").select("tr.row-item[data-id]"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        if len(cells) < 9:
            continue
        created = cells[6] or None
        date_from, date_to = _apply_dates(cells[5], created)
        status_text = re.sub(r"\s+", " ", cells[3]).strip().lower()
        rows.append(
            {
                "id": row["data-id"],
                "no": cells[1],
                "priority": cells[2],
                "status": STATUS_LABELS.get(status_text, status_text),
                "type": cells[4],
                "kind": _kind_from_type(cells[4]),
                "apply_date": cells[5],
                "date_from": date_from,
                "date_to": date_to,
                "created": created,
                "progress": re.sub(r"\s+", "", cells[7]),
                "reason": cells[8],
                "message": cells[9] if len(cells) > 9 else "",
            }
        )
    return rows


def parse_detail_form(html: str) -> dict[str, str]:
    """
    The detail page form as its FormData would serialise it: enabled named inputs,
    only the checked radio and checkbox, values trimmed.
    """
    from bs4 import BeautifulSoup

    form = BeautifulSoup(html, "html.parser").find(id="viewInDetailOtherApplicationForm")
    if form is None:
        return {}
    fields: dict[str, str] = {}
    for element in form.find_all(["input", "textarea", "select"]):
        name = element.get("name")
        if not name or element.has_attr("disabled"):
            continue
        if element.get("type") in ("radio", "checkbox") and not element.has_attr("checked"):
            continue
        if element.name == "textarea":
            value = element.get_text()
        elif element.name == "select":
            chosen = element.find("option", selected=True) or element.find("option")
            value = chosen.get("value", "") if chosen else ""
        else:
            value = element.get("value", "")
        fields[name] = value.strip()
    return fields


def _error_details(exc: requests.HTTPError) -> dict:
    """The site's own validation message (user_res_msg) when it sent one."""
    response = exc.response
    details: dict = {"error": f"HTTP {response.status_code}"}
    try:
        body = response.json()
    except ValueError:
        details["details"] = response.text[:300]
        return details
    details["details"] = body.get("user_res_msg") or body.get("message") or body
    return details


class ApplicationsApi:
    """The other-applications pages and API, on an authenticated Take100Client."""

    def __init__(self, client: Take100Client):
        self.client = client

    def create_context(self) -> dict:
        response = self.client._request("GET", CREATE_PAGE_URL)
        response.raise_for_status()
        context = parse_create_context(response.text)
        if not context["token"]:
            raise RuntimeError("The create page has no CSRF token; has the site changed?")
        return context

    def list_rows(self, date_from: str | None = None, date_to: str | None = None,
                  status: str = "all", app_type: str = "all") -> list[dict]:
        params = {
            "statusId": APP_STATUS_IDS.get(status, "all"),
            "applicationTypeId": APP_TYPE_IDS.get(app_type, "all"),
        }
        if date_from and date_to:
            params.update({"fromDate": date_from, "toDate": date_to})
        rows: list[dict] = []
        for page in range(1, LIST_PAGE_LIMIT + 1):
            response = self.client._request("GET", LIST_URL, params={**params, "page": page})
            response.raise_for_status()
            page_rows = parse_application_rows(response.text)
            rows.extend(page_rows)
            if not page_rows or f"page={page + 1}" not in response.text:
                break
        return rows

    def create(self, form: dict[str, str]) -> dict:
        response = self.client._request(
            "POST",
            API_URL,
            # Multipart, as the page's FormData; None drops the client's JSON default.
            files={name: (None, value) for name, value in form.items()},
            headers={"Content-Type": None, **_api_referer(CREATE_PAGE_URL)},
        )
        response.raise_for_status()
        return response.json() if response.text else {}

    def submit_draft(self, application_id: str, kind: str) -> dict:
        detail_url = f"{LIST_URL}/{application_id}"
        page = self.client._request("GET", detail_url)
        page.raise_for_status()
        fields = parse_detail_form(page.text)
        if not fields:
            raise RuntimeError(f"Application {application_id} has no editable form.")
        body = {
            **fields,
            "_method": "PUT",
            "application_type_id": str(KIND_APP_TYPE.get(kind, APP_TYPE_IDS["paid-leave"])),
            "action_type": "submit",
            "id": str(application_id),
        }
        response = self.client._request(
            "PUT", f"{API_URL}/{application_id}", json=body, headers=_api_referer(detail_url)
        )
        response.raise_for_status()
        return response.json() if response.text else {}

    def delete(self, application_id: int) -> dict:
        response = self.client._request(
            "DELETE",
            f"{API_URL}/{application_id}",
            headers=_api_referer(f"{LIST_URL}/{application_id}"),
        )
        response.raise_for_status()
        return response.json() if response.text else {"status": "deleted"}


def _with_times(rows: list[dict], periods: list) -> list[dict]:
    """Add filed times from /my-workingtimes, which embeds them for both periods."""
    details = {
        str(app.get("application_id")): app for _period, apps in periods for app in apps
    }
    enriched = []
    for row in rows:
        app = details.get(row["id"])
        if app:
            row = {
                **row,
                "date_from": app.get("from_date") or row["date_from"],
                "date_to": app.get("to_date") or row["date_to"],
                "from_time": app.get("from_time") if row["kind"] != "day-leave" else None,
                "to_time": app.get("to_time") if row["kind"] != "day-leave" else None,
            }
        enriched.append(row)
    return enriched


def file_applications(api, specs: list[dict], *, submit: bool, dry_run: bool,
                      allow_duplicate: bool = False) -> dict:
    """Validate the whole batch, then send it; nothing is sent if any spec is wrong."""
    context = api.create_context()
    well_formed = [spec for spec in specs if not _shape_problems(spec)]
    rows = []
    if well_formed:
        rows = api.list_rows(
            min(spec["date"] for spec in well_formed),
            max(spec["to_date"] for spec in well_formed),
        )
    problems = plan_problems(specs, context, rows, allow_duplicate)
    if problems:
        raise CommandRefused(
            {
                "error": "Some applications cannot be filed. Nothing was sent.",
                "problems": problems,
                "balance": context["balance"],
                "hint": "Fix or drop those applications and run again.",
            }
        )

    action_type = "submit" if submit else "save"
    if dry_run:
        return {"success": True, "action": "file-applications", "dry_run": True,
                "action_type": action_type, "applications": specs,
                "balance": context["balance"]}

    results = []
    for spec in specs:
        result = {"kind": spec["kind"], "date": spec["date"]}
        try:
            response = api.create(build_form(spec, action_type, context["token"]))
            results.append({**result, "success": True, "response": response})
        except requests.HTTPError as exc:
            results.append({**result, "success": False, **_error_details(exc)})
    return {"success": all(item["success"] for item in results),
            "action": "file-applications", "action_type": action_type, "results": results}


def submit_application_drafts(api, ids: list[str] | None, date_range: tuple | None,
                              dry_run: bool) -> dict:
    """Submit saved draft applications through the detail page's own PUT request."""
    if ids is None and date_range is None:
        raise CommandRefused({"error": "Pass --ids <id,id,...> or --all-saved"})
    if date_range is None:
        drafts = api.list_rows(status="draft")
    else:
        drafts = api.list_rows(date_range[0], date_range[1], status="draft")
    by_id = {row["id"]: row for row in drafts}

    if ids is not None:
        missing = [value for value in ids if value not in by_id]
        if missing:
            raise CommandRefused(
                {"error": "Not saved drafts (already submitted, or not yours). Nothing was sent.",
                 "ids": missing}
            )
        chosen = [by_id[value] for value in ids]
    else:
        chosen = drafts
    if not chosen:
        raise CommandRefused({"error": "No draft applications found to submit."})

    if dry_run:
        return {"success": True, "action": "submit-application-drafts", "dry_run": True,
                "drafts": chosen}
    results = []
    for row in chosen:
        try:
            response = api.submit_draft(row["id"], row["kind"])
            results.append({"id": row["id"], "no": row["no"], "success": True,
                            "response": response})
        except requests.HTTPError as exc:
            results.append({"id": row["id"], "no": row["no"], "success": False,
                            **_error_details(exc)})
    return {"success": all(item["success"] for item in results),
            "action": "submit-application-drafts", "results": results}


def _cmd_list_applications(client: Take100Client, args: argparse.Namespace) -> dict:
    date_from, date_to, periods = _list_range(client, args)
    api = ApplicationsApi(client)
    rows = api.list_rows(date_from, date_to, args.app_status, args.app_type)
    return {
        "success": True,
        "range": {"from": date_from, "to": date_to},
        "applications": sorted(_with_times(rows, periods), key=lambda row: row["date_from"] or ""),
        "balance": api.create_context()["balance"],
    }


def _cmd_file_applications(client: Take100Client, args: argparse.Namespace) -> dict:
    return file_applications(
        ApplicationsApi(client),
        parse_specs(args.applications),
        submit=args.submit,
        dry_run=args.dry_run,
        allow_duplicate=args.allow_duplicate,
    )


def _cmd_submit_application_drafts(client: Take100Client, args: argparse.Namespace) -> dict:
    ids = [value.strip() for value in args.ids.split(",") if value.strip()] if args.ids else None
    date_range = None
    if ids is None and args.all_saved:
        date_from, date_to, _ = _list_range(client, args)
        date_range = (date_from, date_to)
    return submit_application_drafts(ApplicationsApi(client), ids, date_range, args.dry_run)


def _cmd_delete_application(client: Take100Client, args: argparse.Namespace) -> dict:
    if not args.application_id:
        raise CommandRefused({"error": "--application-id is required for delete-application"})
    response = ApplicationsApi(client).delete(args.application_id)
    return {"success": True, "action": "delete-application", "response": response}


COMMANDS = {
    "list-applications": _cmd_list_applications,
    "file-applications": _cmd_file_applications,
    "submit-application-drafts": _cmd_submit_application_drafts,
    "delete-application": _cmd_delete_application,
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--applications",
        help="file-applications: JSON array of {kind, date, from, to, reason} objects; "
        "kind is overtime, hour-leave or day-leave",
    )
    parser.add_argument(
        "--allow-duplicate",
        action="store_true",
        help="file-applications: file even when one of the same kind covers the date",
    )
    parser.add_argument(
        "--app-status",
        choices=["all", *APP_STATUS_IDS],
        default="all",
        help="list-applications: status filter (default: all)",
    )
    parser.add_argument(
        "--app-type",
        choices=["all", *APP_TYPE_IDS],
        default="all",
        help="list-applications: type filter (default: all)",
    )
