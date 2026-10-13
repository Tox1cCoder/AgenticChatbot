"""
take100_api.py — Direct API client for take100dot.com timesheet management.

This script provides a reliable, token-efficient alternative to browser automation
for filling timesheets on take100dot.com. It uses direct HTTP API calls instead of
driving a browser via playwright-cli.

Late arrivals and overtime are detected automatically: save/submit look up the day's
applications on take100dot.com and derive time_in / time_out from them, then refuse to
send entries that do not match that working window.

Usage — the skill runtime runs ``scripts/take100.py`` with argv and no shell, so
Unicode entries can be passed inline:
    --action list                                   timesheets + weekdays still missing
    --action check-day --date 2026-03-03            the day's working window
    --action save --date 2026-03-03 --entries '[...]'      draft (submit: --action submit)
    --action save-many --days '[{"date", "entries"}, ...]' [--submit]
    --action submit-drafts --ids 36519,36520 | --all-saved [--dry-run]
    --action delete --application-id 30459

    # Escape hatches
    --time-in / --time-out   override the detected window
    --no-auto                skip detection entirely, use 08:00-17:00
    --skip-validation        send entries even if they do not tile the window
    --entries-file PATH      for shells that corrupt non-ASCII arguments

Entry JSON format:
    [
        {
            "from": "08:00",
            "to": "08:05",
            "project_id": "K202201",
            "work_item_id": 122,
            "work_content": "Tập thể dục, báo cáo buổi sáng"
        },
        ...
    ]
"""

import argparse
import contextlib
import json
import os
import re
import sys
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

import requests

BASE_URL = "https://take100dot.com"
DEFAULT_SESSION_CACHE_PATH = str(Path.home() / ".cache" / "take100" / "session.json")

# Default shift work
DEFAULT_SHIFT_WORK_ID = "C001"  # 08:00-17:00
DEFAULT_TIME_IN = "08:00"
DEFAULT_TIME_OUT = "17:00"
LUNCH_START = "12:00"
LUNCH_END = "13:00"

# Application semantics, read from take100dot.com's own js/workingtime_datatable.js
APP_STATUS_DENIED = -1
APP_STATUS_IN_PROGRESS = 1
APP_STATUS_FINISHED = 2

REASON_ID_ALL_DAY_LEAVE = 1
REASON_ID_HOUR_LEAVE = 5
REASON_ID_OVERTIME = 6

# Observed convention: paid leave until 08:48 is filed with time_in 08:49 — work
# resumes the minute after the leave window closes.
LATE_RESUME_OFFSET_MINUTES = 1

# /my-workingtimes ignores ?from=&to= and serves a payroll period (21st -> 20th)
# chosen by ?month=. These are the only two values its month selector offers.
AVAILABLE_PERIODS = ("this_month", "last_month")

# The status tabs on /wt-applications (getByStatus(n) in wt-application-management.js).
LIST_STATUSES = {"saved": 0, "pending": 1, "finished": 2, "denied": -1}
# Statuses that mean a timesheet is already on file for its dates.
ON_FILE_STATUSES = ("saved", "pending", "finished")
# Upper bound on pages read per tab; the finished tab alone holds hundreds of rows.
LIST_PAGE_LIMIT = 20


def to_minutes(hhmm: str) -> int:
    """Convert 'HH:MM' (or 'HH:MM:SS') to minutes since midnight."""
    hours, minutes = hhmm[:5].split(":")
    return int(hours) * 60 + int(minutes)


def from_minutes(total: int) -> str:
    """Convert minutes since midnight to 'HH:MM'."""
    return f"{total // 60:02d}:{total % 60:02d}"


def hhmm(value: str | None) -> str | None:
    """Normalise 'HH:MM:SS' to 'HH:MM'; pass through None."""
    return value[:5] if value else None


def _classify_application(app: dict) -> str | None:
    """
    Map an application to the role it plays in a timesheet.

    Roles: 'all_day_leave', 'late', 'early_leave', 'mid_day_leave', 'overtime'.
    Returns None for applications that do not affect the working window.
    """
    type_name = app.get("application_type_name")
    reason_id = app.get("reason_id")

    if reason_id == REASON_ID_OVERTIME or type_name == "over time":
        return "overtime"

    if type_name != "paid leave":
        return None

    if reason_id == REASON_ID_ALL_DAY_LEAVE or app.get("reason_name") == "All day":
        return "all_day_leave"

    if reason_id != REASON_ID_HOUR_LEAVE and app.get("reason_name") != "Hour":
        return None

    from_time, to_time = app.get("from_time"), app.get("to_time")
    if not from_time or not to_time or to_minutes(to_time) <= to_minutes(from_time):
        return None

    # A late arrival eats into the start of the shift; an early leave runs to its end.
    # Anything else is an errand in the middle of the day and must not move time_in/out.
    if hhmm(from_time) == DEFAULT_TIME_IN:
        return "late"
    if hhmm(to_time) == DEFAULT_TIME_OUT:
        return "early_leave"
    return "mid_day_leave"


def _select_effective_applications(day_apps: list) -> tuple[dict, list]:
    """
    Reduce a day's applications to the one that counts per role.

    Re-filing is how corrections are made on take100dot.com (see getLastCreatedApps in
    the site's own workingtime_datatable.js), so the newest application wins. Denied
    ones are skipped, which lets an older still-valid application stand.

    Returns (effective_by_role, roles_where_every_application_was_denied).
    """
    ordered = sorted(day_apps, key=lambda app: app.get("created_at", ""), reverse=True)

    effective: dict = {}
    mid_day: list = []
    seen_roles: set = set()
    denied_only: dict = {}

    for app in ordered:
        role = _classify_application(app)
        if role is None:
            continue

        if role == "mid_day_leave":
            if app.get("status") != APP_STATUS_DENIED:
                mid_day.append(app)
            continue

        if role in seen_roles:
            continue

        if app.get("status") == APP_STATUS_DENIED:
            denied_only.setdefault(role, {**app, "role": role})
            continue

        seen_roles.add(role)
        effective[role] = app
        denied_only.pop(role, None)

    effective["mid_day_leave_list"] = mid_day
    return effective, list(denied_only.values())


def validate_entries(entries: list, time_in: str, time_out: str) -> list[str]:
    """
    Check that entries tile the working window exactly, with only lunch left out.

    Returns a list of human-readable problems; empty means the entries are consistent.
    """
    if not entries:
        return ["No entries provided."]

    problems = []
    rows = sorted(entries, key=lambda e: to_minutes(e["from"]))

    for entry in rows:
        if to_minutes(entry["to"]) <= to_minutes(entry["from"]):
            problems.append(f"Entry {entry['from']}-{entry['to']} does not move forward in time.")
        if to_minutes(entry["from"]) < to_minutes(LUNCH_END) and to_minutes(
            entry["to"]
        ) > to_minutes(LUNCH_START):
            problems.append(
                f"Entry {entry['from']}-{entry['to']} overlaps the "
                f"{LUNCH_START}-{LUNCH_END} lunch break."
            )

    if rows[0]["from"] != time_in:
        problems.append(f"First entry starts at {rows[0]['from']} but time_in is {time_in}.")

    last_end = max(entry["to"] for entry in rows)
    if last_end != time_out:
        problems.append(f"Last entry ends at {last_end} but time_out is {time_out}.")

    for previous, current in zip(rows, rows[1:], strict=False):
        if to_minutes(current["from"]) < to_minutes(previous["to"]):
            problems.append(
                f"Entries {previous['from']}-{previous['to']} and "
                f"{current['from']}-{current['to']} overlap."
            )
        elif to_minutes(current["from"]) > to_minutes(previous["to"]) and (
            previous["to"],
            current["from"],
        ) != (LUNCH_START, LUNCH_END):
            problems.append(f"Gap {previous['to']}-{current['from']} is not the lunch break.")

    return problems


def _api_referer(page_url: str) -> dict:
    """
    /api/* routes accept the session cookie only from a same-site page (Laravel
    Sanctum's stateful check): without a Referer they answer 401 "not authenticated".
    """
    return {"Referer": page_url}


def build_working_day(date: str, entries: list, time_in: str, time_out: str) -> dict:
    """One ``working_days`` item in the shape the create page posts."""
    return {
        "date": f"{date}T00:00:00.000Z",
        "time_in": time_in,
        "time_out": time_out,
        "break_start": None,
        "break_end": None,
        "shift_work_id": DEFAULT_SHIFT_WORK_ID,
        "working_day_projects": [
            {
                "project_id": entry["project_id"],
                "work_content": entry["work_content"],
                "from": entry["from"],
                "to": entry["to"],
                "work_item_id": entry["work_item_id"],
            }
            for entry in entries
        ],
    }


def draft_days(application: dict) -> list[dict]:
    """A saved application's days as {date, time_in, time_out, shift_work_id, entries}."""
    days = []
    for day in application.get("working_days") or []:
        days.append(
            {
                "date": day["date"][:10],
                "time_in": hhmm(day.get("time_in")),
                "time_out": hhmm(day.get("time_out")),
                "shift_work_id": day.get("shift_work_id") or DEFAULT_SHIFT_WORK_ID,
                "entries": [
                    {
                        "from": hhmm(project["from"]),
                        "to": hhmm(project["to"]),
                        "project_id": project["project_id"],
                        "work_item_id": project["work_item_id"],
                        "work_content": project["work_content"],
                    }
                    for project in day.get("working_day_projects") or []
                ],
            }
        )
    return days


def _parse_list_date(text: str) -> str | None:
    """'Oct 8, 2026' -> '2026-10-08'; None for anything else."""
    try:
        return datetime.strptime(text.strip(), "%b %d, %Y").date().isoformat()
    except ValueError:
        return None


def parse_timesheet_rows(html: str, status: str) -> list[dict]:
    """Rows of the /wt-applications table: No | Id | Application | Requested | Process."""
    from bs4 import BeautifulSoup

    rows = []
    for row in BeautifulSoup(html, "html.parser").select("#products-list tbody tr"):
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cells) < 5:
            continue
        start, _, end = cells[2].partition("~")
        link = row.find("a", href=True)
        url = link["href"].strip() if link else ""
        rows.append(
            {
                "id": url.rstrip("/").rsplit("/", 1)[-1] if url else "",
                "no": cells[1],
                "status": status,
                "date_from": _parse_list_date(start),
                "date_to": _parse_list_date(end or start),
                "requested_time": cells[3] or None,
                "process": cells[4],
            }
        )
    return rows


def summarize_day(date: str, period: dict | None, applications: list) -> dict:
    """
    Reduce one payroll period's applications to a date's working window.

    ``period`` is None when the date lies outside every period the site exposes.
    """
    if period is None:
        return {
            "date": date,
            "data_available": False,
            "error": (
                f"{date} is outside the two payroll periods take100dot.com exposes "
                f"(this_month / last_month). Late and overtime cannot be checked for it."
            ),
            "is_all_day_leave": False,
            "is_late": False,
            "is_early_leave": False,
            "is_overtime": False,
            "late_until": None,
            "early_from": None,
            "ot_from": None,
            "ot_to": None,
            "ot_reason": None,
            "recommended_time_in": DEFAULT_TIME_IN,
            "recommended_time_out": DEFAULT_TIME_OUT,
            "warnings": [],
            "leave_apps": [],
        }

    day_apps = [
        app
        for app in applications
        if app.get("from_date", "") <= date <= app.get("to_date", app.get("from_date", ""))
    ]
    effective, denied = _select_effective_applications(day_apps)

    all_day = effective.get("all_day_leave")
    late = effective.get("late")
    early = effective.get("early_leave")
    overtime = effective.get("overtime")

    warnings = []
    for app in denied:
        warnings.append(
            f"All {app['role'].replace('_', ' ')} applications for {date} were denied "
            f"({app['from_time']}-{app['to_time']}); treating the day as if none was filed."
        )
    for app in effective.get("mid_day_leave_list", []):
        warnings.append(
            f"Hourly leave {app['from_time']}-{app['to_time']} is mid-shift, not a late "
            f"arrival or early leave — adjust entries manually to leave that window out."
        )

    late_until = hhmm(late["to_time"]) if late else None
    early_from = hhmm(early["from_time"]) if early else None
    ot_from = hhmm(overtime["from_time"]) if overtime else None
    ot_to = hhmm(overtime["to_time"]) if overtime else None

    if overtime and to_minutes(ot_to) <= to_minutes(ot_from):
        warnings.append(
            f"Overtime window {ot_from}-{ot_to} does not move forward (overnight OT is not "
            f"supported); pass --time-out explicitly."
        )
        ot_to = None

    recommended_time_in = DEFAULT_TIME_IN
    if late_until:
        recommended_time_in = from_minutes(to_minutes(late_until) + LATE_RESUME_OFFSET_MINUTES)

    recommended_time_out = DEFAULT_TIME_OUT
    if early_from and ot_to:
        warnings.append(
            f"Conflicting applications: early leave from {early_from} and overtime until "
            f"{ot_to}. Pass --time-out explicitly to resolve."
        )
        recommended_time_out = None
    elif early_from:
        recommended_time_out = early_from
    elif ot_to:
        recommended_time_out = ot_to

    return {
        "date": date,
        "data_available": True,
        "period": period,
        "is_all_day_leave": all_day is not None,
        "is_late": late is not None,
        "is_early_leave": early is not None,
        "is_overtime": overtime is not None and ot_to is not None,
        "late_until": late_until,
        "early_from": early_from,
        "ot_from": ot_from,
        "ot_to": ot_to,
        "ot_reason": overtime.get("reason") if overtime else None,
        "recommended_time_in": recommended_time_in,
        "recommended_time_out": recommended_time_out,
        "warnings": warnings,
        "leave_apps": [
            app for app in day_apps if app.get("application_type_name") == "paid leave"
        ],
        "applications": day_apps,
    }


class Take100Client:
    """HTTP client for take100dot.com with session-based authentication."""

    def __init__(
        self,
        email: str,
        password: str,
        session_cache_path: str | os.PathLike[str] | None = None,
    ):
        self.session = requests.Session()
        self.email = email
        self.password = password
        self._authenticated = False
        self.session_cache_path = Path(
            session_cache_path
            or os.getenv("TAKE100_SESSION_CACHE_PATH")
            or DEFAULT_SESSION_CACHE_PATH
        )
        self._load_session_cache()

    def _load_session_cache(self) -> None:
        """Restore a persisted cookie jar if it belongs to the same account."""
        with contextlib.suppress(OSError, UnicodeDecodeError, json.JSONDecodeError):
            data = json.loads(self.session_cache_path.read_text(encoding="utf-8"))
            if data.get("email") != self.email:
                return

            cookies = data.get("cookies")
            if not isinstance(cookies, list):
                # Pre-domain cache format; drop it rather than restoring domain-less
                # cookies that would collide with the ones the server sets.
                self._clear_session_cache()
                return

            for cookie in cookies:
                self.session.cookies.set_cookie(requests.cookies.create_cookie(**cookie))

    def _save_session_cache(self) -> None:
        """Persist cookies so the next CLI invocation can reuse the session."""
        self.session_cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "email": self.email,
            # Domain/path must be preserved: restoring bare name/value pairs creates
            # domain-less duplicates that make cookie lookups raise CookieConflictError.
            "cookies": [
                {
                    "name": cookie.name,
                    "value": cookie.value,
                    "domain": cookie.domain,
                    "path": cookie.path,
                    "secure": cookie.secure,
                    "expires": cookie.expires,
                }
                for cookie in self.session.cookies
            ],
        }
        self.session_cache_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _clear_session_cache(self) -> None:
        """Drop persisted session state when auth is no longer valid."""
        with contextlib.suppress(OSError):
            self.session_cache_path.unlink()

    def _has_session_cookies(self) -> bool:
        return any(True for _cookie in self.session.cookies)

    def _response_requires_login(self, response: requests.Response) -> bool:
        """Detect when the server has redirected the request back to login."""
        redirect_url = (response.url or "").rstrip("/")
        return response.status_code in {401, 419} or redirect_url.endswith("/login")

    def _session_is_authenticated(self) -> bool:
        """Validate whether the current cookie jar still grants access."""
        if not self._has_session_cookies():
            return False

        response = self.session.get(f"{BASE_URL}/wt-applications", timeout=15, allow_redirects=True)
        return not self._response_requires_login(response)

    def ensure_authenticated(self, *, validate: bool = True) -> bool:
        """Use persisted cookies when possible, otherwise log in and refresh them."""
        if self._authenticated:
            return True

        if self._has_session_cookies():
            if not validate or self._session_is_authenticated():
                self._authenticated = True
                return True

            self.session.cookies.clear()
            self._clear_session_cache()

        return self.login()

    def _request(
        self,
        method: str,
        url: str,
        *,
        headers: dict | None = None,
        requires_auth: bool = True,
        timeout: int = 15,
        **kwargs,
    ) -> requests.Response:
        """
        Execute a request with one automatic re-authentication attempt.

        Persisted cookies are reused across CLI invocations. If the server redirects
        a protected request back to /login, the client logs in again and retries once.
        """
        custom_headers = dict(headers or {})
        if requires_auth and not self.ensure_authenticated(validate=False):
            raise RuntimeError("Login failed")

        attempts = 2 if requires_auth else 1
        response: requests.Response | None = None
        for attempt in range(attempts):
            request_headers = custom_headers
            if requires_auth:
                request_headers = {**self._get_headers(), **custom_headers}

            response = self.session.request(
                method,
                url,
                headers=request_headers,
                timeout=timeout,
                **kwargs,
            )

            if not requires_auth or not self._response_requires_login(response):
                return response

            self._authenticated = False
            self.session.cookies.clear()
            self._clear_session_cache()

            if attempt == attempts - 1 or not self.login():
                return response

        assert response is not None
        return response

    def login(self) -> bool:
        """Authenticate and establish session cookies. Retries once on failure."""
        for attempt in range(2):
            # GET login page for a fresh CSRF token
            r = self.session.get(f"{BASE_URL}/login", timeout=15)
            token_match = re.search(r'name="_token" value="(.*?)"', r.text)
            if not token_match:
                print("ERROR: Could not find CSRF token on login page.", file=sys.stderr)
                return False

            csrf_token = token_match.group(1)
            login_data = {
                "_token": csrf_token,
                "email": self.email,
                "password": self.password,
                "remember": "on",
            }

            r2 = self.session.post(
                f"{BASE_URL}/login",
                data=login_data,
                timeout=15,
                allow_redirects=True,
            )

            if "/login" not in r2.url:
                self._authenticated = True
                self._save_session_cache()
                return True

            if attempt == 0:
                # Reset cookies and retry once
                self.session.cookies.clear()
                self._clear_session_cache()
            else:
                print("ERROR: Login failed — still on login page.", file=sys.stderr)

        self._authenticated = False
        return False

    def _get_cookie(self, name: str) -> str | None:
        """Read a cookie by name, tolerating duplicates across domains (last wins)."""
        value = None
        for cookie in self.session.cookies:
            if cookie.name == name:
                value = cookie.value
        return value

    def _get_headers(self) -> dict:
        """Build headers with XSRF token for API requests."""
        xsrf_cookie = self._get_cookie("XSRF-TOKEN")
        headers = {
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if xsrf_cookie:
            headers["X-XSRF-TOKEN"] = urllib.parse.unquote(xsrf_cookie)
        return headers

    def _fetch_period(self, month: str) -> tuple[dict | None, list]:
        """
        Fetch one payroll period from /my-workingtimes.

        Returns (range, applications) where range is {'from': ..., 'to': ...}.
        The endpoint ignores ?from=&to=; ?month= is what selects the period.
        """
        r = self._request(
            "GET",
            f"{BASE_URL}/my-workingtimes",
            params={"month": month},
            timeout=15,
        )

        range_match = re.search(r"const range = (\{.*?\});", r.text, re.DOTALL)
        apps_match = re.search(r"const otherApps = (\[.*?\]);", r.text, re.DOTALL)

        period = None
        applications: list = []
        with contextlib.suppress(json.JSONDecodeError):
            if range_match:
                period = json.loads(range_match.group(1))
            if apps_match:
                applications = json.loads(apps_match.group(1))

        return period, applications

    def get_employee_code(self) -> str | None:
        """Read the signed-in account's employee code (Mã NV, e.g. 'VT0123')."""
        r = self._request("GET", f"{BASE_URL}/user-setting", timeout=15)
        r.raise_for_status()
        tag = re.search(r"<input\b[^>]*\bid=\"id\"[^>]*>", r.text)
        if not tag:
            return None
        value = re.search(r"\bvalue=\"([^\"]*)\"", tag.group(0))
        if not value:
            return None
        return value.group(1).strip() or None

    def fetch_periods(self) -> list[tuple[dict, list]]:
        """Fetch every payroll period the site exposes as (period, applications) pairs."""
        periods = []
        for month in AVAILABLE_PERIODS:
            period, applications = self._fetch_period(month)
            if period:
                periods.append(({**period, "month": month}, applications))
        return periods

    def get_day_status(self, date: str) -> dict:
        """
        Report leave / late / early-leave / overtime applications for a date.

        Applications are the only source of truth available to an employee account:
        the raw attendance feed (employeeWorkingTimeData) is empty and
        /employee-working-time-data returns 403 for non-managers.

        Args:
            date: ISO date string, e.g. "2026-08-06"

        Returns a dict with is_late / is_overtime / is_early_leave / is_all_day_leave
        flags plus recommended_time_in and recommended_time_out to file with.
        """
        datetime.strptime(date, "%Y-%m-%d")  # validate format early

        for month in AVAILABLE_PERIODS:
            candidate_period, candidate_apps = self._fetch_period(month)
            if not candidate_period:
                continue
            if candidate_period["from"] <= date <= candidate_period["to"]:
                return summarize_day(date, {**candidate_period, "month": month}, candidate_apps)

        return summarize_day(date, None, [])


    def save_timesheet(
        self,
        date: str,
        entries: list,
        message: str = "",
        time_in: str = DEFAULT_TIME_IN,
        time_out: str = DEFAULT_TIME_OUT,
    ) -> dict:
        """Save a timesheet as DRAFT."""
        return self._post_timesheet(
            f"{BASE_URL}/wt-applications/save",
            date,
            entries,
            message,
            time_in,
            time_out,
        )

    def submit_timesheet(
        self,
        date: str,
        entries: list,
        message: str = "",
        time_in: str = DEFAULT_TIME_IN,
        time_out: str = DEFAULT_TIME_OUT,
    ) -> dict:
        """Save and SUBMIT a timesheet for approval."""
        return self._post_timesheet(
            f"{BASE_URL}/wt-applications/submit",
            date,
            entries,
            message,
            time_in,
            time_out,
        )

    def delete_application(self, application_id: int) -> dict:
        """Delete a timesheet application, as the Delete button on its page does."""
        r = self._request(
            "DELETE",
            f"{BASE_URL}/api/wt-applications/{application_id}",
            headers=_api_referer(f"{BASE_URL}/wt-applications/{application_id}"),
            timeout=15,
        )
        r.raise_for_status()
        return r.json() if r.text else {"status": "deleted"}

    def get_application(self, application_id: int) -> dict:
        """The timesheet application its detail page embeds as ``let application = {...}``."""
        r = self._request("GET", f"{BASE_URL}/wt-applications/{application_id}", timeout=15)
        r.raise_for_status()
        marker = re.search(r"let application\s*=\s*", r.text)
        if not marker:
            raise RuntimeError(f"Application {application_id} has no embedded data; is it yours?")
        application, _ = json.JSONDecoder().raw_decode(r.text, marker.end())
        return application

    def submit_working_days(self, working_days: list[dict], message: str = "") -> dict:
        """Submit already-built working days, as the create page's Submit button does."""
        return self._post_working_days(f"{BASE_URL}/wt-applications/submit", working_days, message)

    def list_timesheets(self, statuses: list[str], date_from: str | None = None) -> list[dict]:
        """
        Read timesheet applications from /wt-applications, one status tab at a time.

        The page renders a single tab per request (``?status=``, 15 rows a page) and
        opens on "pending", which is why reading it without a status looks empty.
        Reading a tab stops at an empty page, or at the first page whose rows all end
        before ``date_from``: the finished tab is ordered by submission time and runs
        back for years.
        """
        rows: list[dict] = []
        for status in statuses:
            for page in range(1, LIST_PAGE_LIMIT + 1):
                r = self._request(
                    "GET",
                    f"{BASE_URL}/wt-applications",
                    params={"status": LIST_STATUSES[status], "page": page},
                    timeout=15,
                )
                r.raise_for_status()
                page_rows = parse_timesheet_rows(r.text, status)
                rows.extend(page_rows)
                if not page_rows:
                    break
                if date_from and all(
                    row["date_to"] and row["date_to"] < date_from for row in page_rows
                ):
                    break
        return rows

    def _post_timesheet(
        self,
        url: str,
        date: str,
        entries: list,
        message: str,
        time_in: str = DEFAULT_TIME_IN,
        time_out: str = DEFAULT_TIME_OUT,
    ) -> dict:
        """Internal method to POST one day's timesheet data."""
        return self._post_working_days(
            url, [build_working_day(date, entries, time_in, time_out)], message
        )

    def _post_working_days(self, url: str, working_days: list[dict], message: str) -> dict:
        r = self._request(
            "POST", url, json={"message": message, "working_days": working_days}, timeout=15
        )
        r.raise_for_status()
        return r.json()


class CommandRefused(Exception):
    """A command was not carried out; ``payload`` is the JSON result to print."""

    def __init__(self, payload: dict):
        super().__init__(payload.get("error", "refused"))
        self.payload = {"success": False, **payload}


def _status_from_periods(periods: list[tuple[dict, list]], date: str) -> dict:
    """take100's day summary for a date, from periods fetched once for many dates."""
    for period, applications in periods:
        if period["from"] <= date <= period["to"]:
            return summarize_day(date, period, applications)
    return summarize_day(date, None, [])


def _working_window(
    day_status: dict | None, time_in: str | None, time_out: str | None
) -> tuple[str, str]:
    """
    The time_in / time_out to file, from the day's applications unless overridden.

    ``day_status`` is None when detection was skipped (--no-auto).
    """
    if day_status is None:
        return time_in or DEFAULT_TIME_IN, time_out or DEFAULT_TIME_OUT

    date = day_status["date"]
    if day_status["is_all_day_leave"]:
        raise CommandRefused(
            {
                "error": f"{date} is an all-day paid leave — no timesheet should be filed. "
                "Use --no-auto to override.",
                "day_status": day_status,
            }
        )
    if not day_status["data_available"]:
        raise CommandRefused(
            {
                "error": day_status["error"],
                "hint": "Pass --time-in/--time-out explicitly, or --no-auto.",
            }
        )

    resolved_in = time_in or day_status["recommended_time_in"]
    resolved_out = time_out or day_status["recommended_time_out"]
    if resolved_out is None:
        raise CommandRefused(
            {
                "error": f"Could not determine time_out for {date} automatically.",
                "warnings": day_status["warnings"],
            }
        )
    return resolved_in, resolved_out


def _applied_summary(day_status: dict) -> dict:
    return {
        "late": day_status["is_late"],
        "late_until": day_status["late_until"],
        "overtime": day_status["is_overtime"],
        "ot_window": (
            f"{day_status['ot_from']}-{day_status['ot_to']}" if day_status["is_overtime"] else None
        ),
        "early_leave": day_status["is_early_leave"],
        "warnings": day_status["warnings"],
    }


def _timesheets_on_file(
    client: Take100Client, dates: list[str], statuses: tuple[str, ...] = ON_FILE_STATUSES
) -> dict[str, list[dict]]:
    """Existing timesheets (saved / pending / finished by default) covering each date."""
    rows = client.list_timesheets(list(statuses), date_from=min(dates))
    on_file: dict[str, list[dict]] = {}
    for row in rows:
        if not row["date_from"]:
            continue
        for date in dates:
            if row["date_from"] <= date <= (row["date_to"] or row["date_from"]):
                on_file.setdefault(date, []).append(
                    {"id": row["id"], "no": row["no"], "status": row["status"]}
                )
    return on_file


def _refuse_if_on_file(client: Take100Client, dates: list[str]) -> None:
    on_file = _timesheets_on_file(client, dates)
    if on_file:
        raise CommandRefused(
            {
                "error": "A timesheet is already on file for these dates. Nothing was sent.",
                "on_file": on_file,
                "hint": "Filing again would create a second application. To submit a saved "
                "draft use --action submit-drafts --ids <id>; otherwise delete the existing "
                f"one first (--action delete) or review it on {BASE_URL}/wt-applications.",
            }
        )


def _read_entries(args: argparse.Namespace) -> list:
    if args.entries_file:
        # utf-8-sig strips the BOM that PowerShell's Set-Content -Encoding utf8 adds
        with open(args.entries_file, encoding="utf-8-sig") as f:
            entries = json.load(f)
        # Auto-cleanup: delete the temp file after reading so it isn't left on disk
        with contextlib.suppress(OSError):
            os.remove(args.entries_file)
        return entries
    if args.entries:
        return json.loads(args.entries)
    raise CommandRefused({"error": "--entries (or --entries-file) is required for save/submit"})


def _require_date(args: argparse.Namespace) -> str:
    if not args.date:
        raise CommandRefused({"error": f"--date is required for {args.action}"})
    datetime.strptime(args.date, "%Y-%m-%d")
    return args.date


def _cmd_check_day(client: Take100Client, args: argparse.Namespace) -> dict:
    return {"success": True, "day_status": client.get_day_status(_require_date(args))}


def _cmd_save(client: Take100Client, args: argparse.Namespace) -> dict:
    date = _require_date(args)
    entries = _read_entries(args)
    day_status = None if args.no_auto else client.get_day_status(date)
    time_in, time_out = _working_window(day_status, args.time_in, args.time_out)

    if not args.skip_validation:
        problems = validate_entries(entries, time_in, time_out)
        if problems:
            raise CommandRefused(
                {
                    "error": "Entries are inconsistent with the working window "
                    f"{time_in}-{time_out}. Nothing was sent.",
                    "problems": problems,
                    "day_status": day_status,
                    "hint": "Rebuild the entries to cover the window, or pass "
                    "--skip-validation to file them as-is.",
                }
            )
    _refuse_if_on_file(client, [date])

    post = client.save_timesheet if args.action == "save" else client.submit_timesheet
    resp = post(date, entries, args.message, time_in=time_in, time_out=time_out)
    result = {
        "success": True,
        "action": args.action,
        "time_in": time_in,
        "time_out": time_out,
        "application": resp,
    }
    if day_status:
        result["applied"] = _applied_summary(day_status)
    return result


def _parse_days(raw: str | None) -> list[dict]:
    if not raw:
        raise CommandRefused({"error": "--days is required for save-many"})
    days = json.loads(raw)
    if not isinstance(days, list) or not days:
        raise CommandRefused({"error": "--days must be a non-empty JSON array"})
    for day in days:
        if not isinstance(day, dict) or "date" not in day or "entries" not in day:
            raise CommandRefused(
                {"error": "Each --days item needs 'date' and 'entries'.", "item": day}
            )
        datetime.strptime(day["date"], "%Y-%m-%d")
    dates = [day["date"] for day in days]
    if len(set(dates)) != len(dates):
        raise CommandRefused({"error": "--days lists the same date more than once."})
    return days


def _plan_days(periods: list, days: list[dict]) -> tuple[list[tuple], dict[str, list[str]]]:
    """Resolve every day's window and validate its entries; collect problems per date."""
    plan, problems = [], {}
    for day in days:
        status = _status_from_periods(periods, day["date"])
        try:
            time_in, time_out = _working_window(status, day.get("time_in"), day.get("time_out"))
        except CommandRefused as refused:
            problems[day["date"]] = [refused.payload["error"]]
            continue
        day_problems = validate_entries(day["entries"], time_in, time_out)
        if day_problems:
            problems[day["date"]] = day_problems
            continue
        plan.append((day, time_in, time_out, status))
    return plan, problems


def _cmd_save_many(client: Take100Client, args: argparse.Namespace) -> dict:
    """
    Save (or with --submit, submit) several days in one run, all-or-nothing on validation.

    Every day is checked (window, entries, nothing already on file) before the first
    one is sent, so a bad day never leaves half a period filed.
    """
    days = _parse_days(args.days)
    plan, problems = _plan_days(client.fetch_periods(), days)
    for date, existing in _timesheets_on_file(client, [day["date"] for day in days]).items():
        problems.setdefault(date, []).append(
            f"A timesheet is already on file: {', '.join(item['no'] for item in existing)}."
        )
    if problems:
        raise CommandRefused(
            {
                "error": "Some days cannot be filed. Nothing was sent.",
                "problems": problems,
                "hint": "Fix or drop those days and run save-many again.",
            }
        )

    post = client.submit_timesheet if args.submit else client.save_timesheet
    results = []
    for day, time_in, time_out, status in plan:
        try:
            resp = post(
                day["date"], day["entries"], day.get("message", ""), time_in=time_in,
                time_out=time_out,
            )
            results.append(
                {
                    "date": day["date"],
                    "success": True,
                    "time_in": time_in,
                    "time_out": time_out,
                    "application": resp,
                    "applied": _applied_summary(status),
                }
            )
        except requests.HTTPError as exc:
            results.append(
                {
                    "date": day["date"],
                    "success": False,
                    "error": f"HTTP {exc.response.status_code}",
                    "details": exc.response.text[:300],
                }
            )
    return {
        "success": all(item["success"] for item in results),
        "action": "save-many",
        "submitted": args.submit,
        "results": results,
    }


def _draft_ids(client: Take100Client, args: argparse.Namespace) -> list[int]:
    if args.ids:
        try:
            return [int(value) for value in args.ids.split(",") if value.strip()]
        except ValueError:
            raise CommandRefused({"error": "--ids must be comma-separated numbers"}) from None
    if args.all_saved:
        date_from, date_to, _ = _list_range(client, args)
        rows = client.list_timesheets(["saved"], date_from=date_from)
        return [
            int(row["id"])
            for row in rows
            if row["id"] and row["date_from"] and date_from <= row["date_from"] <= date_to
        ]
    raise CommandRefused({"error": "Pass --ids <id,id,...> or --all-saved for submit-drafts"})


def _plan_drafts(client: Take100Client, ids: list[int]) -> tuple[list[dict], dict]:
    """Load each draft and check it is a saved, internally consistent timesheet."""
    plan, problems = [], {}
    for application_id in ids:
        application = client.get_application(application_id)
        key = str(application.get("no") or application_id)
        if application.get("status") != LIST_STATUSES["saved"]:
            problems[key] = [f"Not a saved draft (status {application.get('status')})."]
            continue
        days = draft_days(application)
        day_problems = [
            f"{day['date']}: {problem}"
            for day in days
            for problem in validate_entries(day["entries"], day["time_in"], day["time_out"])
        ]
        if not days:
            day_problems.append("The draft has no working days.")
        if day_problems:
            problems[key] = day_problems
            continue
        plan.append({"id": application_id, "no": key, "days": days,
                     "message": application.get("message") or ""})
    return plan, problems


def _note_already_submitted(client: Take100Client, plan: list[dict], problems: dict) -> None:
    """Add a problem for every draft day that already has a pending or finished timesheet."""
    dates = [day["date"] for draft in plan for day in draft["days"]]
    if not dates:
        return
    on_file = _timesheets_on_file(client, dates, statuses=("pending", "finished"))
    for draft in plan:
        for day in draft["days"]:
            if day["date"] in on_file:
                numbers = ", ".join(item["no"] for item in on_file[day["date"]])
                problems.setdefault(draft["no"], []).append(
                    f"{day['date']} is already submitted as {numbers}."
                )


def _submit_draft(client: Take100Client, draft: dict) -> dict:
    """Submit a draft's content as a new application, then delete the draft."""
    working_days = []
    for day in draft["days"]:
        working_day = build_working_day(day["date"], day["entries"], day["time_in"],
                                        day["time_out"])
        working_day["shift_work_id"] = day["shift_work_id"]
        working_days.append(working_day)
    result = {"draft": draft["no"], "dates": [day["date"] for day in draft["days"]]}
    try:
        result["application"] = client.submit_working_days(working_days, draft["message"])
    except requests.HTTPError as exc:
        return {**result, "success": False, "error": f"HTTP {exc.response.status_code}",
                "details": exc.response.text[:300]}
    try:
        client.delete_application(draft["id"])
        result["draft_deleted"] = True
    except requests.HTTPError as exc:
        result["draft_deleted"] = False
        result["warning"] = (
            f"Submitted, but draft {draft['no']} could not be deleted "
            f"(HTTP {exc.response.status_code}); delete it on {BASE_URL}/wt-applications."
        )
    return {**result, "success": True}


def _cmd_submit_drafts(client: Take100Client, args: argparse.Namespace) -> dict:
    """
    Submit saved drafts for approval.

    The site has no "submit this draft" request: a draft's page only offers Delete,
    and Submit on the create page posts a new application. So each draft's own
    content is submitted as a new application and the draft is then deleted —
    what a person does by hand. All drafts are checked before the first is sent.
    """
    ids = _draft_ids(client, args)
    if not ids:
        raise CommandRefused({"error": "No saved drafts found to submit."})
    plan, problems = _plan_drafts(client, ids)
    _note_already_submitted(client, plan, problems)
    if problems:
        raise CommandRefused({"error": "Some drafts cannot be submitted. Nothing was sent.",
                              "problems": problems})

    preview = [{"draft": d["no"], "days": d["days"]} for d in plan]
    if args.dry_run:
        return {"success": True, "action": "submit-drafts", "dry_run": True, "drafts": preview}
    results = [_submit_draft(client, draft) for draft in plan]
    return {"success": all(item["success"] for item in results), "action": "submit-drafts",
            "results": results}


def _list_range(client: Take100Client, args: argparse.Namespace) -> tuple[str, str, list]:
    periods = client.fetch_periods()
    current = next((period for period, _ in periods if period["month"] == "this_month"), None)
    date_from = args.date_from or (current["from"] if current else None)
    date_to = args.date_to or (current["to"] if current else None)
    if not date_from or not date_to:
        raise CommandRefused({"error": "Pass --from and --to; the current period is unknown."})
    return date_from, date_to, periods


def _weekdays_without_timesheet(
    rows: list[dict], date_from: str, date_to: str, periods: list
) -> list[dict]:
    """Past weekdays in the range with no timesheet on file, and whether a leave explains it."""
    covered = set()
    for row in rows:
        if row["status"] in ON_FILE_STATUSES and row["date_from"]:
            start = datetime.fromisoformat(row["date_from"])
            end = datetime.fromisoformat(row["date_to"] or row["date_from"])
            while start <= end:
                covered.add(start.date().isoformat())
                start += timedelta(days=1)

    missing = []
    day = datetime.fromisoformat(date_from)
    last = min(datetime.fromisoformat(date_to), datetime.now())
    while day <= last:
        iso = day.date().isoformat()
        if day.weekday() < 5 and iso not in covered:
            status = _status_from_periods(periods, iso)
            missing.append({"date": iso, "all_day_leave": status["is_all_day_leave"]})
        day += timedelta(days=1)
    return missing


def _cmd_list(client: Take100Client, args: argparse.Namespace) -> dict:
    date_from, date_to, periods = _list_range(client, args)
    statuses = list(LIST_STATUSES) if args.status == "all" else [args.status]
    rows = [
        row
        for row in client.list_timesheets(statuses, date_from=date_from)
        if row["date_from"] and row["date_from"] <= date_to and (row["date_to"] or "") >= date_from
    ]
    result = {
        "success": True,
        "range": {"from": date_from, "to": date_to},
        "statuses": statuses,
        "timesheets": sorted(rows, key=lambda row: row["date_from"]),
    }
    if args.status == "all":
        result["weekdays_without_timesheet"] = _weekdays_without_timesheet(
            rows, date_from, date_to, periods
        )
    return result


def _cmd_delete(client: Take100Client, args: argparse.Namespace) -> dict:
    if not args.application_id:
        raise CommandRefused({"error": "--application-id is required for delete"})
    resp = client.delete_application(args.application_id)
    return {"success": True, "action": "delete", "response": resp}


COMMANDS = {
    "check-day": _cmd_check_day,
    "save": _cmd_save,
    "submit": _cmd_save,
    "save-many": _cmd_save_many,
    "submit-drafts": _cmd_submit_drafts,
    "list": _cmd_list,
    "delete": _cmd_delete,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Take100 Timesheet API Client")
    parser.add_argument(
        "--action",
        choices=list(COMMANDS),
        required=True,
        help="check-day, save (draft), submit (for approval), save-many (several days), "
        "submit-drafts (submit saved drafts), list (timesheets in a range), or delete",
    )
    parser.add_argument("--date", help="Date for the timesheet (YYYY-MM-DD)")
    parser.add_argument("--entries", help="JSON array of work entries")
    parser.add_argument(
        "--entries-file",
        help="Path to UTF-8 entries JSON, for shells that corrupt non-ASCII arguments",
    )
    parser.add_argument(
        "--days",
        help='save-many: JSON array of {"date", "entries"} objects, optionally with '
        '"time_in", "time_out" and "message"',
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="save-many: submit for approval instead of saving drafts",
    )
    parser.add_argument("--ids", help="submit-drafts: comma-separated draft application ids")
    parser.add_argument(
        "--all-saved",
        action="store_true",
        help="submit-drafts: every saved draft in the range (default: current period)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="submit-drafts: show what would be submitted without sending anything",
    )
    parser.add_argument(
        "--status",
        choices=["all", *LIST_STATUSES],
        default="all",
        help="list: which tab to read (default: all)",
    )
    parser.add_argument("--from", dest="date_from", help="list: start date (default: period)")
    parser.add_argument("--to", dest="date_to", help="list: end date (default: period)")
    parser.add_argument("--application-id", type=int, help="Application ID (for delete)")
    parser.add_argument("--message", default="", help="Optional message")
    parser.add_argument(
        "--email",
        default=None,
        help="Login email. Precedence: --email > TAKE100_EMAIL env (skill-secret injection).",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Login password. Precedence: --password > TAKE100_PASSWORD env (skill injection).",
    )
    parser.add_argument(
        "--time-in",
        default=None,
        help="Override time_in, e.g. '08:06' when late (default: 08:00)",
    )
    parser.add_argument(
        "--time-out",
        default=None,
        help="Override time_out, e.g. '16:30' for early leave (default: 17:00)",
    )
    parser.add_argument(
        "--no-auto",
        action="store_true",
        help="Do not look up late/overtime applications; use plain 08:00-17:00 defaults.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="File entries even if they do not tile the working window.",
    )
    return parser


def _print(result: dict) -> None:
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    args = _build_parser().parse_args()

    email = args.email or os.getenv("TAKE100_EMAIL")
    password = args.password or os.getenv("TAKE100_PASSWORD")
    if not email or not password:
        _print(
            {
                "success": False,
                "error": "Missing credentials. Provide --email/--password, "
                "bind TAKE100_EMAIL/TAKE100_PASSWORD in the skill secret settings.",
            }
        )
        sys.exit(1)

    client = Take100Client(email=email, password=password)
    try:
        result = COMMANDS[args.action](client, args)
    except CommandRefused as refused:
        _print(refused.payload)
        sys.exit(1)
    except requests.HTTPError as e:
        error_body = e.response.text[:500] if e.response is not None else str(e)
        _print(
            {
                "success": False,
                "error": f"HTTP {e.response.status_code}",
                "details": error_body,
            }
        )
        sys.exit(1)
    except Exception as e:
        _print({"success": False, "error": str(e)})
        sys.exit(1)

    _print(result)
    if not result.get("success", True):
        sys.exit(1)


if __name__ == "__main__":
    main()
