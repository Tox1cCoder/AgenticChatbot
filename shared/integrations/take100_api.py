"""
take100_api.py — Direct API client for take100dot.com timesheet management.

This script provides a reliable, token-efficient alternative to browser automation
for filling timesheets on take100dot.com. It uses direct HTTP API calls instead of
driving a browser via playwright-cli.

Late arrivals and overtime are detected automatically: save/submit look up the day's
applications on take100dot.com and derive time_in / time_out from them, then refuse to
send entries that do not match that working window.

Usage (called by the agent via bash):
    # Recommended: pass entries via file to avoid Windows terminal encoding issues with Unicode
    python skills/take100/take100_api.py --action check-day --date 2026-03-03
    python skills/take100/take100_api.py --action save --date 2026-03-03 --entries-file entries.json
    python skills/take100/take100_api.py --action submit --date 2026-03-03
        --entries-file entries.json
    python skills/take100/take100_api.py --action delete --application-id 30459
    python skills/take100/take100_api.py --action list

    # Escape hatches
    --time-in / --time-out   override the detected window
    --no-auto                skip detection entirely, use 08:00-17:00
    --skip-validation        send entries even if they do not tile the window

    # Inline entries (ASCII-only content only — Unicode may be garbled on Windows):
    python skills/take100/take100_api.py --action save --date 2026-03-03 --entries '[...]'

Entry JSON format (write to a .json file with UTF-8 encoding):
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
from datetime import datetime
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

        period = None
        applications: list = []
        for month in AVAILABLE_PERIODS:
            candidate_period, candidate_apps = self._fetch_period(month)
            if not candidate_period:
                continue
            if candidate_period["from"] <= date <= candidate_period["to"]:
                period = {**candidate_period, "month": month}
                applications = candidate_apps
                break

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
        """Delete an existing application by ID.

        Note: The /api/ DELETE endpoint requires Bearer token auth which is not
        available via session cookies. Users should delete from the website instead.
        This method attempts the call and returns a clear error if rejected.
        """
        r = self._request(
            "DELETE",
            f"{BASE_URL}/api/wt-applications/{application_id}",
            timeout=15,
        )
        if r.status_code == 401:
            return {
                "success": False,
                "error": "Delete via API is not supported (requires Bearer token auth). "
                f"Please delete application {application_id} manually at "
                f"{BASE_URL}/wt-applications/{application_id}",
            }
        r.raise_for_status()
        return r.json() if r.text else {"status": "deleted"}

    def list_applications(self) -> list:
        """Get list of recent applications from the main page."""
        r = self._request("GET", f"{BASE_URL}/wt-applications", timeout=15)
        # Parse the table from HTML
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table")
        if not table:
            return []

        applications = []
        for row in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) >= 5:
                # Extract link from last cell
                link = row.find("a", href=True)
                app_url = link["href"].strip() if link else ""
                app_id = app_url.rstrip("/").split("/")[-1] if app_url else ""
                applications.append(
                    {
                        "no": cells[1],
                        "date_range": cells[2],
                        "requested_time": cells[3],
                        "process": cells[4],
                        "id": app_id,
                    }
                )
        return applications

    def _post_timesheet(
        self,
        url: str,
        date: str,
        entries: list,
        message: str,
        time_in: str = DEFAULT_TIME_IN,
        time_out: str = DEFAULT_TIME_OUT,
    ) -> dict:
        """Internal method to POST timesheet data."""
        # Build the ISO date
        date_iso = f"{date}T00:00:00.000Z"

        # Build working_day_projects from entries
        working_day_projects = []
        for entry in entries:
            working_day_projects.append(
                {
                    "project_id": entry["project_id"],
                    "work_content": entry["work_content"],
                    "from": entry["from"],
                    "to": entry["to"],
                    "work_item_id": entry["work_item_id"],
                }
            )

        payload = {
            "message": message,
            "working_days": [
                {
                    "date": date_iso,
                    "time_in": time_in,
                    "time_out": time_out,
                    "break_start": None,
                    "break_end": None,
                    "shift_work_id": DEFAULT_SHIFT_WORK_ID,
                    "working_day_projects": working_day_projects,
                }
            ],
        }

        r = self._request("POST", url, json=payload, timeout=15)
        r.raise_for_status()
        return r.json()


def main():
    parser = argparse.ArgumentParser(description="Take100 Timesheet API Client")
    parser.add_argument(
        "--action",
        choices=["save", "submit", "delete", "list", "check-day"],
        required=True,
        help="Action to perform: save (draft), submit (for approval), delete, list, or check-day",
    )
    parser.add_argument("--date", help="Date for the timesheet (YYYY-MM-DD)")
    parser.add_argument(
        "--entries",
        help="JSON array of work entries (ASCII-safe only; use --entries-file for Unicode)",
    )
    parser.add_argument(
        "--entries-file",
        help="Path to UTF-8 entries JSON (recommended for Vietnamese/Unicode content)",
    )
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

    args = parser.parse_args()

    email = args.email or os.getenv("TAKE100_EMAIL")
    password = args.password or os.getenv("TAKE100_PASSWORD")

    if not email or not password:
        print(
            json.dumps(
                {
                    "success": False,
                    "error": "Missing credentials. Provide --email/--password, "
                    "bind TAKE100_EMAIL/TAKE100_PASSWORD in the skill secret settings.",
                }
            )
        )
        sys.exit(1)

    client = Take100Client(email=email, password=password)

    try:
        if args.action == "check-day":
            if not args.date:
                print(json.dumps({"success": False, "error": "--date is required for check-day"}))
                sys.exit(1)
            status = client.get_day_status(args.date)
            result = {"success": True, "day_status": status}

        elif args.action == "list":
            apps = client.list_applications()
            result = {"success": True, "applications": apps}

        elif args.action in ("save", "submit"):
            if not args.date or not (args.entries or args.entries_file):
                print(
                    json.dumps(
                        {
                            "success": False,
                            "error": "--date and (--entries or --entries-file) are required "
                            "for save/submit",
                        }
                    )
                )
                sys.exit(1)

            if args.entries_file:
                # Read entries from file — avoids Windows terminal encoding issues with Unicode
                # utf-8-sig strips the BOM that PowerShell's Set-Content -Encoding utf8 adds
                with open(args.entries_file, encoding="utf-8-sig") as f:
                    entries = json.load(f)
                # Auto-cleanup: delete the temp file after reading so it isn't left on disk
                with contextlib.suppress(OSError):
                    os.remove(args.entries_file)
            else:
                entries = json.loads(args.entries)

            time_in = args.time_in or DEFAULT_TIME_IN
            time_out = args.time_out or DEFAULT_TIME_OUT
            day_status = None

            if not args.no_auto:
                day_status = client.get_day_status(args.date)

                if day_status["is_all_day_leave"]:
                    print(
                        json.dumps(
                            {
                                "success": False,
                                "error": f"{args.date} is an all-day paid leave — no timesheet "
                                f"should be filed. Use --no-auto to override.",
                                "day_status": day_status,
                            },
                            indent=2,
                            ensure_ascii=False,
                        )
                    )
                    sys.exit(1)

                if not day_status["data_available"]:
                    print(
                        json.dumps(
                            {
                                "success": False,
                                "error": day_status["error"],
                                "hint": "Pass --time-in/--time-out explicitly, or --no-auto.",
                            },
                            indent=2,
                            ensure_ascii=False,
                        )
                    )
                    sys.exit(1)

                if not args.time_in:
                    time_in = day_status["recommended_time_in"]
                if not args.time_out:
                    if day_status["recommended_time_out"] is None:
                        print(
                            json.dumps(
                                {
                                    "success": False,
                                    "error": "Could not determine time_out automatically.",
                                    "warnings": day_status["warnings"],
                                },
                                indent=2,
                                ensure_ascii=False,
                            )
                        )
                        sys.exit(1)
                    time_out = day_status["recommended_time_out"]

            if not args.skip_validation:
                problems = validate_entries(entries, time_in, time_out)
                if problems:
                    print(
                        json.dumps(
                            {
                                "success": False,
                                "error": "Entries are inconsistent with the working window "
                                f"{time_in}-{time_out}. Nothing was sent.",
                                "problems": problems,
                                "day_status": day_status,
                                "hint": "Rebuild the entries to cover the window, or pass "
                                "--skip-validation to file them as-is.",
                            },
                            indent=2,
                            ensure_ascii=False,
                        )
                    )
                    sys.exit(1)

            if args.action == "save":
                resp = client.save_timesheet(
                    args.date, entries, args.message, time_in=time_in, time_out=time_out
                )
            else:
                resp = client.submit_timesheet(
                    args.date, entries, args.message, time_in=time_in, time_out=time_out
                )

            result = {
                "success": True,
                "action": args.action,
                "time_in": time_in,
                "time_out": time_out,
                "application": resp,
            }
            if day_status:
                result["applied"] = {
                    "late": day_status["is_late"],
                    "late_until": day_status["late_until"],
                    "overtime": day_status["is_overtime"],
                    "ot_window": (
                        f"{day_status['ot_from']}-{day_status['ot_to']}"
                        if day_status["is_overtime"]
                        else None
                    ),
                    "early_leave": day_status["is_early_leave"],
                    "warnings": day_status["warnings"],
                }

        elif args.action == "delete":
            if not args.application_id:
                print(
                    json.dumps(
                        {
                            "success": False,
                            "error": "--application-id is required for delete",
                        }
                    )
                )
                sys.exit(1)

            resp = client.delete_application(args.application_id)
            result = {"success": True, "action": "delete", "response": resp}

        print(json.dumps(result, indent=2, ensure_ascii=False))

    except requests.HTTPError as e:
        error_body = e.response.text[:500] if e.response else str(e)
        print(
            json.dumps(
                {
                    "success": False,
                    "error": f"HTTP {e.response.status_code}",
                    "details": error_body,
                }
            )
        )
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
