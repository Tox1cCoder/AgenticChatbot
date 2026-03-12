"""
take100_api.py — Direct API client for take100dot.com timesheet management.

This script provides a reliable, token-efficient alternative to browser automation
for filling timesheets on take100dot.com. It uses direct HTTP API calls instead of
driving a browser via playwright-cli.

Usage (called by the agent via bash):
    # Recommended: pass entries via file to avoid Windows terminal encoding issues with Unicode
    python skills/take100/take100_api.py --action save --date 2026-03-03 --entries-file entries.json
    python skills/take100/take100_api.py --action submit --date 2026-03-03 --entries-file entries.json
    python skills/take100/take100_api.py --action delete --application-id 30459
    python skills/take100/take100_api.py --action list

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
import calendar
import json
import re
import sys
import urllib.parse
from datetime import datetime

import requests

# Fix Windows console encoding for Unicode (e.g. Vietnamese) output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")


BASE_URL = "https://take100dot.com"
EMAIL = "thaind@v-takeuchi.vn"
PASSWORD = "Gm123123@"

# Default shift work
DEFAULT_SHIFT_WORK_ID = "C001"  # 08:00-17:00
DEFAULT_TIME_IN = "08:00"
DEFAULT_TIME_OUT = "17:00"


class Take100Client:
    """HTTP client for take100dot.com with session-based authentication."""

    def __init__(self):
        self.session = requests.Session()
        self._authenticated = False

    def login(self) -> bool:
        """Authenticate and establish session cookies. Retries once on failure."""
        for attempt in range(2):
            # GET login page for a fresh CSRF token
            r = self.session.get(f"{BASE_URL}/login", timeout=15)
            token_match = re.search(r'name="_token" value="(.*?)"', r.text)
            if not token_match:
                print(
                    "ERROR: Could not find CSRF token on login page.", file=sys.stderr
                )
                return False

            csrf_token = token_match.group(1)
            login_data = {
                "_token": csrf_token,
                "email": EMAIL,
                "password": PASSWORD,
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
                return True

            if attempt == 0:
                # Reset cookies and retry once
                self.session.cookies.clear()
            else:
                print("ERROR: Login failed — still on login page.", file=sys.stderr)

        return False

    def _get_headers(self) -> dict:
        """Build headers with XSRF token for API requests."""
        xsrf_cookie = self.session.cookies.get("XSRF-TOKEN")
        headers = {
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if xsrf_cookie:
            headers["X-XSRF-TOKEN"] = urllib.parse.unquote(xsrf_cookie)
        return headers

    def get_day_status(self, date: str) -> dict:
        """
        Check if a given date has paid leave or late/early-leave from /my-workingtimes.

        Args:
            date: ISO date string, e.g. "2026-03-06"

        Returns:
            {
                'date': '2026-03-06',
                'is_all_day_leave': bool,
                'is_late': bool,
                'is_early_leave': bool,
                'late_until': '08:06' or None,   # actual start time if late
                'early_from': '16:30' or None,   # actual end time if early leave
                'leave_apps': [...]               # raw approved paid-leave apps for the date
            }
        """
        dt = datetime.strptime(date, "%Y-%m-%d").date()
        from_date = dt.replace(day=1).strftime("%Y-%m-%d")
        last_day = calendar.monthrange(dt.year, dt.month)[1]
        to_date = dt.replace(day=last_day).strftime("%Y-%m-%d")

        r = self.session.get(
            f"{BASE_URL}/my-workingtimes",
            params={"from": from_date, "to": to_date},
            timeout=15,
        )

        other_apps_match = re.search(r"const otherApps = (\[.*?\]);", r.text, re.DOTALL)
        if not other_apps_match:
            return {
                "date": date,
                "is_all_day_leave": False,
                "is_late": False,
                "is_early_leave": False,
                "late_until": None,
                "early_from": None,
                "leave_apps": [],
            }

        try:
            other_apps = json.loads(other_apps_match.group(1))
        except json.JSONDecodeError:
            other_apps = []

        # Only approved paid-leave entries covering the target date
        day_apps = [
            app
            for app in other_apps
            if app.get("from_date", "")
            <= date
            <= app.get("to_date", app.get("from_date", ""))
            and app.get("status") == 1
            and app.get("application_type_name") == "paid leave"
        ]

        is_all_day_leave = any(app.get("reason_name") == "All day" for app in day_apps)

        # Late: "Hour" paid leave that starts in the morning (covers shift start → actual arrival)
        late_app = next(
            (
                app
                for app in day_apps
                if app.get("reason_name") == "Hour"
                and app.get("to_time", "99:99") <= "12:00"
                and app.get("from_time", "99:99") < app.get("to_time", "00:00")
            ),
            None,
        )

        # Early leave: "Hour" paid leave that starts in the afternoon
        early_app = next(
            (
                app
                for app in day_apps
                if app.get("reason_name") == "Hour"
                and app.get("from_time", "00:00") >= "13:00"
            ),
            None,
        )

        return {
            "date": date,
            "is_all_day_leave": is_all_day_leave,
            "is_late": late_app is not None,
            "is_early_leave": early_app is not None,
            "late_until": late_app["to_time"] if late_app else None,
            "early_from": early_app["from_time"] if early_app else None,
            "leave_apps": day_apps,
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
        r = self.session.delete(
            f"{BASE_URL}/api/wt-applications/{application_id}",
            headers=self._get_headers(),
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
        r = self.session.get(f"{BASE_URL}/wt-applications", timeout=15)
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

        r = self.session.post(
            url, json=payload, headers=self._get_headers(), timeout=15
        )
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
        help="Path to a UTF-8 JSON file containing the entries array (recommended for Vietnamese/Unicode content)",
    )
    parser.add_argument(
        "--application-id", type=int, help="Application ID (for delete)"
    )
    parser.add_argument("--message", default="", help="Optional message")
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

    args = parser.parse_args()

    client = Take100Client()

    # Login
    if not client.login():
        print(json.dumps({"success": False, "error": "Login failed"}))
        sys.exit(1)

    try:
        if args.action == "check-day":
            if not args.date:
                print(
                    json.dumps(
                        {"success": False, "error": "--date is required for check-day"}
                    )
                )
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
                            "error": "--date and (--entries or --entries-file) are required for save/submit",
                        }
                    )
                )
                sys.exit(1)

            if args.entries_file:
                # Read entries from file — avoids Windows terminal encoding issues with Unicode
                # utf-8-sig strips the BOM that PowerShell's Set-Content -Encoding utf8 adds
                import os

                with open(args.entries_file, encoding="utf-8-sig") as f:
                    entries = json.load(f)
                # Auto-cleanup: delete the temp file after reading so it isn't left on disk
                try:
                    os.remove(args.entries_file)
                except OSError:
                    pass
            else:
                entries = json.loads(args.entries)

            time_in = args.time_in or DEFAULT_TIME_IN
            time_out = args.time_out or DEFAULT_TIME_OUT

            if args.action == "save":
                resp = client.save_timesheet(
                    args.date, entries, args.message, time_in=time_in, time_out=time_out
                )
            else:
                resp = client.submit_timesheet(
                    args.date, entries, args.message, time_in=time_in, time_out=time_out
                )

            result = {"success": True, "action": args.action, "application": resp}

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
