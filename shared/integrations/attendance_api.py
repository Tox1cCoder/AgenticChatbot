"""
attendance_api.py — Direct API client for the VT Check Attendance admin panel.

Reads real check-in / check-out punches over HTTP instead of driving a browser, then
compares them with the late / early-leave / overtime applications filed on
take100dot.com, so late arrivals and unclaimed overtime surface per day.

The panel is a Next.js app behind an ngrok tunnel. It authenticates with a NextAuth
credentials provider (identifier = email or Mã NV) and serves
``GET /api/attendance?fromDate=&toDate=&maNhanVien=`` as ``{"data": [...]}``.

Usage (called by the agent via bash):
    python skills/attendance/scripts/attendance.py
    python skills/attendance/scripts/attendance.py --from 2026-09-21 --to 2026-10-12
    python skills/attendance/scripts/attendance.py --employee VT0123
    python skills/attendance/scripts/attendance.py --base-url https://xxxx.ngrok-free.app
"""

import argparse
import contextlib
import json
import os
import sys
import urllib.parse
from datetime import date, timedelta
from pathlib import Path

import requests

from shared.integrations import take100_api
from shared.integrations.take100_api import to_minutes

DEFAULT_SESSION_CACHE_PATH = str(Path.home() / ".cache" / "attendance" / "session.json")

# The admin UI's own "Trễ sau: 08:00 AM" threshold and the company's default shift.
SHIFT_START = "08:00"
SHIFT_END = "17:00"
# Payroll periods run from the 21st of one month to the 20th of the next.
PAYROLL_CYCLE_START_DAY = 21
WEEKEND_DAYS = (5, 6)
REQUEST_TIMEOUT = 30

# Without this header ngrok's free tier may serve its browser-warning interstitial.
NGROK_HEADERS = {"ngrok-skip-browser-warning": "1"}


class AttendanceUnavailableError(RuntimeError):
    """The tunnel is offline or the panel did not answer like the attendance app."""


class AttendanceLoginError(RuntimeError):
    """The panel rejected the configured credentials."""


def normalize_base_url(url: str) -> str:
    """Reduce a pasted URL such as 'https://x.ngrok-free.app/login' to its origin."""
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Not an http(s) URL: {url!r}")
    return f"{parsed.scheme}://{parsed.netloc}"


def payroll_cycle(today: date) -> tuple[date, date]:
    """Return the current payroll cycle's start and today (never a future date)."""
    if today.day >= PAYROLL_CYCLE_START_DAY:
        start = today.replace(day=PAYROLL_CYCLE_START_DAY)
    else:
        last_month_end = today.replace(day=1) - timedelta(days=1)
        start = last_month_end.replace(day=PAYROLL_CYCLE_START_DAY)
    return start, today


def read_cached_base_url(session_cache_path: str | os.PathLike[str]) -> str | None:
    """The tunnel URL that last worked, so a new URL only has to be given once."""
    with contextlib.suppress(OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        data = json.loads(Path(session_cache_path).read_text(encoding="utf-8"))
        return data.get("base_url") or None
    return None


class AttendanceClient:
    """HTTP client for the attendance panel with a persisted NextAuth session."""

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        session_cache_path: str | os.PathLike[str] | None = None,
    ):
        self.base_url = normalize_base_url(base_url)
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.session.headers.update(NGROK_HEADERS)
        self.session_cache_path = Path(session_cache_path or DEFAULT_SESSION_CACHE_PATH)
        self._load_session_cache()

    def _load_session_cache(self) -> None:
        """Restore cookies only if they were issued to the same tunnel and account."""
        with contextlib.suppress(OSError, UnicodeDecodeError, json.JSONDecodeError):
            data = json.loads(self.session_cache_path.read_text(encoding="utf-8"))
            if data.get("base_url") != self.base_url or data.get("username") != self.username:
                return
            for cookie in data.get("cookies") or []:
                self.session.cookies.set_cookie(requests.cookies.create_cookie(**cookie))

    def _save_session_cache(self) -> None:
        self.session_cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_url": self.base_url,
            "username": self.username,
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
        self.session_cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _check_tunnel(self, response: requests.Response) -> None:
        """Turn ngrok's offline page into an actionable error instead of a parse failure."""
        error_code = response.headers.get("ngrok-error-code", "")
        if error_code or "ERR_NGROK" in response.text[:5000]:
            raise AttendanceUnavailableError(
                f"The attendance tunnel {self.base_url} is offline "
                f"({error_code or 'ngrok error page'}). Ask the user for the current ngrok "
                f"URL and pass it with --base-url."
            )

    def _get(self, path: str, **kwargs) -> requests.Response:
        response = self.session.get(f"{self.base_url}{path}", timeout=REQUEST_TIMEOUT, **kwargs)
        self._check_tunnel(response)
        return response

    def login(self) -> None:
        """Sign in through NextAuth's credentials callback, as the login form does."""
        self.session.cookies.clear()
        csrf = self._get("/api/auth/csrf")
        csrf.raise_for_status()
        response = self.session.post(
            f"{self.base_url}/api/auth/callback/credentials",
            data={
                "csrfToken": csrf.json()["csrfToken"],
                "identifier": self.username,
                "password": self.password,
                "callbackUrl": f"{self.base_url}/",
                "json": "true",
            },
            timeout=REQUEST_TIMEOUT,
            # NextAuth answers with a redirect to its configured NEXTAUTH_URL, which
            # points at an older tunnel; the session cookie is all that matters.
            allow_redirects=False,
        )
        self._check_tunnel(response)

        session_info = self._get("/api/auth/session").json()
        if not session_info.get("user"):
            raise AttendanceLoginError(
                "Login to the attendance panel failed. Check the ATTENDANCE_USERNAME / "
                "ATTENDANCE_PASSWORD bindings in the attendance-lookup skill settings."
            )
        self._save_session_cache()

    def fetch_attendance(self, date_from: date, date_to: date, employee_code: str) -> list:
        """Return one employee's punches between two dates, oldest first."""
        params = {
            "fromDate": date_from.isoformat(),
            "toDate": date_to.isoformat(),
            "maNhanVien": employee_code,
        }
        response = self._get("/api/attendance", params=params)
        if response.status_code == 401:
            self.login()
            response = self._get("/api/attendance", params=params)
        response.raise_for_status()

        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise AttendanceUnavailableError(
                f"{self.base_url}/api/attendance did not return {{'data': [...]}}; "
                f"is this still the attendance panel?"
            )
        # The server filters by maNhanVien today; filter again so a change there can
        # never mix another employee's punches into this report.
        rows = [row for row in payload["data"] if row.get("maNhanVien") == employee_code]
        return sorted(rows, key=lambda row: row.get("attendanceDate", ""))


def _check_in_fields(check_in: str, status: dict | None) -> dict:
    late = max(0, to_minutes(check_in) - to_minutes(SHIFT_START))
    unexcused = None
    if status is not None:
        allowed = to_minutes(status["recommended_time_in"])
        unexcused = max(0, to_minutes(check_in) - allowed)
    return {"late_minutes": late, "unexcused_late_minutes": unexcused}


def _check_out_fields(check_out: str | None, status: dict | None) -> dict:
    if check_out is None:
        return {
            "early_leave_minutes": None,
            "unexcused_early_minutes": None,
            "overtime_minutes": None,
            "unclaimed_overtime_minutes": None,
        }

    out = to_minutes(check_out)
    shift_end = to_minutes(SHIFT_END)
    fields = {
        "early_leave_minutes": max(0, shift_end - out),
        "unexcused_early_minutes": None,
        "overtime_minutes": max(0, out - shift_end),
        "unclaimed_overtime_minutes": None,
    }
    if status is not None:
        allowed_out = to_minutes(status["early_from"]) if status["is_early_leave"] else shift_end
        claimed_end = to_minutes(status["ot_to"]) if status["is_overtime"] else shift_end
        fields["unexcused_early_minutes"] = max(0, allowed_out - out)
        fields["unclaimed_overtime_minutes"] = max(0, out - max(shift_end, claimed_end))
    return fields


def _filed_summary(status: dict | None) -> dict | None:
    if status is None:
        return None
    return {
        "all_day_leave": status["is_all_day_leave"],
        "late_until": status["late_until"],
        "early_from": status["early_from"],
        "ot_from": status["ot_from"],
        "ot_to": status["ot_to"],
    }


def _day_notes(day: date, check_out: str | None, status: dict | None) -> list[str]:
    notes = []
    if check_out is None:
        notes.append("No check-out recorded (still at work, or the punch is missing).")
    if day.weekday() in WEEKEND_DAYS:
        notes.append("Weekend: the 08:00-17:00 shift rules may not apply.")
    if status is None:
        return notes
    if status["is_all_day_leave"]:
        notes.append("An all-day leave is filed for this date, but there are punches.")
    if status["is_overtime"] and check_out and to_minutes(check_out) < to_minutes(status["ot_to"]):
        notes.append(
            f"Overtime is filed until {status['ot_to']} but check-out was {check_out[:5]}."
        )
    notes.extend(status["warnings"])
    return notes


def analyze_day(record: dict, status: dict | None) -> dict:
    """
    Compare one day's punches with the shift and with what was filed on take100.

    ``status`` is take100's day summary, or None when applications were not checked;
    the comparison fields are then None rather than a guess.
    """
    day = date.fromisoformat(record["attendanceDate"])
    check_in, check_out = record["checkIn"], record.get("checkOut")
    return {
        "date": record["attendanceDate"],
        "weekday": day.strftime("%a"),
        "check_in": check_in,
        "check_out": check_out,
        **_check_in_fields(check_in, status),
        **_check_out_fields(check_out, status),
        "filed": _filed_summary(status),
        "notes": _day_notes(day, check_out, status),
    }


def missing_workdays(
    records: list, date_from: date, date_to: date, statuses: dict[str, dict]
) -> list[dict]:
    """Weekdays in the range with no punches, and whether a leave explains them."""
    punched = {record["attendanceDate"] for record in records}
    missing = []
    day = date_from
    while day <= date_to:
        iso = day.isoformat()
        if day.weekday() not in WEEKEND_DAYS and iso not in punched:
            status = statuses.get(iso)
            missing.append(
                {
                    "date": iso,
                    "weekday": day.strftime("%a"),
                    "all_day_leave_filed": status["is_all_day_leave"] if status else None,
                }
            )
        day += timedelta(days=1)
    return missing


def _sum(days: list, field: str) -> int | None:
    values = [day[field] for day in days if day[field] is not None]
    return sum(values) if values else None


def build_report(
    records: list, date_from: date, date_to: date, statuses: dict[str, dict] | None
) -> dict:
    """Assemble the per-day analysis, missing days and totals for one employee."""
    lookup = statuses or {}
    days = [analyze_day(record, lookup.get(record["attendanceDate"])) for record in records]
    first = records[0] if records else {}
    return {
        "employee": {
            "code": first.get("maNhanVien"),
            "name": first.get("tenNhanVien"),
            "department": first.get("phongBan"),
        },
        "period": {"from": date_from.isoformat(), "to": date_to.isoformat()},
        "applications_checked": statuses is not None,
        "shift": {"start": SHIFT_START, "end": SHIFT_END},
        "days": days,
        "missing_workdays": missing_workdays(records, date_from, date_to, lookup),
        "totals": {
            "days_with_punches": len(days),
            "days_late": sum(1 for day in days if day["late_minutes"] > 0),
            "late_minutes": _sum(days, "late_minutes"),
            "unexcused_late_minutes": _sum(days, "unexcused_late_minutes"),
            "overtime_minutes": _sum(days, "overtime_minutes"),
            "unclaimed_overtime_minutes": _sum(days, "unclaimed_overtime_minutes"),
        },
    }


def day_statuses(periods: list[tuple[dict, list]], date_from: date, date_to: date) -> dict:
    """take100 day summaries for every date in the range that a payroll period covers."""
    statuses = {}
    day = date_from
    while day <= date_to:
        iso = day.isoformat()
        for period, applications in periods:
            if period["from"] <= iso <= period["to"]:
                statuses[iso] = take100_api.summarize_day(iso, period, applications)
                break
        day += timedelta(days=1)
    return statuses


def _take100_context(employee: str | None, warnings: list) -> tuple[str | None, list | None]:
    """
    Resolve the target employee code and, when it is the take100 account's own,
    that account's filed applications.

    Applications on take100 belong to the signed-in account only, so another
    employee's punches are reported without the comparison.
    """
    email, password = os.getenv("TAKE100_EMAIL"), os.getenv("TAKE100_PASSWORD")
    if not email or not password:
        warnings.append("take100 credentials are not bound; filed applications not checked.")
        return employee, None

    try:
        client = take100_api.Take100Client(email=email, password=password)
        own_code = client.get_employee_code()
        if employee and employee != own_code:
            warnings.append(
                f"{employee} is not the take100 account's own code ({own_code}); "
                f"filed applications not checked."
            )
            return employee, None
        return own_code, client.fetch_periods()
    except (requests.RequestException, RuntimeError) as exc:
        warnings.append(f"take100 lookup failed ({exc}); filed applications not checked.")
        return employee, None


def _resolve_range(args: argparse.Namespace) -> tuple[date, date]:
    cycle_start, today = payroll_cycle(date.today())
    date_from = date.fromisoformat(args.date_from) if args.date_from else cycle_start
    date_to = date.fromisoformat(args.date_to) if args.date_to else today
    if date_to < date_from:
        raise ValueError(f"--to {date_to} is before --from {date_from}.")
    return date_from, date_to


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VT Check Attendance lookup (read-only)")
    parser.add_argument("--from", dest="date_from", help="Start date YYYY-MM-DD "
                        "(default: the current payroll cycle's 21st)")
    parser.add_argument("--to", dest="date_to", help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--employee", help="Mã NV to look up (default: the take100 account's)")
    parser.add_argument(
        "--base-url",
        help="Attendance panel URL. Precedence: --base-url > ATTENDANCE_BASE_URL env > "
        "the last URL that worked.",
    )
    return parser.parse_args()


def _fail(error: str, **extra) -> None:
    print(json.dumps({"success": False, "error": error, **extra}, ensure_ascii=False, indent=2))
    sys.exit(1)


def main():
    args = _parse_args()
    username = os.getenv("ATTENDANCE_USERNAME")
    password = os.getenv("ATTENDANCE_PASSWORD")
    if not username or not password:
        _fail("Missing credentials. Bind ATTENDANCE_USERNAME/ATTENDANCE_PASSWORD in the "
              "attendance-lookup skill secret settings.")

    session_cache_path = os.getenv("ATTENDANCE_SESSION_CACHE_PATH") or DEFAULT_SESSION_CACHE_PATH
    # The panel's address is an ngrok tunnel that changes; it is never baked in here.
    base_url = (
        args.base_url
        or os.getenv("ATTENDANCE_BASE_URL")
        or read_cached_base_url(session_cache_path)
    )
    if not base_url:
        _fail("The attendance panel URL is unknown. Ask the user for the current ngrok URL "
              "and pass it with --base-url.")

    try:
        date_from, date_to = _resolve_range(args)
        warnings: list[str] = []
        employee, periods = _take100_context(args.employee, warnings)
        if not employee:
            _fail("Could not determine the employee code from take100. Ask the user for "
                  "their Mã NV and pass --employee.", warnings=warnings)

        client = AttendanceClient(base_url, username, password, session_cache_path)
        records = client.fetch_attendance(date_from, date_to, employee)
        statuses = day_statuses(periods, date_from, date_to) if periods is not None else None
        report = build_report(records, date_from, date_to, statuses)
        if not records:
            warnings.append(f"No punches for {employee} between {date_from} and {date_to}.")
        report["employee"]["code"] = employee
        print(json.dumps({"success": True, **report, "warnings": warnings},
                         ensure_ascii=False, indent=2))
    except (AttendanceUnavailableError, AttendanceLoginError, ValueError) as exc:
        _fail(str(exc), base_url=base_url)
    except requests.HTTPError as exc:
        _fail(f"HTTP {exc.response.status_code}", details=exc.response.text[:500],
              base_url=base_url)
    except requests.RequestException as exc:
        _fail(f"Could not reach {base_url}: {exc}. If the tunnel moved, ask the user for "
              f"the current ngrok URL and pass --base-url.", base_url=base_url)


if __name__ == "__main__":
    main()
