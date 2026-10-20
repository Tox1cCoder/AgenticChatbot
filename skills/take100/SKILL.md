---
name: take100
description: Attendance, leave/overtime applications and timesheets for take100dot.com in one skill, through bundled commands (direct HTTP API, no browser). Reads real check-in/check-out punches from the VT Check Attendance panel to find late arrivals and unclaimed overtime, files the missing late (hourly paid leave), overtime and day-leave applications, fills daily timesheets, and submits drafts. Use for "xem chấm công", "kiểm tra đi trễ/OT", "tạo đơn đi trễ", "tạo đơn OT", "điền timesheet", "Fill timesheet", "nộp timesheet/đơn", or which days still need a timesheet.
secrets: TAKE100_EMAIL, TAKE100_PASSWORD, ATTENDANCE_USERNAME, ATTENDANCE_PASSWORD
---

# take100: attendance, applications, timesheets

## How to run it
Two Python commands, both run from this skill's folder (the one containing this file):
- `scripts/attendance.py` — real punches from the attendance panel (read-only).
- `scripts/take100.py --action ...` — applications and timesheets on take100dot.com.

The examples below are written as `{"argv": [...], "cwd": "skill"}`:
- **Client with a command tool for this skill** (it takes `argv` and `cwd`): pass them as written. There is no shell, so Vietnamese text and JSON go in verbatim.
- **Claude Code or another agent with a shell**: `cd` into this skill's folder and run `python` followed by the argv items, wrapping each JSON argument in single quotes — `python scripts/take100.py --action list`.

If a command reports missing Python packages, relay the install command it prints. **Never write your own scripts, temp files, or HTML dumps for this skill.** If a command cannot do what the user asks, say so and stop — do not improvise a workaround.

## Credentials
`TAKE100_EMAIL` / `TAKE100_PASSWORD` (take100dot.com) and `ATTENDANCE_USERNAME` / `ATTENDANCE_PASSWORD` (attendance panel) are read from the environment. A skill runtime injects them from this skill's secret settings; under Claude Code they are the user's environment variables (see `README.md`). Never ask for credential values or include them in prompts, output, or arguments. Logins are cached in `~/.cache/take100/` and `~/.cache/attendance/`; the commands log in again by themselves.

## Company rules
- Shift **C001 08:00–17:00**, lunch **12:00–13:00**. Payroll period: the **21st → the 20th** of the next month. Commands default to the current period.
- CRITICAL: check today's date (a time tool, or the system date) before working with dates.
- **Late** = an hourly paid leave from 08:00 to the check-in minute (check-in 08:51:03 → leave 08:00–08:51). The timesheet then starts one minute later (08:52).
- **Overtime** = 17:00 to the check-out rounded **down** to 5 minutes (check-out 19:36 → OT 17:00–19:35).
- Order matters: **applications first, then timesheets** — the timesheet window (time_in/time_out) is derived from the filed applications.

## The full workflow ("check my attendance and fix everything")
1. `scripts/attendance.py` → show late days, unclaimed OT, missing days (Step A).
2. For each `suggested_applications` item, get the **reason** from the user, show the list, then `file-applications` (Step B). Use `--submit` only when the user says submit/nộp; otherwise drafts.
3. `list` → days without a timesheet; ask what they worked on; `save-many` (Step C).
4. Submit drafts when the user asks: `submit-application-drafts` / `submit-drafts` (dry run first).

Any step can run alone — the user may only want one of them.

## Step A — Attendance (read-only)
```json
{"argv": ["scripts/attendance.py"], "cwd": "skill"}
```
Extra argv: `"--from", "2026-09-21", "--to", "2026-10-20"` (only when the user asks for another period), `"--employee", "VT0123"` (someone else; no application comparison), `"--base-url", "https://xxxx.ngrok-free.app"`.

The panel is behind an **ngrok tunnel whose URL changes**; the command remembers the last one that worked. If it reports the tunnel offline or unreachable, ask the user for the current URL and re-run with `--base-url` (a pasted `.../login` URL is fine). The Mã NV is read from take100 automatically — do not ask for it unless the command says so.

Output: `days`, `missing_workdays`, `suggested_applications`, `totals`, `warnings`. Per day:
| Field | Meaning |
|---|---|
| `check_in` / `check_out` | Real punches; `check_out` null = missing or still at work |
| `late_minutes` / `unexcused_late_minutes` | After 08:00 / not covered by a filed late leave |
| `overtime_minutes` / `unclaimed_overtime_minutes` | After 17:00 / beyond the filed OT end |
| `filed` | take100's `late_until`, `early_from`, `ot_from`, `ot_to`, `all_day_leave` |
| `notes` | Missing check-out, weekend, filed late/OT that stops short of the punches |

`suggested_applications` lists late and OT applications the punches call for where none is filed, ready for Step B (weekdays only; `reason` is null). A 1–4 minute "unclaimed" OT that rounds down to 17:00 or to the filed end is not suggested — that is the rounding rule, not missing OT. When `applications_checked` is false, report raw minutes only and say why (`warnings`).

Report what needs action first (late without leave, unclaimed OT, weekdays with no punches and no leave, notes), then totals. A compact table per day works well. Reply in the user's language.

## Step B — Leave and overtime applications
### `list-applications` — what is filed (any status) and the leave balance
```json
{"argv": ["scripts/take100.py", "--action", "list-applications"], "cwd": "skill"}
```
Optional: `--from`/`--to`, `--app-status all|draft|in_progress|approved|denied`, `--app-type all|overtime|paid-leave`. Rows carry `id` (for submit/delete), `no`, `status`, `kind`, dates, times and `reason`; `balance` shows remaining paid leave. Past reasons are good defaults to offer the user (e.g. late: "Lý do sức khỏe", "Đi trễ"; OT: the task worked on).

### `file-applications` — create one or more
```json
{"argv": ["scripts/take100.py", "--action", "file-applications", "--applications", "[{\"kind\": \"hour-leave\", \"date\": \"2026-10-16\", \"from\": \"08:00\", \"to\": \"08:51\", \"reason\": \"Lý do sức khỏe\"}, {\"kind\": \"overtime\", \"date\": \"2026-10-14\", \"from\": \"17:00\", \"to\": \"18:00\", \"reason\": \"Fix Metadata Streaming Backport\"}]"], "cwd": "skill"}
```
| kind | Fields | Use for |
|---|---|---|
| `hour-leave` | `date`, `from`, `to` within 08:00–17:00 | Late (from 08:00), early leave (to 17:00) |
| `overtime` | `date`, `from`, `to`; on weekdays outside 08:00–17:00 | OT |
| `day-leave` | `date`, optional `to_date` | All-day paid leave |

Every item needs a `reason` — **ask the user; never invent one**. Optional: `message`, `"priority": true`.
- Saves **drafts** by default. Add `"--submit"` only when the user explicitly asks to submit. `"--dry-run"` validates and shows the batch without sending.
- The whole batch is checked first (time windows, locked dates, paid-leave balance, an application of the same kind already on that date); if anything fails, **nothing is sent** and `problems` lists each item. Fix or drop those and run again.
- A same-kind application already on file is refused because the newest one replaces the older for timesheets. Only if the user wants to re-file a correction, add `"--allow-duplicate"`.

### `submit-application-drafts` — submit saved draft applications
```json
{"argv": ["scripts/take100.py", "--action", "submit-application-drafts", "--all-saved", "--dry-run"], "cwd": "skill"}
```
Dry run first, show the drafts, then run without `--dry-run` after the user confirms. `"--ids", "36700,36701"` picks specific drafts (the `id` from `list-applications`).

### `delete-application`
```json
{"argv": ["scripts/take100.py", "--action", "delete-application", "--application-id", "36700"], "cwd": "skill"}
```
Only when the user asks. Submitted applications usually cannot be deleted; the site's error is returned.

## Step C — Timesheets
### Default content
- **Morning warm-up** (5 min): `"Tập thể dục, báo cáo buổi sáng"`, project=**K202201**, work_item_id=**122**. **TUESDAY: 25 min** with `"Tập thể dục, báo cáo buổi sáng, Trực nhật vệ sinh"`.
- The warm-up starts at `time_in` (08:00, or the late start such as 08:52) and keeps its duration.
- Main tasks: project=**F131**, `work_item_id` by the work described: 125 Requirement analysis | 126 Database design | 127 UI Design | 128 Logic Handle | 129 Geological data Input and Check.
- **K202201** (管理・その他): 122=その他, 121=会議参加, 120=面談, 119=活動管理, 118=法務業務, 117=総務業務, 116=経理業務.

### `list` — run first for any multi-day request
```json
{"argv": ["scripts/take100.py", "--action", "list"], "cwd": "skill"}
```
Returns `timesheets` (date, status, `id`, `no`) and `weekdays_without_timesheet` (with `all_day_leave`). Only offer days whose `all_day_leave` is false. `--status saved|pending|finished|denied` reads one tab; `--from`/`--to` narrow it.

### `check-day` — one date's working window
```json
{"argv": ["scripts/take100.py", "--action", "check-day", "--date", "2026-10-16"], "cwd": "skill"}
```
`recommended_time_in` / `recommended_time_out` come from the filed late/early/OT applications; `is_all_day_leave: true` → do not file; `ot_reason` is a good start for the OT entry's `work_content`; `data_available: false` → outside both periods, ask for times. Always surface `warnings`. An application saved only as a draft may not be visible here — if the window looks wrong, check `list-applications` and pass `--time-in`/`--time-out`.

### Entries
Each entry: `from`, `to`, `project_id`, `work_item_id`, `work_content`. Entries must **tile the window** `recommended_time_in` → `recommended_time_out` exactly: no overlaps, no gaps except 12:00–13:00. Late until 08:51, OT to 18:00:
```json
[
  {"from": "08:52", "to": "08:57", "project_id": "K202201", "work_item_id": 122, "work_content": "Tập thể dục, báo cáo buổi sáng"},
  {"from": "08:57", "to": "12:00", "project_id": "F131", "work_item_id": 128, "work_content": "Task A"},
  {"from": "13:00", "to": "18:00", "project_id": "F131", "work_item_id": 128, "work_content": "Task B"}
]
```

### `save` / `submit` (one day) and `save-many` (several days, one call)
```json
{"argv": ["scripts/take100.py", "--action", "save", "--date", "2026-10-16", "--entries", "<entries JSON>"], "cwd": "skill"}
{"argv": ["scripts/take100.py", "--action", "save-many", "--days", "[{\"date\": \"2026-10-13\", \"entries\": [...]}, {\"date\": \"2026-10-14\", \"entries\": [...]}]"], "cwd": "skill"}
```
Drafts by default; `--action submit` / `"--submit"` only when the user explicitly says submit. The command re-checks each day and applies time_in/time_out itself. `save-many` validates every day first; if any is wrong, already on file, or an all-day leave, nothing is sent and `problems` lists each date. Prefer `save-many` over repeated `save` (each call needs the user's approval).

### `submit-drafts` — submit saved timesheet drafts
```json
{"argv": ["scripts/take100.py", "--action", "submit-drafts", "--all-saved", "--dry-run"], "cwd": "skill"}
```
Dry run first, then without `--dry-run` after the user confirms; or `"--ids", "36519,36520"`. Each draft is submitted as a new application and the draft deleted (the site has no "submit this draft" request); if a draft could not be deleted, tell the user to delete it on the website.

### `delete` — remove a timesheet application
`"--action", "delete", "--application-id", "36519"` — only when the user asks.

### Escape hatches (only when the automatic result is wrong)
| Flag | Use when |
|---|---|
| `--time-in` / `--time-out` | The user states times that differ from what was filed (save/submit) |
| `--no-auto` | Skip the application lookup; plain 08:00–17:00 |
| `--skip-validation` | Entries deliberately do not tile the window — ask first |

## Reporting
Say what was detected and done, one line per day/application with its number: *"Saved draft OT 17:00–19:35 on 10-12 (Application #100-36700). Late leave 08:00–08:51 on 10-16 submitted (#100-36701)."* For refusals, relay `problems` and what to change.

## Errors
Every command prints `{"success": false, "error": ...}` and exits 1 on failure; `problems`/`on_file` mean nothing was sent.
| Error | Action |
|---|---|
| Attendance tunnel offline / unreachable | Ask for the current ngrok URL, re-run with `--base-url` |
| Login failed | Tell the user which bindings (TAKE100_* or ATTENDANCE_*) are wrong; don't retry in a loop |
| Missing credentials | Ask the user to set them in this skill's secret settings, or as environment variables (`README.md`) |
| Missing Python packages | Relay the printed `pip install` command; do not install anything yourself unless the user asks |
| Could not determine the employee code | Ask for their Mã NV, re-run attendance with `--employee` |
| HTTP 422 with `details` | The site rejected a field; relay `details` and fix that field |
