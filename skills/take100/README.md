# take100 skill — setup

An agent skill for take100dot.com and the VT Check Attendance panel: checks real
check-in/check-out punches for late arrivals and unclaimed overtime, files leave and
overtime applications, and fills timesheets. `SKILL.md` is what the agent reads; this
file is for the person installing it.

## 1. Requirements
Python 3.10 or newer, then from this folder:

```
python -m pip install -r requirements.txt
```

`truststore` is required: take100dot.com sends its certificate without the intermediate,
and only the operating system's trust store can complete the chain. Windows and macOS
do; on Linux the take100 commands may fail with `CERTIFICATE_VERIFY_FAILED`.

## 2. Install the skill
| Agent | Where |
|---|---|
| Claude Code | Copy this folder to `~/.claude/skills/take100/` (so `SKILL.md` is at `~/.claude/skills/take100/SKILL.md`), then restart Claude Code. |
| Skill-runtime client (Sample Chatbot) | Zip this folder and upload it under Skills; set the four credentials in the skill's secret settings. |

## 3. Credentials
| Variable | Used for |
|---|---|
| `TAKE100_EMAIL`, `TAKE100_PASSWORD` | take100dot.com — every command |
| `ATTENDANCE_USERNAME`, `ATTENDANCE_PASSWORD` | the attendance panel — `scripts/attendance.py` only |

For Claude Code, set them as user environment variables and open a new terminal:

```
setx TAKE100_EMAIL "you@company.com"          # Windows
export TAKE100_EMAIL="you@company.com"        # macOS/Linux, in ~/.zshrc or ~/.bashrc
```

The values never go into the skill folder. To log out, delete `~/.cache/take100/` and
`~/.cache/attendance/`.

## 4. Attendance panel address
The panel sits behind an ngrok tunnel whose URL changes. The first run needs it: give
the agent the current URL (it passes `--base-url`), or set `ATTENDANCE_BASE_URL`. The
last URL that worked is remembered.

## 5. Adjust to your own work
`SKILL.md` carries the company rules (shift, payroll period, late/OT conventions) and
default timesheet content — project codes and work items under "Step C — Timesheets".
Change the defaults to your own project before filing timesheets.

## Check it works
From this folder: `python scripts/take100.py --action list` should print
`"success": true` with this period's timesheets.
