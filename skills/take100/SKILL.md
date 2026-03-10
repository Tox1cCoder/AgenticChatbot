---
name: take100-timesheet
description: Automates entering daily working time schedules on take100dot.com via direct HTTP API. Use this skill when the user provides their work schedule or content for the day (e.g., "Fill timesheet"). This approach uses direct API calls — no browser needed.
---

# Take100 Timesheet Automation

## Credentials
- Email: `thaind@v-takeuchi.vn` | Password: `Gm123123@`

## Default Rules
- **Morning warm-up** (08:00–08:05): `"Tập thể dục, báo cáo buổi sáng"`, project=**K202201**, work_item_id=**122**
- **Tuesday warm-up** (08:00–08:25): same content + `"Trực nhật vệ sinh"` (25 min instead of 5)
- Default shift: **C001** (08:00–17:00) | Skip 12:00–13:00 (lunch — no entries)
- Main tasks: project=**F131**, choose `work_item_id` based on actual work described:
  - 125 Requirement analysis | 126 Database design | 127 UI Design | 128 Logic Handle | 129 Geological data Input and Check

## Project & Work Item IDs
**K202201** (管理・その他): 122=その他, 121=会議参加, 120=面談, 119=活動管理, 118=法務業務, 117=総務業務, 116=経理業務  
**F131**: 125=Requirement analysis, 126=Database design, 127=UI Design, 128=Logic Handle, 129=Geological data Input and Check

## Workflow

### Step 1 — Check day status (late / paid leave)
Always run this first for the target date:
```bash
python skills/take100/take100_api.py --action check-day --date 2026-03-06
```

**Interpret the response:**
- `"is_all_day_leave": true` → Full day off. Do NOT fill timesheet; inform the user.
- `"is_late": true, "late_until": "HH:MM"` → Shift first entry to start at `HH:MM`. Pass `--time-in HH:MM` when saving.
- `"is_early_leave": true, "early_from": "HH:MM"` → Last entry ends at `HH:MM`. Pass `--time-out HH:MM` when saving.
- All `false` → Normal day, proceed as usual.

### Step 2 — Build entries JSON
Convert the user's work description into a JSON array. Each entry: `from`, `to`, `project_id`, `work_item_id`, `work_content`.

**Normal day example:**
```json
[
  {"from": "08:00", "to": "08:05", "project_id": "K202201", "work_item_id": 122, "work_content": "Tập thể dục, báo cáo buổi sáng"},
  {"from": "08:05", "to": "12:00", "project_id": "F131", "work_item_id": 125, "work_content": "Task A"},
  {"from": "13:00", "to": "17:00", "project_id": "F131", "work_item_id": 126, "work_content": "Task B"}
]
```

**If late until `HH:MM`:** shift warm-up to start at `HH:MM` (keep same duration), then adjust subsequent entries.

### Step 3 — Write entries to temp file
**CRITICAL — NEVER pass Vietnamese text through any shell command** (PowerShell/cmd corrupt non-ASCII before Python sees it, regardless of encoding flags).

Instead, use the `create_file` tool to write the JSON **directly to disk** with the exact content:

- File path: `c:\Users\ADMIN\Documents\Code Practice\Sample Chatbot\skills\take100\entries_tmp.json`
- Content: the JSON array from Step 2, as a plain string (the tool handles encoding correctly)

> **Note:** The script auto-deletes this file immediately after reading it — no manual cleanup needed.

Example content to pass to `create_file`:
```
[{"from":"08:00","to":"08:05","project_id":"K202201","work_item_id":122,"work_content":"Tập thể dục, báo cáo buổi sáng"},{"from":"08:05","to":"12:00","project_id":"F131","work_item_id":125,"work_content":"Task A"},{"from":"13:00","to":"17:00","project_id":"F131","work_item_id":126,"work_content":"Task B"}]
```

### Step 4 — Save or submit
```bash
# Save as draft (default — always use unless user explicitly asks to submit)
python skills/take100/take100_api.py --action save --date 2026-03-06 --entries-file skills/take100/entries_tmp.json

# If late: add --time-in HH:MM  |  If early leave: add --time-out HH:MM
python skills/take100/take100_api.py --action save --date 2026-03-06 --entries-file skills/take100/entries_tmp.json --time-in 08:06

# Submit for approval (only when user explicitly requests "submit" or "nộp")
python skills/take100/take100_api.py --action submit --date 2026-03-06 --entries-file skills/take100/entries_tmp.json

# List recent applications
python skills/take100/take100_api.py --action list
```

**On save success:** `{"success": true, "action": "save", "application": {"id": 30459, "no": "100-30459", ...}}`  
Tell user: *"Saved as draft (Application #100-30459). Review on take100dot.com and submit when ready."*

**On submit success:** `{"success": true, "action": "submit", ...}`  
Tell user: *"Submitted for approval (Application #100-XXXXX)."*

