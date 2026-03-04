---
name: take100-timesheet
description: Automates entering daily working time schedules on take100dot.com via direct HTTP API. Use this skill when the user provides their work schedule or content for the day (e.g., "Fill timesheet"). This approach uses a single API call instead of browser automation — fast, reliable, and token-efficient.
---

# Take100 Timesheet Automation (API-based)

## Overview
This skill fills timesheets on take100dot.com by calling a Python script that uses the site's REST API directly — **no browser needed**. One command saves the entire day's entries.

## Credentials
- **Email:** `thaind@v-takeuchi.vn`  |  **Password:** `Gm123123@`

## Default Rules
- **Morning warm-up** (08:00–08:05): Content = "Tập thể dục, báo cáo buổi sáng", Project = **K202201**, work_item_id = **122** (その他)
- **All other tasks**: Project = **F131**, work_item_id = **128** (Logic Handle)
- Default shift: C001 (08:00–17:00)
- Lunch break: 12:00–13:00 (do NOT create entries during this time)

## Known Project & Category IDs

### Project K202201 (管理・その他) — project_class_id=5
| work_item_id | Category Name |
|---|---|
| 116 | 経理業務 |
| 117 | 総務業務 |
| 118 | 法務業務 |
| 119 | 活動管理 |
| 120 | 面談 |
| 121 | 会議参加 |
| 122 | その他 |

### Project F131 (TAKEUCHI AI) — project_class_id=6
| work_item_id | Category Name |
|---|---|
| 125 | Requirement analysis |
| 126 | Database design |
| 127 | UI Design |
| 128 | Logic Handle |

## How to Use

### Step 1 — Parse the user's schedule into entries
Convert the user's description into a JSON array. Each entry needs:
- `from` / `to`: time strings like `"08:00"`, `"12:00"`
- `project_id`: e.g. `"K202201"` or `"F131"`
- `work_item_id`: integer ID from the tables above
- `work_content`: description text

**Example entries for a typical day:**
```json
[
  {"from": "08:00", "to": "08:05", "project_id": "K202201", "work_item_id": 122, "work_content": "Tập thể dục, báo cáo buổi sáng"},
  {"from": "08:05", "to": "12:00", "project_id": "F131", "work_item_id": 128, "work_content": "Develop chatbot AI features"},
  {"from": "13:00", "to": "17:00", "project_id": "F131", "work_item_id": 128, "work_content": "Develop chatbot AI features"}
]
```

### Step 2 — Write entries to a temp JSON file (required for Vietnamese/Unicode content)

Windows terminal mangles UTF-8 characters when passed as command-line arguments. Always write entries to a temp file first:

```bash
# Write the entries JSON to a temp file with UTF-8 encoding
Set-Content -Path "skills/take100/entries_tmp.json" -Value '[{"from":"08:00","to":"08:05","project_id":"K202201","work_item_id":122,"work_content":"Tập thể dục, báo cáo buổi sáng"},{"from":"08:05","to":"12:00","project_id":"F131","work_item_id":128,"work_content":"Develop chatbot AI features"},{"from":"13:00","to":"17:00","project_id":"F131","work_item_id":128,"work_content":"Develop chatbot AI features"}]' -Encoding utf8
```

Or using Python to write the file (works in both PowerShell and cmd):
```bash
python -c "import json; open('skills/take100/entries_tmp.json','w',encoding='utf-8').write(json.dumps([{'from':'08:00','to':'08:05','project_id':'K202201','work_item_id':122,'work_content':'Tập thể dục, báo cáo buổi sáng'},{'from':'08:05','to':'12:00','project_id':'F131','work_item_id':128,'work_content':'Develop chatbot AI features'},{'from':'13:00','to':'17:00','project_id':'F131','work_item_id':128,'work_content':'Develop chatbot AI features'}],ensure_ascii=False))"
```

### Step 3 — Call the API script with `--entries-file`

**Save as draft** (recommended — lets user review before submitting):
```bash
python skills/take100/take100_api.py --action save --date 2026-03-03 --entries-file skills/take100/entries_tmp.json
```

**Submit for approval** (saves AND submits — use only when user explicitly asks):
```bash
python skills/take100/take100_api.py --action submit --date 2026-03-03 --entries-file skills/take100/entries_tmp.json
```

**List recent applications:**
```bash
python skills/take100/take100_api.py --action list
```

**Delete an application** (note: delete from the site directly — the API delete endpoint requires Bearer auth):
```
https://take100dot.com/wt-applications/{id}
```

### Step 3 — Report result
The script outputs JSON. On success:
```json
{"success": true, "action": "save", "application": {"id": 30459, "no": "100-30459", ...}}
```
Tell the user: "Timesheet saved as draft (Application #{no}). Please review on take100dot.com and submit when ready."

## Rules
- Always include the morning warm-up entry (08:00–08:05) unless the user explicitly says otherwise.
- Skip 12:00–13:00 (lunch break) — entries must not overlap this gap.
- Default to `--action save` (draft). Only use `submit` when the user explicitly requests submission.
- If the user doesn't specify detailed times, split evenly between morning (08:05–12:00) and afternoon (13:00–17:00).
- **Always use `--entries-file`** (never `--entries` on the command line) to avoid Windows terminal mangling Vietnamese/Unicode characters.
- Write the JSON file using Python (`open(..., encoding='utf-8')`) or `Set-Content -Encoding utf8` in PowerShell.
- The temp file `skills/take100/entries_tmp.json` is overwritten on each use — no cleanup needed.
