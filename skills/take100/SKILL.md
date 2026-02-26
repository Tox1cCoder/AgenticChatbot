---
name: take100-timesheet
description: Automates entering daily working time schedules on take100dot.com using playwright-cli and Edge. Use this skill when the user provides their work schedule or content for the day (e.g., "Fill timesheet").
---

# Take100 Timesheet Automation

## Credentials
- **Email:** `thaind@v-takeuchi.vn`  |  **Password:** `Gm123123@`

## Default Rules
- **Morning warm-up** (08:00–08:05): Content = "Tập thể dục, báo cáo buổi sáng", Project = **K202201**, Category = **その他**
- **All other tasks**: Project = **F131**, Category = **Logic Handle**

## Rules
- Every `playwright-cli` command returns a snapshot. **Never** call `playwright-cli snapshot` separately.
- If a command fails, retry once, then stop and report the error.
- **Never** search projects with an empty keyword — it loads ~4000 rows and crashes the browser.
- **STRICT ORDER**: For each entry you MUST complete steps 4a → 4b → 4c → 4d → 4e **in order**. Do NOT click "Add" until ALL four fields (times, project, category, content) are filled.
- After adding an entry, the form keeps the previous values. You MUST set the correct project/category for EACH entry — do NOT assume leftover values are correct.

---

## Step 1 — Close previous session and login
```
playwright-cli close
playwright-cli open https://take100dot.com/login --browser=msedge --persistent --headed
playwright-cli fill "Email" "thaind@v-takeuchi.vn"
playwright-cli fill "Password" "Gm123123@"
playwright-cli click "Sign in"
```
The `close` before `open` prevents creating a duplicate browser tab from a previous session. If still on `/login` after Sign in, stop and report failure.

## Step 2 — Navigate to today's application
- **Row exists for today:** click the link (with `img`) in the **last cell** of that row.
- **No row:** click **`+ Create`**.

## Step 3 — Open Works dialog
Find the table (columns: Apply | **Works** | Date | Day | Shift-work | …). Click the **button** (with `img` inside) in the **Works** column (2nd). Do NOT click Shift-work (5th column).

## Step 4 — Fill entries in "Work Details" dialog
For EACH schedule entry, do ALL of 4a–4d BEFORE clicking Add:

**4a. Times** — In the Working Time row, the **1st textbox** = FROM, the **2nd textbox** (after `~`) = TO. Identify by position, not name.
```
playwright-cli fill <from-ref> "08:00"
playwright-cli fill <to-ref> "08:05"
```

**4b. Project** — Click the 🔍 button → in the popup, fill `textbox "Keyword..."` with the project code FIRST → click the popup's search button → click the result row → click "Select".

**4c. Category** — `playwright-cli select <combobox-ref> "その他"`

**4d. Content** — `playwright-cli fill <content-ref> "Tập thể dục, báo cáo buổi sáng"`

**4e. Add** — Only AFTER all four fields above are filled, click **"Add"**.

Repeat 4a–4e for each entry. After all entries are added, click **"Save changes"**.

## Step 5 — Save and close
```
playwright-cli click "Save"
playwright-cli close
```
Tell the user: "Timesheet saved as draft. Please review and submit."