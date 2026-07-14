# Task Plans API

Tài liệu cho nhóm FE tích hợp module **Task Plans** (danh sách công việc/plan gắn với 1 conversation).

## 1. Thông tin chung

| Mục | Giá trị |
|---|---|
| Base URL | không có prefix riêng, gọi thẳng path bên dưới (VD: `{HOST}:{PORT}/task-plans/{task_id}`) |
| Auth | Bắt buộc header `Authorization: Bearer <access_token>` (JWT lấy từ API login) trên **tất cả** endpoint bên dưới |
| Content-Type | `application/json` cho các request có body |
| Field naming | Request/response body dùng **camelCase** (`conversationId`, `taskOrder`, `taskMetadata`, `userMessage`, `taskDescriptions`...) |

### Response envelope (áp dụng cho mọi endpoint)

```json
{
  "success": true,
  "message": "Task plan retrieved successfully",
  "data": { }
}
```

- `success`: `true`/`false`.
- `message`: mô tả kết quả, hiển thị được cho user nếu cần.
- `data`: payload thực tế — kiểu tùy endpoint (object, array, hoặc `null`).
- Khi lỗi, response có thêm `code` (string, machine-readable) và `error` (object chi tiết field lỗi nếu là lỗi validate), `data` sẽ vắng mặt/`null`.

### Response khi lỗi

| HTTP status | code | Khi nào xảy ra |
|---|---|---|
| 401/403 | `http_error` | Thiếu/sai token |
| 404 | `not_found` | Không tìm thấy `conversation_id` / `task_id` |
| 422 | `invalid_input` | Body/param không hợp lệ — `error` chứa map `{ "field.path": ["message"] }` |
| 500 | `internal_server_error` | Lỗi hệ thống |

Ví dụ lỗi validate:
```json
{
  "success": false,
  "code": "invalid_input",
  "message": "Invalid input",
  "error": { "userMessage": ["String should have at least 1 character"] }
}
```

### Enum

**`status`** (trạng thái 1 task):
`pending` | `in_progress` | `completed` | `skipped`

**`planLifecycle`** (trạng thái toàn bộ plan, trong `planning-status`):
`draft` | `ready` | `executing` | `paused` | `completed`

---

## 2. Danh sách endpoint

### 2.1 Tạo task plan bằng AI — `POST /conversations/{conversation_id}/task-plans`

AI planning agent phân tích yêu cầu người dùng và tự sinh danh sách task có thứ tự.

**Request body**
```json
{
  "userMessage": "Break down the current architecture into implementation tasks."
}
```
| Field | Kiểu | Bắt buộc | Ghi chú |
|---|---|---|---|
| `userMessage` | string | ✓ | tối thiểu 1 ký tự |

**Response** — `201 Created`, `data`: mảng `TaskPlan` (xem mục 3)

---

### 2.2 Tạo task plan thủ công — `POST /conversations/{conversation_id}/task-plans/manual`

Tạo task từ danh sách mô tả do FE/người dùng nhập tay. Nếu conversation đã có plan, task mới được **append** vào cuối (nối tiếp `taskOrder`).

**Request body**
```json
{
  "taskDescriptions": [
    "Document the server architecture",
    "Document the client runtime bridge",
    "Update the API examples"
  ]
}
```
| Field | Kiểu | Bắt buộc | Ghi chú |
|---|---|---|---|
| `taskDescriptions` | string[] | ✓ | tối thiểu 1 phần tử |

**Response** — `201 Created`, `data`: mảng `TaskPlan` (chỉ các task vừa tạo)

---

### 2.3 Lấy danh sách task theo conversation — `GET /conversations/{conversation_id}/task-plans`

| Query param | Kiểu | Mặc định | Ghi chú |
|---|---|---|---|
| `include_completed` | boolean | `false` | `false` → ẩn task `completed` và `skipped`; `true` → trả tất cả |

**Response** — `200 OK`, `data`: mảng `TaskPlan`

---

### 2.4 Lấy chi tiết 1 task — `GET /task-plans/{task_id}`

**Response** — `200 OK`, `data`: 1 object `TaskPlan`

---

### 2.5 Cập nhật task — `PATCH /task-plans/{task_id}`

Tất cả field đều optional, chỉ gửi field muốn đổi.

**Request body**
```json
{
  "status": "in_progress",
  "taskMetadata": { "owner": "docs" }
}
```
| Field | Kiểu | Ghi chú |
|---|---|---|
| `description` | string \| null | tối thiểu 1 ký tự nếu có |
| `status` | string \| null | 1 trong 4 giá trị enum `status` |
| `taskMetadata` | object \| null | metadata tự do (key-value) |
| `completedAt` | datetime (ISO 8601) \| null | thời điểm hoàn thành |

**Response** — `200 OK`, `data`: object `TaskPlan` sau update

---

### 2.6 Đánh dấu hoàn thành — `POST /task-plans/{task_id}/complete`

Không cần body. Set `status = completed` và ghi `completedAt`.

**Response** — `200 OK`, `data`: object `TaskPlan` sau update

---

### 2.7 Xóa task — `DELETE /task-plans/{task_id}`

Không cần body.

**Response** — `200 OK`, `data`: `null`

---

### 2.8 Trạng thái tiến độ plan — `GET /conversations/{conversation_id}/planning-status`

**Response** — `200 OK`
```json
{
  "success": true,
  "message": "Planning status retrieved successfully",
  "data": {
    "planningModeEnabled": true,
    "planLifecycle": "executing",
    "totalTasks": 5,
    "pendingTasks": 2,
    "inProgressTasks": 1,
    "completedTasks": 2,
    "skippedTasks": 0,
    "progressPercentage": 40.0,
    "nextTask": { "...": "TaskPlan object, có thể null nếu không còn task nào chờ" }
  }
}
```
| Field | Kiểu | Ghi chú |
|---|---|---|
| `planningModeEnabled` | boolean | conversation có bật planning mode không |
| `planLifecycle` | string \| null | xem enum ở mục 1 |
| `totalTasks` / `pendingTasks` / `inProgressTasks` / `completedTasks` / `skippedTasks` | int | đếm theo trạng thái |
| `progressPercentage` | float (0–100) | % hoàn thành |
| `nextTask` | `TaskPlan` \| null | task tiếp theo cần làm |

---

## 3. Object `TaskPlan`

```json
{
  "id": "b3f1c2a4-...-uuid",
  "conversationId": "a1b2c3d4-...-uuid",
  "taskOrder": 0,
  "description": "Document the server architecture",
  "status": "pending",
  "taskMetadata": {},
  "createdAt": "2026-07-14T08:30:00Z",
  "updatedAt": "2026-07-14T08:30:00Z",
  "completedAt": null
}
```

| Field | Kiểu | Ghi chú |
|---|---|---|
| `id` | UUID | |
| `conversationId` | UUID | |
| `taskOrder` | int | thứ tự hiển thị, bắt đầu từ 0 |
| `description` | string | |
| `status` | string | xem enum `status` |
| `taskMetadata` | object | mặc định `{}` |
| `createdAt` / `updatedAt` | datetime ISO 8601 | |
| `completedAt` | datetime \| null | |

---

## 4. Bảng tổng hợp nhanh

| # | Method | Path | Body | Trả về |
|---|---|---|---|---|
| 1 | POST | `/conversations/{conversation_id}/task-plans` | `userMessage` | `TaskPlan[]` |
| 2 | POST | `/conversations/{conversation_id}/task-plans/manual` | `taskDescriptions[]` | `TaskPlan[]` |
| 3 | GET | `/conversations/{conversation_id}/task-plans?include_completed=` | — | `TaskPlan[]` |
| 4 | GET | `/task-plans/{task_id}` | — | `TaskPlan` |
| 5 | PATCH | `/task-plans/{task_id}` | `description?`, `status?`, `taskMetadata?`, `completedAt?` | `TaskPlan` |
| 6 | POST | `/task-plans/{task_id}/complete` | — | `TaskPlan` |
| 7 | DELETE | `/task-plans/{task_id}` | — | `null` |
| 8 | GET | `/conversations/{conversation_id}/planning-status` | — | `PlanningStatus` |
