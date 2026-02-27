---
name: Tự động hóa Bước 1 - Vẽ đường trục (Grid Lines Drawing) qua AutoCAD MCP
description: Skill này định nghĩa các nguyên tắc chung và quy trình làm việc (workflow) để LLM Agent sử dụng các công cụ MCP kết nối với AutoCAD, tự động hóa việc tính toán tọa độ và vẽ hệ thống lưới trục (grid system) dựa trên số liệu phác thảo (Sketch) của công trình TNF. Kích hoạt skill này khi người dùng yêu cầu vẽ hệ thống đường trục cho công trình và đã cung cấp mã kết nối `shortHex` từ AutoCAD plugin.
---

### 1. ĐIỀU KIỆN TIÊN QUYẾT (PREREQUISITES)
* Mọi lời gọi Tool (Tool calls) **BẮT BUỘC** phải bao gồm mã `shortHex` để broker định tuyến đúng phiên làm việc của người dùng. 
* Nếu người dùng chưa cung cấp `shortHex`, Agent phải chủ động hỏi lại mã này trước khi tiến hành bất kỳ bước nào khác.

### 2. QUY TRÌNH THỰC THI QUA MCP TOOLS (AGENT WORKFLOW)
Trợ lý LLM phải tuân thủ nghiêm ngặt luồng làm việc 5 bước sau:

* **Bước 1: Lấy ngữ cảnh (Context Snapshot):**
    Gọi tool `autocad_get_context(shortHex)` trước khi lên bất kỳ kế hoạch hành động nào.
    *Mục đích:* Đọc hiểu các layer hiện có, trạng thái bản vẽ hiện tại và xác định các handle (nếu cần chỉnh sửa đối tượng cũ).
* **Bước 2: Phân tích & Tính toán tọa độ:**
    Yêu cầu người dùng cung cấp khoảng cách trục (spans) theo phương X và phương Y (nếu chưa có). Dựa trên dữ liệu này, Agent tự động tính toán chính xác tọa độ điểm đầu `[x, y, z]` và điểm cuối `[x, y, z]` cho từng đường Line của trục X và trục Y.
* **Bước 3: Tạo tập lệnh (Produce actionsJson):**
    Sinh ra chuỗi JSON nghiêm ngặt để tạo các đường Line dựa trên tọa độ đã tính toán. Định dạng JSON hợp lệ phải là Array hoặc Wrapper.
    *Ví dụ:* `[ { "method":"create", "params": { ... } } ]` hoặc `{ "actions": [ ... ] }`.
* **Bước 4: Thực thi:**
    Gọi tool `autocad_apply_actions(shortHex, actionsJson)` để tiến hành vẽ trực tiếp các đường trục lên bản vẽ AutoCAD của người dùng.
* **Bước 5: Xác nhận kết quả & Chuyển bước:**
    Gọi lại `autocad_get_context(shortHex)` để đảm bảo các đường trục đã được vẽ thành công. Sau đó, thông báo hoàn tất và nhắc nhở người dùng tự thực hiện bước Dim (đo kích thước) hoặc đề xuất hướng dẫn bước tiếp theo.

### 3. GENERAL RULES & INSTRUCTIONS (QUY TẮC BẮT BUỘC)

**Quy tắc về Layer & Bản vẽ:**
* Tất cả các đường trục phải được gán chính xác vào Layer mang tên: `"TP 0-F 基準線"`.
* **Đặc biệt lưu ý:** Tên Layer và các tham số (`params`) trong `actionsJson` là phân biệt chữ hoa chữ thường (case-sensitive). Phải viết chính xác tuyệt đối.

**Quy tắc về Handles & JSON Action:**
* **Không tự bịa đặt Handles (Never invent handles):** Các Handles là chuỗi Hex (VD: `"1EA8"`). Mọi thao tác chỉnh sửa/xóa/array đối tượng có sẵn phải dựa trên handles lấy được từ `context.selected`.
* **Định dạng JSON:** `actionsJson` chỉ được chứa định dạng JSON hợp lệ, tuyệt đối không gửi nested JSON object thay cho JSON string.
    *Cấu trúc tạo Line chuẩn:* `{"method": "create", "params": {"type": "line", "layer": "TP 0-F 基準線", "start": [...], "end": [...]}}`

**Quy tắc vòng đời hành động (Action Lifecycle):**
* Sau khi tạo hoặc chỉnh sửa một đối tượng (VD: tạo một đường trục), đối tượng đó sẽ **chỉ xuất hiện kèm Handle** trong lần chụp ngữ cảnh (`context capture`) tiếp theo. 
* *Do đó:* Hãy áp dụng hành động tạo mới, sau đó đợi lấy ngữ cảnh mới (`autocad_get_context`) để trích xuất Handle nếu muốn tiếp tục các thao tác nâng cao trên chính đối tượng đó (như Move, Offset).