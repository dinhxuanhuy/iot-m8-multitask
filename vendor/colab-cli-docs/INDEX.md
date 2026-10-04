# Tài liệu Colab CLI (tham chiếu, không thuộc gói bàn giao)

Bản sao tài liệu chính thức của **Colab CLI** dùng để điều khiển Google Colab từ
terminal — cụ thể là để chạy khối thực nghiệm mở rộng **E3–E6** trên GPU thuê bao
(máy local không có driver NVIDIA nên không dùng được GPU tại chỗ).

* **Nguồn:** <https://github.com/googlecolab/google-colab-cli>
* **Ghi chú:** `AGENTS.md` đã đổi tên thành `AGENTS-colab-cli.md` — harness tự động nạp mọi `AGENTS.md`
  làm chỉ dẫn cho repo, mà đây là hướng dẫn nội bộ của dự án khác nên không đúng phạm vi.

**Phiên bản đã cài:** `0.7.4` (`pip install google-colab-cli`)
* **Giấy phép:** Apache-2.0 — xem `LICENSE` của repo gốc
* **Ngày sao chép:** 2026-10-03

## Nội dung

| Tệp | Nội dung |
|---|---|
| `README.md` | Tổng quan, cài đặt, chỉ mục lệnh, ví dụ |
| `AGENTS-colab-cli.md` | Kiến trúc nội bộ + **giới hạn của agent** (lệnh nào chạy được, lệnh nào cần người) |
| `docs/01_session_management.md` | Vòng đời phiên: `new`, `sessions`, `status`, `stop` |
| `docs/02_execution_and_interactive.md` | `exec`, `repl`, `console` — chạy code từ stdin/tệp/notebook |
| `docs/03_file_management.md` | `ls`, `upload`, `download`, `rm`, `edit` |
| `docs/04_automation_and_utility.md` | `auth`, `drivemount`, `install`, `log`, `usage` |
| `docs/05_run_command.md` | `colab run` — chạy script trên VM dùng-một-lần rồi tự huỷ |
| `docs/06_ssh_access.md` | SSH qua WebSocket, làm `ProxyCommand` cho IDE remote-dev |
| `docs/demos.md` | 11 kịch bản tự động hoá thực tế |
| `skills/colab-operator/SKILL.md` | Skill hướng dẫn thao tác Colab |
| `examples/finetune_run.py` | Ví dụ fine-tune trên GPU |
| `CHANGELOG.md`, `CONTRIBUTING.md`, `integration-README.md` | Nhật ký thay đổi, đóng góp, kiểm thử tích hợp |

## Ba điều rút ra khi dùng trong dự án này

**1. `~/.config` của môi trường này là read-only**, nhưng CLI bắt buộc ghi state vào
`~/.config/colab-cli/` (token, log, `sessions.json`). Cách xử lý: **chuyển `HOME`** vào
trong workspace cho mọi lệnh `colab`:

```bash
export CH="$PWD/.colab-home"
HOME="$CH" .venv-build/bin/colab sessions
```

**2. Auth là luồng copy-paste, KHÔNG có đường không tương tác.** `auth.py` đọc code bằng
`input()`, và `TOKEN_CONFIG_PATH` được tính từ `$HOME` **lúc import**. Vì mã PKCE gắn với
tiến trình, phải giữ tiến trình sống trong lúc người dùng duyệt URL. Cách làm đã kiểm
chứng: mở một **FIFO**, chạy `colab sessions < fifo` dưới dạng background job, đọc URL từ
log, rồi ghi code vào FIFO từ một lệnh khác. (Đã thử: ghi FIFO xuyên sandbox **thông**.)

**3. Lệnh nào agent chạy được.** Theo `AGENTS.md`: chạy được `new`, `status`, `stop`,
`ls`, `upload`, `download`, `install`, `exec -f`, và `repl`/`console` khi có stdin được
pipe. **Không** chạy được `auth`, `drivemount`, và `repl`/`console` tương tác — đều cần
người dùng.

## Cách dùng cho dự án

```bash
export CH="$PWD/.colab-home"

# 1. Cấp GPU (L4 cân bằng tốc độ/chi phí; T4 rẻ hơn; A100 nhanh hơn)
HOME="$CH" .venv-build/bin/colab new -s m8 --gpu L4

# 2. Đưa code lên (bundle nhỏ, kèm splits.json để dùng ĐÚNG phép chia đã đóng băng)
HOME="$CH" .venv-build/bin/colab upload -s m8 .colab-bundle/m8-bundle.zip /content/m8-bundle.zip

# 3. Chạy
cat <<'EOF' | HOME="$CH" .venv-build/bin/colab exec -s m8
import os, subprocess, sys
os.environ['M8_CONFIGS'] = 'E3'
sys.exit(subprocess.run([sys.executable, '/content/colab_run_ext.py']).returncode)
EOF

# 4. Tải kết quả về
HOME="$CH" .venv-build/bin/colab download -s m8 /content/m8run/results/summary.json \
    results/summary_all_remote.json

# 5. Giải phóng GPU
HOME="$CH" .venv-build/bin/colab stop -s m8
```

Kiểm tra hạn mức trước khi chạy dài: `HOME="$CH" .venv-build/bin/colab usage`.
