# Trọng số bàn giao

Hai checkpoint trong thư mục này là **trọng số đã huấn luyện xong** của các lượt được báo cáo
trong `report/main.pdf`. Chúng được đưa vào kho mã nguồn theo yêu cầu của đề cương mục 5.2
(*"Đóng gói checkpoint được chọn bằng validation cho các lượt chính"*).

Mỗi tệp `.pt` là một `dict` gồm `{"state_dict": ..., "spec": ...}` do `src/m8/train.py:save_run`
sinh ra; `state_dict` là checkpoint **được chọn bằng Macro-F1 validation** (không phải epoch cuối).

| Tệp | Cấu hình | Hàm mục tiêu | Macro-F1 test | Accuracy test | Epoch được chọn | SHA-256 |
|---|---|---|---|---|---|---|
| `E3_L10_s42.pt` | E3 | $\mathcal{L}_{sup} + \lambda_c \mathcal{L}_{NT\text{-}Xent}$ | **0,9413** | 0,9435 | 85 / 100 | `e686c66670b517881ff418e5ba7b9ab6969bb19987f089f2042f710510ec5c87` |
| `E4_L10_s42.pt` | E4 | $\mathcal{L}_{sup} + \lambda_c \mathcal{L}_{SupCon}$ | **0,9529** | 0,9549 | 89 / 100 | `8c278f2d99624d8adc0b3c7b25dc86c58c523dd89c941ad0ca201a7354089159` |

Cả hai đều ở **mức 10 % nhãn** ($|\mathcal{D}_L| = 4.500$), **seed 42**, $\lambda_c = 1{,}0$
(giá trị đã khóa bằng validation), 100 epoch, dung lượng 43,9 MiB/tệp.

## Vì sao chỉ có hai tệp

Bảy cấu hình của báo cáo (E1–E6) chạy tổng cộng 39 lượt. Toàn bộ checkpoint của 39 lượt nặng
khoảng **1,7 GB**, vượt mức hợp lý cho một kho mã nguồn, nên ở đây chỉ đóng gói hai lượt đại
diện. Hai lượt này được chọn vì:

* **E4** là mô hình **tốt thứ nhì toàn báo cáo** (0,9529), và tốt hơn **mọi** cấu hình bắt buộc
  của đề cương (E1 0,9319 · E2-L 0,9313 · E2 0,9470). Nó đại diện cho kết luận trung tâm của
  phần mở rộng: SupCon chỉ dùng $\mathcal{D}_L$ đã vượt cả M8 khai thác 45.000 ảnh không nhãn.
* **E3** là đối chứng trực tiếp của E4 — cùng điều kiện (không nhánh xoay, mức 10 %, seed 42),
  chỉ khác **loại** hàm contrastive. Đặt cạnh nhau, hai tệp cho phép kiểm chứng ngay so sánh
  SupCon > NT-Xent mà không cần huấn luyện lại.

Hai lượt này được huấn luyện trên Google Colab L4; máy local không dùng được GPU vì driver
NVIDIA chưa nạp. Toàn bộ 39 lượt có chỉ số đầy đủ trong `results/runs_table.csv`.

## Cách nạp và kiểm chứng

```bash
pip install torch torchvision
python - <<'PY'
import torch
from pathlib import Path
from m8.models import build_model

p = Path("results/checkpoints_ext/E4_L10_s42.pt")
blob = torch.load(p, map_location="cpu", weights_only=False)
spec = blob["spec"]
print(spec)                     # config/seed/label_pct/lambda_c của lượt này

model = build_model(with_rotation_head=False, proj_dim=None)
missing = model.load_state_dict(blob["state_dict"], strict=False)
# thiếu khoá projection_head là bình thường: head đó chỉ dùng khi HUẤN LUYỆN
print("thiếu khoá:", [k for k in missing.missing_keys if "projection_head" not in k])
model.eval()
PY
```

Để tái lập chỉ số trên toàn bộ 10.000 ảnh test:

```bash
python scripts/prepare_data.py            # tải CIFAR-10 và đóng băng split
python scripts/run_error_analysis.py --tag E4_L10_s42 --ext
```

Script sẽ in ra Macro-F1/Accuracy tính lại từ chính dự đoán và so với `<tag>.json`. Trên máy
dùng CPU, sai khác cỡ **một ảnh trên 10.000** so với con số đã báo cáo (do số học dấu phẩy động
khác giữa CPU và GPU) — script in ra độ lệch thực tế chứ không chỉ báo đạt/không.

## Dự đoán test kèm theo

Không cần nạp lại trọng số cũng kiểm tra được chỉ số, vì dự đoán test theo chỉ mục đã có sẵn:

* `results/test_predictions/E3_L10_s42.npz` — Macro-F1 tính lại **0,9414**
* `results/test_predictions/E4_L10_s42.npz` — Macro-F1 tính lại **0,9529**

Mỗi tệp gồm `test_index` (0–9999, đúng thứ tự tập test chính thức, **không** xáo trộn),
`y_true`, `y_pred`, `logits` (N, 2) và `prob` (N, 2).

## Trọng số không được đóng gói

Sáu lượt còn lại của báo cáo (E1, E2-L, E2 ở cả ba mức nhãn và E5, E6) **không** kèm trọng số.
Chỉ số của chúng vẫn được bàn giao đầy đủ, và dự đoán test theo chỉ mục của khối chính nằm ở
`results/test_predictions/`. Muốn dựng lại trọng số, chạy:

```bash
python scripts/run_experiments.py --configs E1 E2-L E2 --fractions 0.1 0.2 0.5
python scripts/run_extensions.py                                  # E3–E6
```

Kết quả đã có sẽ được bỏ qua nên chỉ tốn thời gian cho phần còn thiếu.
