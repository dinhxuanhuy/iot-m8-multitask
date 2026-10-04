"""Chạy TRÊN COLAB: huấn luyện 3 cấu hình chính ở mức 10 % seed 42, rồi xuất dự đoán test.

Mục đích: đề cương mục 5.2 yêu cầu *"ma trận nhầm lẫn và một số trường hợp dự đoán sai"* cùng
*"dự đoán test theo chỉ mục để kiểm tra lại chỉ số"*. Hai thứ này cần checkpoint của khối chính,
mà `results/checkpoints/` không kèm trong kho mã nguồn.

Lượt chạy ở đây dùng ĐÚNG giao thức và ĐÚNG seed của khối chính, nên chỉ số thu được là của
chính lượt `E1_L10_s42`, `E2-L_L10_s42`, `E2_L10_s42` đã báo cáo (sai khác chỉ do số học
dấu phẩy động giữa hai thiết bị, ở mức vài ảnh trên 10.000).

Xuất ra:
  * results/checkpoints/<tag>.json|.pt
  * results/test_predictions/<tag>.npz   (test_index, y_true, y_pred, logits, prob)

Cách dùng (trên VM):
    python /content/colab_error_analysis.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

RUN = Path("/content/m8run")
CONFIGS = ["E1", "E2-L", "E2"]
FRACTION = "0.1"
SEED = "42"

print("=" * 74, flush=True)
print(f"BƯỚC 1 — Huấn luyện khối chính: {CONFIGS} @ 10 % nhãn, seed {SEED}", flush=True)
print("=" * 74, flush=True)

t0 = time.perf_counter()
r = subprocess.run(
    [sys.executable, str(RUN / "scripts" / "run_experiments.py"),
     "--configs", *CONFIGS, "--fractions", FRACTION, "--seeds", SEED],
    cwd=RUN,
)
print(f"--- huấn luyện xong sau {(time.perf_counter()-t0)/60:.1f} phút (exit={r.returncode})",
      flush=True)

print()
print("=" * 74, flush=True)
print("BƯỚC 2 — Xuất dự đoán test theo chỉ mục", flush=True)
print("=" * 74, flush=True)

r2 = subprocess.run(
    [sys.executable, str(RUN / "scripts" / "run_error_analysis.py"),
     "--configs", *CONFIGS, "--fraction", FRACTION, "--seed", SEED],
    cwd=RUN,
)
print(f"--- exit={r2.returncode}", flush=True)

print()
print("=" * 74, flush=True)
print("BƯỚC 3 — Kiểm kê tệp để tải về", flush=True)
print("=" * 74, flush=True)
for sub in ("checkpoints", "test_predictions"):
    d = RUN / "results" / sub
    files = sorted(d.glob("*")) if d.is_dir() else []
    print(f"{sub}/: {len(files)} tệp", flush=True)
    for f in files:
        print(f"    {f.name}  ({f.stat().st_size/1024:.0f} KB)", flush=True)

sys.exit(0 if (r.returncode == 0 and r2.returncode == 0) else 1)
