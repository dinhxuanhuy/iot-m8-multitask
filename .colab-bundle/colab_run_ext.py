"""Chạy một LÔ cấu hình mở rộng trên Colab. Đọc danh sách từ env M8_CONFIGS.

Chạy theo lô để tải kết quả về sau mỗi lô: nếu Colab ngắt giữa chừng thì chỉ mất
lô đang chạy, các lô trước đã nằm ở máy local.
"""
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

RUN = Path("/content/m8run")
cfgs = os.environ.get("M8_CONFIGS", "E3").split()
fractions = os.environ.get("M8_FRACTIONS", "0.1").split()

print("=" * 72, flush=True)
print(f"BẮT ĐẦU LÔ: {cfgs}  (mức nhãn {fractions})", flush=True)
print("=" * 72, flush=True)

cmd = [
    sys.executable, str(RUN / "scripts" / "run_extensions.py"),
    "--configs", *cfgs,
    "--fractions", *fractions,
    "--seeds", "42", "43", "44",
]
print("lệnh:", " ".join(cmd), flush=True)

t0 = time.perf_counter()
r = subprocess.run(cmd, cwd=RUN)
dt = time.perf_counter() - t0

print()
print("=" * 72, flush=True)
print(f"LÔ {cfgs} KẾT THÚC sau {dt / 60:.1f} phút (exit={r.returncode})", flush=True)
print("=" * 72, flush=True)

files = sorted(glob.glob(str(RUN / "results" / "checkpoints_ext" / "*.json")))
print(f"Đã có {len(files)} file kết quả trong checkpoints_ext/:", flush=True)
for f in files:
    print("   ", Path(f).name, flush=True)

sys.exit(r.returncode)
