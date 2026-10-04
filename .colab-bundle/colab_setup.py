"""Bước 1 trên Colab: kiểm tra GPU, giải nén bundle, tải CIFAR-10, smoke test.

Không thuộc gói bàn giao — dùng để chuẩn bị VM trước khi chạy 12 lượt thật.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

WORK = Path("/content/m8")
BUNDLE = Path("/content/m8-bundle.zip")

print("=" * 72)
print("1. KIỂM TRA GPU")
print("=" * 72)
import torch  # noqa: E402
import torchvision  # noqa: E402

print(f"  torch        : {torch.__version__}")
print(f"  torchvision  : {torchvision.__version__}")
print(f"  cuda avail   : {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("!!! KHÔNG CÓ GPU. Vào Runtime > Change runtime type > chọn GPU rồi chạy lại.")
print(f"  device       : {torch.cuda.get_device_name(0)}")
print(f"  capability   : {torch.cuda.get_device_capability(0)}")
print(f"  VRAM         : {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")

print()
print("=" * 72)
print("2. GIẢI NÉN BUNDLE")
print("=" * 72)
if not BUNDLE.exists():
    sys.exit(f"!!! Không thấy {BUNDLE}. Phải `colab upload` bundle lên trước.")
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
with zipfile.ZipFile(BUNDLE) as z:
    z.extractall(WORK)
root = WORK / "bundle"
print(f"  giải nén vào : {root}")

# Đưa về đúng cấu trúc mà src/m8/config.py mong đợi:
# PROJECT_ROOT = parents[2] của src/m8/config.py
final = Path("/content/m8run")
if final.exists():
    shutil.rmtree(final)
final.mkdir(parents=True)
for sub in ("src", "scripts", "results", "data"):
    if (root / sub).exists():
        shutil.move(str(root / sub), str(final / sub))
print(f"  cấu trúc     : {final}")
for p in sorted(final.rglob("*")):
    if p.is_file() and "data/artifacts" not in str(p):
        print(f"     {p.relative_to(final)}")
print(f"     data/artifacts/splits.json ({(final / 'data/artifacts/splits.json').stat().st_size / 1e6:.1f} MB)")

sys.path.insert(0, str(final / "src"))
from m8.config import (  # noqa: E402
    DATA_DIR, LABEL_BUDGET, RESULTS_DIR, extension_run_specs,
)

print()
print("=" * 72)
print("3. KIỂM TRA SPLIT ĐÃ ĐÓNG BĂNG (phải khớp máy đã chạy khối chính)")
print("=" * 72)
from m8.data import build_benchmark, get_cifar10  # noqa: E402

train, test = get_cifar10(download=True)
print(f"  CIFAR-10     : train={len(train)}  test={len(test)}  -> {DATA_DIR}")
bench = build_benchmark(train.targets)
print(f"  D_train      : {len(bench.train_idx)}")
print(f"  D_val        : {len(bench.val_idx)}")
print(f"  D_test       : {len(bench.test_idx)}")
for frac in sorted(LABEL_BUDGET):
    dl = bench.dl(42, frac)
    du = bench.du(42, frac)
    print(f"  {int(frac*100):>3d}% seed42   D_L={len(dl):>6d}  D_U={len(du):>6d}")

assert len(bench.train_idx) == 45000, "D_train phải 45.000"
assert len(bench.dl(42, 0.10)) == 4500, "D_L 10% phải 4.500"

print()
print("=" * 72)
print("4. DANH SÁCH LƯỢT MỞ RỘNG SẼ CHẠY")
print("=" * 72)
specs = extension_run_specs(fractions=[0.1])
print(f"  tổng số lượt : {len(specs)}")
for s in specs:
    print(f"    {s.tag:16s} config={s.config:3s} label={s.label_pct}% seed={s.seed} "
          f"rot={s.rotation_source:6s} con={s.contrastive_kind}@{s.contrastive_source} "
          f"lambda_c={s.lambda_c}")

print()
print("=" * 72)
print("5. SMOKE TEST — 1 lượt E4, 2 epoch (kiểm tra đường ống, KHÔNG dùng cho báo cáo)")
print("=" * 72)
cmd = [
    sys.executable, str(final / "scripts" / "run_extensions.py"),
    "--configs", "E4", "--fractions", "0.1", "--seeds", "42",
    "--epochs", "2", "--tag", "smoke", "--no-resume",
]
t0 = time.perf_counter()
r = subprocess.run(cmd, cwd=final, capture_output=True, text=True)
print(r.stdout[-3000:])
if r.returncode != 0:
    print(r.stderr[-3000:])
    sys.exit(f"!!! SMOKE TEST THẤT BẠI (exit {r.returncode})")
print(f"  smoke test OK trong {time.perf_counter()-t0:.0f}s")
print()
print("=> VM SẴN SÀNG. Chạy tiếp colab_run_ext.py để chạy 12 lượt thật.")
