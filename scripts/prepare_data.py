"""Tải CIFAR-10 và kiểm tra tính toàn vẹn của phép chia benchmark (tuần 1)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m8.config import ARTIFACT_DIR, LABEL_BUDGET, RESULTS_DIR, SPLIT_SEED  # noqa: E402
from m8.data import audit_benchmark, build_benchmark, get_cifar10  # noqa: E402


def main() -> int:
    print("== Tải CIFAR-10 ==", flush=True)
    train, test = get_cifar10(download=True)
    print(f"trainset = {len(train)} ảnh | testset = {len(test)} ảnh", flush=True)

    print("== Chia benchmark (seed 2026, lần lặp 42/43/44) ==", flush=True)
    bench = build_benchmark(train.targets, split_seed=SPLIT_SEED)
    report = audit_benchmark(bench, train.targets)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    out = ARTIFACT_DIR / "benchmark_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = bool(report["no_overlap_train_val"])
    for seed_info in report["nesting"].values():
        ok = ok and bool(seed_info["nested"])
    for info in report["label_budget"].values():
        ok = ok and bool(info["disjoint"]) and bool(info["no_val_leak"])

    # đối chiếu số lượng với bảng 1 của đề cương
    for key, info in report["label_budget"].items():
        pct = int(key.split("_")[1])
        budget = LABEL_BUDGET[pct / 100.0]
        if info["dl"] != budget["total"] or info["du"] != budget["unlabeled"]:
            print(f"SAI số lượng tại {key}", flush=True)
            ok = False

    print("KẾT LUẬN:", "ĐẠT" if ok else "KHÔNG ĐẠT", flush=True)
    print(f"Đã lưu: {ARTIFACT_DIR / 'splits.json'}", flush=True)
    print(f"Đã lưu: {out}", flush=True)

    # Đề cương mục 5.2 yêu cầu bàn giao "danh sách chỉ mục split/subset". Thư mục data/
    # bị .gitignore (chứa CIFAR-10 tải về), nên sao chép chỉ mục đã đóng băng và kết quả
    # kiểm tra che nhãn sang results/ để phần bàn giao có đủ. Bản trong results/ là bản
    # sao CHÍNH XÁC của tệp mà mã nguồn dùng.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ARTIFACT_DIR / "splits.json", RESULTS_DIR / "split_indices.json")
    shutil.copyfile(out, RESULTS_DIR / "benchmark_audit.json")
    print(f"Đã sao chép sang gói bàn giao: {RESULTS_DIR / 'split_indices.json'}", flush=True)
    print(f"Đã sao chép sang gói bàn giao: {RESULTS_DIR / 'benchmark_audit.json'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
