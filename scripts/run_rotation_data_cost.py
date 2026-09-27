"""Đo chi phí dữ liệu của hai nguồn ảnh cho nhánh xoay (không cần GPU).

Trả lời câu hỏi C3 ở phần chi phí: khi nhánh phụ dùng toàn bộ ``D_train`` (45.000 ảnh)
thay vì chỉ ``D_L``, mỗi epoch phải dựng thêm bao nhiêu view xoay?

Kết quả ghi vào ``results/rotation_data_cost.json``.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from m8.config import BATCH_SIZE, LABEL_BUDGET, RESULTS_DIR  # noqa: E402
from m8.data import RotationViewDataset, build_benchmark, get_cifar10  # noqa: E402

REPEATS = 3


def time_views(ds: RotationViewDataset, n: int) -> float:
    """Thời gian dựng ``n`` view xoay, tính bằng giây."""
    t0 = time.perf_counter()
    for i in range(n):
        ds[i % len(ds)]
    return time.perf_counter() - t0


def main() -> int:
    train, _ = get_cifar10(download=False)
    bench = build_benchmark(train.targets)

    rows = []
    for frac in sorted(LABEL_BUDGET):
        pct = int(round(frac * 100))
        dl_idx = bench.dl(42, frac)
        labeled_ds = RotationViewDataset(train, dl_idx)
        train_ds = RotationViewDataset(train, bench.train_idx)

        n_total = len(train_ds)
        # mỗi epoch của nhánh phụ xử lý đúng n_total view khi nguồn là D_train,
        # và len(labeled_ds) view khi nguồn là D_L.
        t_lab = min(time_views(labeled_ds, len(labeled_ds)) for _ in range(REPEATS))
        # chỉ đo tỷ lệ trên một mẫu rồi ngoại suy để không mất quá nhiều thời gian
        probe = 2000
        t_probe = min(time_views(train_ds, probe) for _ in range(REPEATS))
        per_view = t_probe / probe
        t_train = per_view * n_total

        rows.append(
            {
                "label_pct": pct,
                "dl_images": int(len(dl_idx)),
                "train_images": int(n_total),
                "seconds_per_epoch_rotation_from_dl": t_lab,
                "seconds_per_epoch_rotation_from_train": t_train,
                "ratio": t_train / t_lab if t_lab else float("nan"),
                "seconds_per_view": per_view,
            }
        )
        print(
            f"{pct:>3}% nhãn: D_L = {len(dl_idx):>6} view -> {t_lab:8.1f}s/epoch | "
            f"D_train = {n_total} view -> {t_train:8.1f}s/epoch "
            f"(x{t_train / t_lab:.1f})",
            flush=True,
        )

    out = RESULTS_DIR / "rotation_data_cost.json"
    out.write_text(
        json.dumps(
            {
                "note": "Thời gian dựng view xoay trên CPU (không tính forward/backward); "
                        "mỗi epoch của nhánh phụ cần đúng số view bằng kích thước tập nguồn.",
                "batch_size": BATCH_SIZE,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nĐã lưu {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
