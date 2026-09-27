"""Đo Rotation Accuracy chẩn đoán trên validation cho một số lượt chạy đã có.

Giao thức: đánh giá *đủ bốn phép xoay* cho mỗi ảnh validation (mục 4.3), tức mỗi ảnh
đóng góp 4 mẫu vào mẫu số. Chỉ số này chỉ mang tính chẩn đoán cho nhánh phụ.

Kết quả ghi vào ``results/rotation_diagnostic.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from m8.config import CHECKPOINT_DIR, RESULTS_DIR  # noqa: E402
from m8.data import build_benchmark, get_cifar10, make_run_loaders  # noqa: E402
from m8.evaluate import rotation_accuracy_full  # noqa: E402
from m8.models import build_model  # noqa: E402
from m8.train import make_optimizer  # noqa: E402


def main() -> int:
    pairs = []
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    if argv:
        for tag in argv:
            path = Path(CHECKPOINT_DIR) / f"{tag}.pt"
            if path.exists():
                pairs.append(path)
    else:
        # mặc định: mọi lượt E2-L và E2 ở mức 10% nhãn (không trùng lặp)
        seen = set()
        for p in sorted(Path(CHECKPOINT_DIR).glob("*L10_*.pt")):
            if p.stem.startswith(("E2-L_", "E2_")):
                seen.add(p)
        pairs = sorted(seen)

    if not pairs:
        print("Không tìm thấy checkpoint nào.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_base, test_base = get_cifar10(download=False)
    bench = build_benchmark(train_base.targets)
    rows = []
    for ckpt_path in pairs:
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        spec = blob["spec"]
        model = build_model(with_rotation_head=True, seed=spec["seed"], device=device)
        model.load_state_dict(blob["state_dict"])
        loaders = make_run_loaders(
            train_base, test_base, bench, spec["seed"], spec["label_fraction"],
            spec["rotation_source"],
        )
        acc = rotation_accuracy_full(model, loaders["val_loader"], device)
        rows.append(
            {
                "tag": spec["tag"],
                "config": spec["config"],
                "label_pct": spec["label_pct"],
                "seed": spec["seed"],
                "rotation_accuracy": float(acc),
            }
        )
        print(f"{spec['tag']:<22} Rotation Accuracy = {acc:.4f}", flush=True)

    out = RESULTS_DIR / "rotation_diagnostic.json"
    out.write_text(
        json.dumps({"protocol": "đánh giá đủ 4 phép xoay cho mỗi ảnh validation",
                    "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nĐã lưu {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
