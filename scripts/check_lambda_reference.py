"""So sánh đường validation của khảo sát λ_c với baseline E1 ở CÙNG số epoch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"D:\IOT")
sys.path.insert(0, str(ROOT / "src"))

from m8.experiments import load_all_runs  # noqa: E402
from m8.config import CHECKPOINT_DIR  # noqa: E402


def curve_best_at(history, n_epochs: int) -> float:
    """Macro-F1 validation tốt nhất trong ``n_epochs`` epoch đầu (chọn theo validation)."""
    best = -1.0
    for h in history[:n_epochs]:
        if h["val_macro_f1"] > best:
            best = h["val_macro_f1"]
    return best


def main() -> int:
    runs = load_all_runs(CHECKPOINT_DIR)
    print("== E1 tại mức 10% nhãn (khối chính, 100 epoch) ==")
    for tag, r in sorted(runs.items()):
        if r["spec"]["config"] != "E1" or r["spec"]["label_pct"] != 10:
            continue
        h = r["history"]
        print(f"  {tag}: val tốt nhất trong 40 epoch đầu = {curve_best_at(h, 40):.4f} | "
              f"trong 100 epoch = {curve_best_at(h, 100):.4f}")

    path = ROOT / "results" / "lambda_c_sweep.json"
    if not path.exists():
        print("chưa có lambda_c_sweep.json")
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    print("\n== Khảo sát λ_c (40 epoch, mức 10% nhãn, seed 42) ==")
    for r in data["records"]:
        print(f"  {r['config']}  λ_c={r['lambda_c']:<5} val tốt nhất = {r['best_val_macro_f1']:.4f}")
    print("\nλ_c được chọn theo validation:", json.dumps(data.get("chosen"), ensure_ascii=False))

    # tham chiếu: E1 cùng mốc 40 epoch
    ref = [curve_best_at(r["history"], 40) for r in runs.values()
           if r["spec"]["config"] == "E1" and r["spec"]["label_pct"] == 10]
    if ref:
        m = sum(ref) / len(ref)
        print(f"\nTham chiếu E1 ở 40 epoch (trung bình 3 seed, chỉ để định vị): {m:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
