"""Kiểm tra nhanh dữ liệu kết quả đã đủ cho mọi hình/bảng của báo cáo chưa."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m8.experiments import load_all_runs  # noqa: E402


def main() -> int:
    runs = load_all_runs(ROOT / "results" / "checkpoints")
    print("số lượt:", len(runs))
    pcts = sorted({r["spec"]["label_pct"] for r in runs.values()})
    print("các mức nhãn:", pcts, [type(p).__name__ for p in pcts])
    print("10 in pcts ->", 10 in pcts)
    for cfg in ("E1", "E2-L", "E2"):
        for pct in pcts:
            n = sum(1 for r in runs.values()
                    if r["spec"]["config"] == cfg and r["spec"]["label_pct"] == pct)
            keys_ok = all({"confusion", "macro_f1", "accuracy"} <= set(r["test"]) for r in runs.values()
                          if r["spec"]["config"] == cfg)
            print(f"  {cfg:<5} {pct:>3.0f}%  n = {n}  test keys ok = {keys_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
