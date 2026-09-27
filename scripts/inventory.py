"""Liệt kê sản phẩm bàn giao và tổng hợp số liệu cuối cùng."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"D:\IOT")


def size_mib(p: Path) -> float:
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024**2


def main() -> int:
    print("=== Sản phẩm chính ===")
    for rel in [
        "report/main.pdf",
        "report/main.tex",
        "notebooks/M8_CIFAR10_MultiTask.ipynb",
        "notebooks/M8_CIFAR10_MultiTask.executed.ipynb",
        "README.md",
        "requirements.txt",
    ]:
        p = ROOT / rel
        if p.exists():
            print(f"  {rel:<52} {p.stat().st_size / 1024:8.1f} KiB")
        else:
            print(f"  {rel:<52} THIẾU")

    print("\n=== Thư mục ===")
    for rel in ["src/m8", "scripts", "tests", "results/checkpoints",
                "results/figures", "report/generated", "data/artifacts"]:
        p = ROOT / rel
        n = len(list(p.rglob("*"))) if p.exists() else 0
        print(f"  {rel:<24} {n:4d} mục  {size_mib(p):8.2f} MiB")

    print("\n=== Kết quả trên D_test (trung bình 3 lần lặp) ===")
    summary = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
    print(f"  {'cấu hình':<9} {'nhãn':>5} {'Macro-F1':>18} {'Accuracy':>10} {'Δ so E1':>9}")
    for cfg in ("E1", "E2-L", "E2"):
        for pct in (10, 20, 50):
            a = summary["aggregate"].get(f"{cfg}_{pct}")
            if not a:
                continue
            d = summary["paired_delta"].get(f"{cfg}_minus_E1_{pct}")
            delta = "—" if d is None else f"{d['mean']:+.2f} pp"
            print(
                f"  {cfg:<9} {pct:>4}% "
                f"{a['test_macro_f1']['mean']:.4f} ± {a['test_macro_f1']['std']:.4f} "
                f"{a['test_accuracy']['mean']:>9.4f} {delta:>9}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
