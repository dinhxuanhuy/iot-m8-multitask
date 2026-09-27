"""In bảng khảo sát λ_c và đề xuất giá trị khóa."""

from __future__ import annotations

import json
from pathlib import Path

PATH = Path(r"D:\IOT\results\lambda_c_sweep.json")


def main() -> int:
    d = json.loads(PATH.read_text(encoding="utf-8"))
    recs = d["records"]
    print(f"{'cấu hình':<9}{'λ_c':>7}{'val Macro-F1':>15}{'epoch tốt':>11}{'phút':>8}")
    for r in recs:
        print(f"{r['config']:<9}{r['lambda_c']:>7}{r['best_val_macro_f1']:>15.4f}"
              f"{r['best_epoch']:>11}{r['seconds'] / 60:>8.1f}")
    print()
    lambdas = sorted({r["lambda_c"] for r in recs})
    print("Trung bình val qua E4 và E6 theo λ_c:")
    for lam in lambdas:
        vals = [r["best_val_macro_f1"] for r in recs if r["lambda_c"] == lam]
        print(f"  λ_c={lam:<6} mean={sum(vals) / len(vals):.4f}   "
              f"({', '.join(f'{v:.4f}' for v in vals)})")
    print()
    print("Theo từng cấu hình:", json.dumps(d["chosen"], ensure_ascii=False, indent=2))
    print()
    print("Nhiệt độ/chiều đã khóa: xem m8/config.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
