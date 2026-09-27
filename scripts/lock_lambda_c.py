"""Áp lại quy tắc khóa λ_c trên kết quả khảo sát đã có (không chạy lại mô hình).

Quy tắc: chọn một λ_c dùng chung cho mọi cấu hình mở rộng, tối đa hóa **trung bình**
Macro-F1 validation qua các cấu hình khảo sát. Quy tắc này được khai báo trước và chỉ
dùng validation — tập test không tham gia.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(r"D:\IOT\results")
SWEEP = RESULTS / "lambda_c_sweep.json"


def main() -> int:
    data = json.loads(SWEEP.read_text(encoding="utf-8"))
    recs = data["records"]
    configs = sorted({r["config"] for r in recs})
    lambdas = sorted({r["lambda_c"] for r in recs})

    per_cfg = {c: {r["lambda_c"]: r["best_val_macro_f1"] for r in recs if r["config"] == c}
               for c in configs}

    shared, score = None, -1.0
    for lam in lambdas:
        vals = [per_cfg[c].get(lam) for c in configs]
        if any(v is None for v in vals):
            continue
        mean = sum(vals) / len(vals)  # type: ignore[arg-type]
        if mean > score:
            shared, score = lam, mean

    chosen = {}
    for c in configs:
        best_lam = max(per_cfg[c], key=per_cfg[c].get)
        chosen[c] = {
            "lambda_c": shared,
            "val_macro_f1": per_cfg[c][shared],
            "own_best_lambda_c": best_lam,
            "own_best_val": per_cfg[c][best_lam],
            "epochs": data["epochs"],
            "fraction": data["fraction"],
            "seed": data["seed"],
        }

    data["shared_lambda_c"] = shared
    data["shared_mean_val"] = score
    data["chosen"] = chosen
    data["note"] = (
        "Khảo sát chỉ dùng Macro-F1 validation; test chưa được đánh giá. "
        "lambda_c được khóa DÙNG CHUNG cho mọi cấu hình theo tiêu chí: tối đa hóa trung bình "
        "Macro-F1 validation qua các cấu hình khảo sát (quy tắc khai báo trước khi chạy)."
    )
    SWEEP.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"λ_c được khóa dùng chung = {shared} (trung bình validation {score:.4f})")
    print(f"{'cấu hình':<9}{'val(λ_c khóa)':>15}{'λ_c riêng':>12}{'val riêng':>12}")
    for c in configs:
        i = chosen[c]
        print(f"{c:<9}{i['val_macro_f1']:>15.4f}{i['own_best_lambda_c']:>12}"
              f"{i['own_best_val']:>12.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
