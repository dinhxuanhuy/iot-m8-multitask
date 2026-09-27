"""Báo cáo trạng thái hai khối thực nghiệm (chính + mở rộng contrastive)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"D:\IOT")
CKPT = ROOT / "results" / "checkpoints"
CKPT_EXT = ROOT / "results" / "checkpoints_ext"


def agg(rows):
    if not rows:
        return None
    m = sum(rows) / len(rows)
    if len(rows) > 1:
        var = sum((x - m) ** 2 for x in rows) / (len(rows) - 1)
        return m, var ** 0.5
    return m, 0.0


def main() -> int:
    main_runs = {}
    for p in sorted(CKPT.glob("*.json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        main_runs[j["spec"]["tag"]] = j
    ext_runs = {}
    for p in sorted(CKPT_EXT.glob("*.json")):
        j = json.loads(p.read_text(encoding="utf-8"))
        ext_runs[j["spec"]["tag"]] = j

    print(f"KHỐI CHÍNH      : {len(main_runs)}/27 lượt")
    print(f"KHỐI MỞ RỘNG    : {len(ext_runs)}/24 lượt")
    print()
    print("== Khối chính: test Macro-F1 (trung bình ± std, 3 seed) ==")
    print(f"{'cấu hình':<9}{'nhãn':>6}{'n':>4}{'test Macro-F1':>20}")
    for cfg, pct in [("E1", 10), ("E2-L", 10), ("E2", 10),
                     ("E1", 20), ("E2-L", 20), ("E2", 20),
                     ("E1", 50), ("E2-L", 50), ("E2", 50)]:
        vals = [r["test"]["macro_f1"] for r in main_runs.values()
                if r["spec"]["config"] == cfg and r["spec"]["label_pct"] == pct]
        a = agg(vals)
        if a:
            print(f"{cfg:<9}{pct:>5}%{len(vals):>4}{a[0]:>12.4f} ± {a[1]:.4f}")
    print()
    print("== Khối mở rộng: từng lượt đã xong ==")
    if not ext_runs:
        print("  (chưa có)")
    for tag, r in sorted(ext_runs.items()):
        s = r["spec"]
        print(f"  {tag:<14} λ_c={s.get('lambda_c')}  val={r['best_val']['macro_f1']:.4f}  "
              f"test={r['test']['macro_f1']:.4f}  acc={r['test']['accuracy']:.4f}  "
              f"epoch={r['best_epoch']:>3}  {r['timing']['train_seconds']:>6.0f}s")
    print()
    print("== So sánh ở mức 10% nhãn (ghép cặp theo seed) ==")
    for cfg in ("E3", "E4", "E5", "E6"):
        sel = {r["spec"]["seed"]: r for r in ext_runs.values()
               if r["spec"]["config"] == cfg and r["spec"]["label_pct"] == 10}
        if not sel:
            continue
        base = {r["spec"]["seed"]: r for r in main_runs.values()
                if r["spec"]["config"] == "E1" and r["spec"]["label_pct"] == 10}
        deltas, vals = [], []
        for seed, r in sorted(sel.items()):
            vals.append(r["test"]["macro_f1"])
            if seed in base:
                deltas.append((r["test"]["macro_f1"] - base[seed]["test"]["macro_f1"]) * 100)
        a = agg(vals)
        d = agg(deltas)
        n_pos = sum(1 for v in deltas if v > 0)
        print(f"  {cfg}: test Macro-F1 = {a[0]:.4f} ± {a[1]:.4f} | "
              f"Δ so E1 = {d[0]:+.2f} ± {d[1]:.2f} điểm % ({n_pos}/{len(deltas)} dương)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
