"""Chạy khối mở rộng contrastive E3--E6 (mục 3.3, bảng 5 của đề cương).

Ví dụ::

    .\\.venv\\Scripts\\python.exe scripts\\run_extensions.py
    .\\.venv\\Scripts\\python.exe scripts\\run_extensions.py --epochs 5 --seeds 42
    .\\.venv\\Scripts\\python.exe scripts\\run_extensions.py --configs E4 E6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m8.config import (  # noqa: E402
    EXTENSION_CONFIGS,
    EXTENSION_LABEL_FRACTIONS,
    LOG_DIR,
    REPETITION_SEEDS,
    RESULTS_DIR,
    extension_run_specs,
)
from m8.experiments import load_all_runs, run_suite, summarize_runs  # noqa: E402

EXT_DIR = RESULTS_DIR / "checkpoints_ext"


def main() -> int:
    ap = argparse.ArgumentParser(description="Khối mở rộng contrastive E3--E6")
    ap.add_argument("--configs", nargs="*", default=None, help="ví dụ E3 E4 E5 E6")
    ap.add_argument("--fractions", nargs="*", type=float, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="ghi đè số epoch (thăm dò)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--lambda-c", type=float, default=None,
                    help="ghi đè λ_c đã khóa (chỉ dùng cho thăm dò)")
    ap.add_argument("--tag", type=str, default="", help="hậu tố thư mục kết quả")
    args = ap.parse_args()

    out_dir = EXT_DIR if not args.tag else RESULTS_DIR / f"checkpoints_ext_{args.tag}"
    specs = extension_run_specs(
        fractions=args.fractions,
        seeds=args.seeds,
        configs=args.configs,
    )
    # λ_c đã khóa bằng validation (xem scripts/sweep_lambda_c.py + lock_lambda_c.py).
    # Không dùng giá trị khác trừ khi người dùng chỉ định rõ.
    locked = None
    sweep_path = RESULTS_DIR / "lambda_c_sweep.json"
    if sweep_path.exists():
        sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
        locked = sweep.get("shared_lambda_c")
    if args.lambda_c is not None:
        locked = args.lambda_c
        print(f"CẢNH BÁO: ghi đè λ_c = {locked} (không dùng giá trị đã khóa)", flush=True)
    elif locked is None:
        print("CẢNH BÁO: chưa có khảo sát λ_c; dùng giá trị mặc định "
              f"{specs[0].lambda_c}", flush=True)
    if locked is not None:
        for s in specs:
            s.lambda_c = float(locked)
        print(f"λ_c khóa dùng chung = {locked}", flush=True)

    if args.epochs:
        for s in specs:
            s.epochs = args.epochs
    if args.tag:
        for s in specs:
            s.tag = f"{s.tag}_{args.tag}"

    print(f"Sẽ chạy {len(specs)} lượt mở rộng; thư mục: {out_dir}", flush=True)
    t0 = time.perf_counter()
    run_suite(
        specs=specs,
        out_dir=out_dir,
        results_dir=RESULTS_DIR,
        amp=not args.no_amp,
        resume=not args.no_resume,
        limit=args.limit,
        verbose=True,
    )
    print(f"\nTổng thời gian: {(time.perf_counter() - t0) / 60:.1f} phút", flush=True)

    # tổng hợp: gộp khối chính và khối mở rộng để tính Δ so với E1 trên cùng mức nhãn
    from m8.config import CHECKPOINT_DIR

    runs = {}
    runs.update(load_all_runs(CHECKPOINT_DIR))
    runs.update(load_all_runs(out_dir))
    summary = summarize_runs(runs)
    out = RESULTS_DIR / ("summary.json" if not args.tag else f"summary_{args.tag}.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu tổng hợp: {out}", flush=True)

    # bảng Δ của khối mở rộng so với E1
    print("\nChênh lệch ghép cặp so với E1 (điểm phần trăm Macro-F1):", flush=True)
    for key in sorted(summary["paired_delta"]):
        if not key.startswith(("E3", "E4", "E5", "E6")):
            continue
        d = summary["paired_delta"][key]
        print(f"  {key:<22} {d['mean']:+7.3f} ± {d['std']:5.3f}  "
              f"({sum(1 for v in d['deltas_pp'] if v > 0)}/{d['n']} lần dương)  "
              f"các Δ={[round(v, 3) for v in d['deltas_pp']]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
