"""Chạy khối thực nghiệm chính: 3 cấu hình x 3 mức nhãn x 3 lần lặp (mục 4.2).

Ví dụ::

    .\\.venv\\Scripts\\python.exe scripts\\run_experiments.py
    .\\.venv\\Scripts\\python.exe scripts\\run_experiments.py --epochs 20      # chạy thăm dò
    .\\.venv\\Scripts\\python.exe scripts\\run_experiments.py --configs E1 E2 --limit 6
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
    CHECKPOINT_DIR,
    LABEL_FRACTIONS,
    REPETITION_SEEDS,
    RESULTS_DIR,
    RunSpec,
    main_run_specs,
)
from m8.experiments import load_all_runs, run_suite, summarize_runs  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Khối thực nghiệm chính của đề tài M8")
    p.add_argument("--epochs", type=int, default=None, help="ghi đè số epoch (thăm dò)")
    p.add_argument("--configs", nargs="*", default=None, help="tập cấu hình, ví dụ E1 E2-L E2")
    p.add_argument("--fractions", nargs="*", type=float, default=None, help="mức nhãn, ví dụ 0.1 0.2")
    p.add_argument("--seeds", nargs="*", type=int, default=None, help="lần lặp, ví dụ 42 43 44")
    p.add_argument("--limit", type=int, default=None, help="giới hạn số lượt chạy")
    p.add_argument("--no-amp", action="store_true", help="tắt mixed precision")
    p.add_argument("--no-resume", action="store_true", help="chạy lại cả lượt đã có")
    p.add_argument("--tag-suffix", type=str, default="", help="hậu tố cho thư mục kết quả")
    return p.parse_args()


def build_specs(args: argparse.Namespace) -> list[RunSpec]:
    configs = args.configs or ["E1", "E2-L", "E2"]
    fractions = args.fractions or list(LABEL_FRACTIONS)
    seeds = args.seeds or list(REPETITION_SEEDS)
    epochs = args.epochs if args.epochs else None
    specs: list[RunSpec] = []
    for c in configs:
        for f in fractions:
            for s in seeds:
                spec = RunSpec(config=c, label_fraction=float(f), seed=int(s))
                if epochs:
                    spec.epochs = epochs
                if args.tag_suffix:
                    spec.tag = f"{spec.tag}_{args.tag_suffix}"
                specs.append(spec)
    return specs


def main() -> int:
    args = parse_args()
    out_dir = CHECKPOINT_DIR if not args.tag_suffix else CHECKPOINT_DIR.parent / f"checkpoints_{args.tag_suffix}"
    specs = build_specs(args)
    print(f"Sẽ chạy {len(specs)} lượt; thư mục kết quả: {out_dir}", flush=True)
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
    elapsed = time.perf_counter() - t0
    print(f"\nTổng thời gian: {elapsed / 60:.1f} phút", flush=True)

    runs = load_all_runs(out_dir)
    summary = summarize_runs(runs)
    summary_path = RESULTS_DIR / (
        "summary.json" if not args.tag_suffix else f"summary_{args.tag_suffix}.json"
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu tổng hợp: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
