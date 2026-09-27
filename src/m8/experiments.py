"""Điều phối khối thực nghiệm chính (mục 4.2)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .config import (
    CHECKPOINT_DIR,
    EXTENSION_CONFIGS,
    LABEL_FRACTIONS,
    MAIN_CONFIGS,
    REPETITION_SEEDS,
    RESULTS_DIR,
    RunSpec,
    main_run_specs,
)
from .data import Benchmark, get_cifar10, make_run_loaders
from .models import build_model, shared_init_state
from .train import RunResult, save_run, train_one_run

__all__ = ["run_suite", "summarize_runs", "load_all_runs"]


def _init_state_for(spec: RunSpec) -> Dict[str, torch.Tensor]:
    """Trạng thái khởi tạo dùng chung cho mọi cấu hình ghép cặp.

    Bảng 7 yêu cầu mỗi cặp so sánh dùng cùng trọng số khởi tạo của encoder và
    Superclass Head. Ta luôn khởi tạo bản có đủ head phụ rồi chỉ nạp lại phần
    trùng tên, nên trọng số dùng chung là hoàn toàn giống nhau giữa các cấu hình.

    Các cấu hình mở rộng (E3--E6) dùng cùng seed nên cũng nhận đúng trạng thái khởi
    tạo này; nhờ vậy so sánh E4 với E1 là so sánh **cùng điểm xuất phát**.
    """
    model = build_model(with_rotation_head=True, seed=spec.seed, device="cpu")
    return shared_init_state(model)


def run_suite(
    specs: Optional[Sequence[RunSpec]] = None,
    out_dir: Path = CHECKPOINT_DIR,
    results_dir: Path = RESULTS_DIR,
    device: Optional[torch.device] = None,
    amp: bool = True,
    epochs: Optional[int] = None,
    verbose: bool = True,
    progress_cb: Optional[Callable[[RunSpec, int, RunResult], None]] = None,
    resume: bool = True,
    limit: Optional[int] = None,
    use_init_state: bool = True,
) -> List[RunResult]:
    """Chạy (hoặc tiếp tục) một khối thực nghiệm (chính hoặc mở rộng).

    Kết quả mỗi lượt được lưu ngay vào ``out_dir/<tag>.json`` nên có thể dừng và
    chạy lại mà không mất kết quả đã có (``resume=True`` bỏ qua lượt đã xong).
    """
    specs = list(specs or main_run_specs())
    if epochs is not None:
        for s in specs:
            s.epochs = epochs
    if limit is not None:
        specs = specs[:limit]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_base, test_base = get_cifar10()
    from .data import build_benchmark, audit_benchmark  # local import tránh vòng

    bench = build_benchmark(train_base.targets)
    results: List[RunResult] = []

    for i, spec in enumerate(specs, start=1):
        run_json = out_dir / f"{spec.tag}.json"
        if resume and run_json.exists():
            data = json.loads(run_json.read_text(encoding="utf-8"))
            res = RunResult(
                spec=data["spec"],
                best_epoch=data["best_epoch"],
                best_val=data["best_val"],
                test=data["test"],
                history=data["history"],
                timing=data["timing"],
                params=data["params"],
                checkpoint_path=data.get("checkpoint_path", ""),
            )
            results.append(res)
            if verbose:
                print(f"[{i}/{len(specs)}] bỏ qua (đã có) {spec.tag}", flush=True)
            continue

        if verbose:
            print(
                f"[{i}/{len(specs)}] {spec.tag} | config={spec.config} "
                f"label={spec.label_pct}% seed={spec.seed} "
                f"rotation_source={spec.rotation_source} "
                f"contrastive={spec.contrastive_kind}"
                f"{'@' + str(spec.contrastive_source) if spec.contrastive_source else ''} "
                f"epochs={spec.epochs}",
                flush=True,
            )
        loaders = make_run_loaders(
            train_base, test_base, bench, spec.seed, spec.label_fraction,
            spec.rotation_source,
            contrastive_kind=spec.contrastive_kind,
            contrastive_source=spec.contrastive_source,
        )
        init_state = _init_state_for(spec) if use_init_state else None
        t0 = time.perf_counter()
        res, model = train_one_run(
            spec, loaders, device, lambda_rot=spec.lambda_rot,
            lambda_c=spec.lambda_c, init_state=init_state, amp=amp, verbose=verbose,
        )
        res.timing["wall_seconds_including_eval"] = time.perf_counter() - t0
        save_run(res, model, out_dir)
        results.append(res)
        if verbose:
            print(
                f"    -> val Macro-F1 {res.best_val['macro_f1']:.4f} "
                f"@epoch {res.best_epoch} | test Macro-F1 {res.test['macro_f1']:.4f} "
                f"| test Acc {res.test['accuracy']:.4f}",
                flush=True,
            )
        if progress_cb is not None:
            progress_cb(spec, i, res)

    return results


def load_all_runs(out_dir: Path = CHECKPOINT_DIR) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for p in sorted(Path(out_dir).glob("*.json")):
        out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return out


def load_all_runs_multi(*dirs: Path) -> Dict[str, dict]:
    """Nạp kết quả từ nhiều thư mục (khối chính + khối mở rộng contrastive)."""
    out: Dict[str, dict] = {}
    for d in dirs:
        out.update(load_all_runs(d))
    return out


def _agg(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def summarize_runs(runs: Dict[str, dict]) -> Dict[str, object]:
    """Tổng hợp trung bình ± độ lệch chuẩn mẫu và chênh lệch ghép cặp Δ_s."""
    rows: List[dict] = []
    for tag, r in runs.items():
        spec = r["spec"]
        rows.append(
            {
                "tag": tag,
                "config": spec["config"],
                "label_pct": spec["label_pct"],
                "seed": spec["seed"],
                "epochs": spec["epochs"],
                "best_epoch": r["best_epoch"],
                "val_macro_f1": r["best_val"]["macro_f1"],
                "val_accuracy": r["best_val"]["accuracy"],
                "test_macro_f1": r["test"]["macro_f1"],
                "test_accuracy": r["test"]["accuracy"],
                "test_f1_animal": r["test"]["f1"]["Animal"],
                "test_f1_vehicle": r["test"]["f1"]["Vehicle"],
                "test_precision_animal": r["test"]["precision"]["Animal"],
                "test_recall_animal": r["test"]["recall"]["Animal"],
                "test_precision_vehicle": r["test"]["precision"]["Vehicle"],
                "test_recall_vehicle": r["test"]["recall"]["Vehicle"],
                "test_confusion": r["test"]["confusion"],
                "train_seconds": r["timing"]["train_seconds"],
                "seconds_per_epoch": r["timing"]["seconds_per_epoch"],
                "inference_params": r["params"]["inference"],
                "training_params": r["params"]["training"],
                "projection_params": r["params"].get("projection_head", 0),
                "rot_images_seen": r["timing"].get("rot_images_seen", 0),
                "contrastive_images_seen": r["timing"].get("contrastive_images_seen", 0),
                "update_steps": r["timing"].get("update_steps", 0),
            }
        )

    configs = [c for c in (*MAIN_CONFIGS, *EXTENSION_CONFIGS)
               if any(r["config"] == c for r in rows)]
    summary: Dict[str, object] = {
        "runs": rows, "aggregate": {}, "paired_delta": {}, "per_class": {},
        "configs": configs,
    }
    for cfg in configs:
        for pct in sorted({r["label_pct"] for r in rows}):
            sel = [r for r in rows if r["config"] == cfg and r["label_pct"] == pct]
            if not sel:
                continue
            key = f"{cfg}_{pct}"
            summary["aggregate"][key] = {
                "n_seeds": len(sel),
                "test_macro_f1": _agg([r["test_macro_f1"] for r in sel]),
                "test_accuracy": _agg([r["test_accuracy"] for r in sel]),
                "val_macro_f1": _agg([r["val_macro_f1"] for r in sel]),
                "train_seconds": _agg([r["train_seconds"] for r in sel]),
                "f1_animal": _agg([r["test_f1_animal"] for r in sel]),
                "f1_vehicle": _agg([r["test_f1_vehicle"] for r in sel]),
            }

    # chênh lệch ghép cặp theo lần lặp: Δ_s = Macro-F1_s(M8) − Macro-F1_s(E1)
    # Bao gồm cả các cấu hình mở rộng E3--E6 khi có kết quả.
    for pct in sorted({r["label_pct"] for r in rows}):
        for cfg in [c for c in configs if c != "E1"]:
            deltas = []
            for seed in sorted({r["seed"] for r in rows}):
                a = [r for r in rows if r["config"] == cfg and r["label_pct"] == pct and r["seed"] == seed]
                b = [r for r in rows if r["config"] == "E1" and r["label_pct"] == pct and r["seed"] == seed]
                if a and b:
                    deltas.append((a[0]["test_macro_f1"] - b[0]["test_macro_f1"]) * 100.0)
            if deltas:
                summary["paired_delta"][f"{cfg}_minus_E1_{pct}"] = {
                    "deltas_pp": deltas,
                    **_agg(deltas),
                }
        # so sánh E6 với E4 và E5 với E3 để tách vai trò của nhánh xoay khi có contrastive
        for hi, lo in (("E6", "E4"), ("E5", "E3"), ("E4", "E1")):
            if hi not in configs or lo not in configs:
                continue
            deltas = []
            for seed in sorted({r["seed"] for r in rows}):
                a = [r for r in rows if r["config"] == hi and r["label_pct"] == pct and r["seed"] == seed]
                b = [r for r in rows if r["config"] == lo and r["label_pct"] == pct and r["seed"] == seed]
                if a and b:
                    deltas.append((a[0]["test_macro_f1"] - b[0]["test_macro_f1"]) * 100.0)
            if deltas:
                summary["paired_delta"][f"{hi}_minus_{lo}_{pct}"] = {
                    "deltas_pp": deltas,
                    **_agg(deltas),
                }
    return summary
