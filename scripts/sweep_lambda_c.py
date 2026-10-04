"""Khảo sát λ_c cho nhánh contrastive — CHỈ dùng validation, không xem test.

Mục 4.2 của đề cương yêu cầu: "Mọi siêu tham số mở rộng phải được khóa bằng validation
trước khi đánh giá test" và "không lựa chọn λ hoặc cấu hình dựa trên kết quả test".
Script này vì vậy **bỏ hoàn toàn bước đánh giá test**: nó chỉ chạy huấn luyện, ghi
Macro-F1 validation tốt nhất, rồi chọn λ_c theo tiêu chí định trước (validation cao nhất).

Kết quả ghi vào ``results/lambda_c_sweep.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m8.config import (  # noqa: E402
    CHECKPOINT_DIR,
    RESULTS_DIR,
    RunSpec,
    TEMPERATURE_NTXENT,
    TEMPERATURE_SUPCON,
)
from m8.data import audit_benchmark, build_benchmark, get_cifar10, make_run_loaders  # noqa: E402
from m8.evaluate import ClassificationReport, evaluate_predictions, predict  # noqa: E402
from m8.experiments import _init_state_for  # noqa: E402
from m8.train import make_optimizer, make_scheduler, validate  # noqa: E402
from m8.models import build_model  # noqa: E402
from m8.contrastive import contrastive_loss  # noqa: E402

#: Lưới λ_c khóa trước khi chạy (không điều chỉnh sau khi thấy kết quả).
#: PHẢI khớp với `grid` trong results/lambda_c_sweep.json — lưới đã dùng thật là
#: [0.05, 0.1, 0.25, 0.5, 1.0]. Trước đây hằng số này ghi (0.0, 0.05, 0.1, 0.25, 0.5),
#: tức KHÔNG khớp lưới đã chạy, nên gây khó tái lập.
LAMBDA_C_GRID: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0)


@dataclass
class SweepRecord:
    config: str
    kind: str
    source: str
    label_pct: int
    seed: int
    lambda_c: float
    best_val_macro_f1: float
    best_val_loss: float
    best_epoch: int
    seconds: float


def train_val_only(
    spec: RunSpec,
    loaders: Dict[str, object],
    device: torch.device,
    lambda_c: float,
    amp: bool = True,
    verbose: bool = True,
) -> SweepRecord:
    """Huấn luyện và **chỉ** đánh giá trên validation (không đụng tới test)."""
    import contextlib
    import math

    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.seed)

    use_rotation = spec.rotation_source != "none"
    con_kind = spec.contrastive_kind
    from m8.config import PROJ_DIM

    model = build_model(with_rotation_head=use_rotation,
                        proj_dim=(PROJ_DIM if con_kind else None),
                        seed=spec.seed, device=device)
    from m8.models import load_shared_init

    load_shared_init(model, _init_state_for(spec))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer, spec.epochs)
    use_amp = bool(amp and device.type == "cuda")
    from torch.amp import GradScaler, autocast

    scaler = GradScaler("cuda", enabled=use_amp) if use_amp else None

    labeled_loader = loaders["labeled_loader"]
    rot_iter = None
    if use_rotation:
        def _cycle(loader):
            while True:
                yield from loader

        rot_iter = _cycle(loaders["rotation_loader"])
        rot_rng = torch.Generator().manual_seed(spec.seed + 7777)
    con_iter = None
    if con_kind:
        def _cycle2(loader):
            while True:
                yield from loader

        con_iter = _cycle2(loaders["contrastive_loader_main"])

    best = {"f1": -math.inf, "loss": math.inf, "epoch": -1}
    t0 = time.perf_counter()
    for epoch in range(1, spec.epochs + 1):
        model.train()
        for x_sup, y_sup, _ in labeled_loader:
            x_sup = x_sup.to(device, non_blocking=True)
            y_sup = y_sup.to(device, non_blocking=True)
            b = int(y_sup.shape[0])
            optimizer.zero_grad(set_to_none=True)
            x_rot = y_rot = None
            v1 = v2 = y_con = None
            if use_rotation:
                x_rot, y_rot, _ = next(rot_iter)
                perm = torch.randperm(x_rot.shape[0], generator=rot_rng)
                x_rot = x_rot[perm][:b].to(device, non_blocking=True)
                y_rot = y_rot[perm][:b].to(device, non_blocking=True)
            if con_kind:
                v1, v2, y_con, _ = next(con_iter)
                v1 = v1[:b].to(device, non_blocking=True)
                v2 = v2[:b].to(device, non_blocking=True)
                if con_kind == "supcon":
                    y_con = y_con[:b].to(device, non_blocking=True)

            with autocast("cuda") if use_amp else contextlib.nullcontext():
                if use_rotation and con_kind:
                    joined = torch.cat([torch.cat([x_sup, x_rot], 0), torch.cat([v1, v2], 0)], 0)
                    feats = model.encoder(joined)
                    out_sup = model.super_head(feats[:b])
                    out_rot = model.rotation_head(feats[b:2 * b])
                    z = model.projection_head(feats[2 * b:])
                elif use_rotation:
                    out_sup, out_rot = model.forward_joined(x_sup, x_rot)
                elif con_kind:
                    feats = model.encoder(torch.cat([x_sup, torch.cat([v1, v2], 0)], 0))
                    out_sup = model.super_head(feats[:b])
                    z = model.projection_head(feats[b:])
                else:
                    out_sup = model(x_sup)
                loss = criterion(out_sup, y_sup)
                if use_rotation:
                    loss = loss + spec.lambda_rot * criterion(out_rot, y_rot)
                if con_kind:
                    labels_con = torch.cat([y_con, y_con], 0) if con_kind == "supcon" else None
                    loss = loss + lambda_c * contrastive_loss(
                        z, labels_con, con_kind,
                        temperature_supcon=TEMPERATURE_SUPCON,
                        temperature_ntxent=TEMPERATURE_NTXENT)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        scheduler.step()

        val = validate(model, loaders["val_loader"], device, criterion)
        if (val.macro_f1 > best["f1"] + 1e-12
                or (abs(val.macro_f1 - best["f1"]) <= 1e-12 and val.loss < best["loss"])):
            best = {"f1": val.macro_f1, "loss": val.loss, "epoch": epoch}
        if verbose and (epoch % 10 == 0 or epoch == spec.epochs):
            print(f"      λ_c={lambda_c:<5} epoch {epoch:3d}/{spec.epochs} "
                  f"valF1={val.macro_f1:.4f} (tốt nhất {best['f1']:.4f})", flush=True)

    return SweepRecord(
        config=spec.config,
        kind=con_kind or "none",
        source=spec.contrastive_source or "-",
        label_pct=spec.label_pct,
        seed=spec.seed,
        lambda_c=lambda_c,
        best_val_macro_f1=best["f1"],
        best_val_loss=best["loss"],
        best_epoch=best["epoch"],
        seconds=time.perf_counter() - t0,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Khảo sát λ_c chỉ trên validation")
    ap.add_argument("--epochs", type=int, default=40, help="số epoch cho khảo sát")
    ap.add_argument("--fraction", type=float, default=0.10, help="mức nhãn ưu tiên")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", nargs="*", default=["E4", "E6"],
                    help="E3/E4 = chỉ contrastive; E5/E6 = contrastive + xoay")
    ap.add_argument("--lambdas", nargs="*", type=float, default=None)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    lambdas = args.lambdas or list(LAMBDA_C_GRID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_base, test_base = get_cifar10(download=False)
    bench = build_benchmark(train_base.targets)

    records: List[SweepRecord] = []
    for cfg in args.configs:
        spec0 = RunSpec(config=cfg, label_fraction=args.fraction, seed=args.seed,
                        epochs=args.epochs)
        loaders = make_run_loaders(
            train_base, test_base, bench, spec0.seed, spec0.label_fraction,
            spec0.rotation_source,
            contrastive_kind=spec0.contrastive_kind,
            contrastive_source=spec0.contrastive_source,
        )
        print(f"\n### {cfg} ({spec0.contrastive_kind}@{spec0.contrastive_source}), "
              f"mức {spec0.label_pct}% nhãn, seed {spec0.seed}, {args.epochs} epoch", flush=True)
        for lam in lambdas:
            spec = RunSpec(config=cfg, label_fraction=args.fraction, seed=args.seed,
                           epochs=args.epochs, lambda_c=lam)
            print(f"    -> λ_c = {lam}", flush=True)
            rec = train_val_only(spec, loaders, device, lam, amp=not args.no_amp)
            records.append(rec)
            print(f"       val Macro-F1 tốt nhất = {rec.best_val_macro_f1:.4f} "
                  f"(epoch {rec.best_epoch}, {rec.seconds / 60:.1f} phút)", flush=True)

    # Chọn MỘT λ_c dùng chung cho mọi cấu hình mở rộng (đúng như bảng 5 của đề cương).
    #
    # Quy tắc khóa, khai báo trước khi chạy: chọn λ_c sao cho
    #   (1) nằm trong khoảng hiệu quả của CẢ HAI cấu hình khảo sát, và
    #   (2) tối đa hóa trung bình Macro-F1 validation giữa chúng.
    # Lý do phải dùng quy tắc "cả hai": mỗi cấu hình có thể có cực trị riêng (E4 thích
    # λ_c lớn, E6 thích λ_c nhỏ); nếu chọn theo một cấu hình duy nhất thì giá trị khóa
    # có thể rất kém cho cấu hình khác. Trung bình qua hai cấu hình là tiêu chí ổn định
    # và không phụ thuộc vào kết quả test.
    shared_lambda, shared_score = None, -1.0
    per_config_rows = {}
    for cfg in args.configs:
        sel = [r for r in records if r.config == cfg]
        per_config_rows[cfg] = {r.lambda_c: r.best_val_macro_f1 for r in sel}
    for lam in lambdas:
        vals = [per_config_rows[cfg].get(lam) for cfg in args.configs]
        vals = [v for v in vals if v is not None]
        if len(vals) != len(args.configs):
            continue
        mean = sum(vals) / len(vals)
        if mean > shared_score:
            shared_lambda, shared_score = lam, mean

    chosen: Dict[str, dict] = {}
    for cfg in args.configs:
        chosen[cfg] = {
            "lambda_c": shared_lambda,
            "val_macro_f1": per_config_rows[cfg].get(shared_lambda),
            "own_best_lambda_c": max(per_config_rows[cfg], key=per_config_rows[cfg].get),
            "own_best_val": max(per_config_rows[cfg].values()),
            "epochs": args.epochs,
            "fraction": args.fraction,
            "seed": args.seed,
        }

    out = RESULTS_DIR / "lambda_c_sweep.json"
    out.write_text(json.dumps({
        "note": "Khảo sát chỉ dùng Macro-F1 validation; test chưa được đánh giá. "
                "lambda_c được khóa dùng chung cho mọi cấu hình theo trung bình validation "
                "cao nhất qua các cấu hình khảo sát.",
        "grid": lambdas,
        "epochs": args.epochs,
        "fraction": args.fraction,
        "seed": args.seed,
        "shared_lambda_c": shared_lambda,
        "shared_mean_val": shared_score,
        "records": [r.__dict__ for r in records],
        "chosen": chosen,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã lưu {out}")
    print(f"λ_c được khóa dùng chung: {shared_lambda} "
          f"(trung bình validation {shared_score:.4f})")
    for cfg in args.configs:
        info = chosen[cfg]
        print(f"  {cfg}: val(λ_c={shared_lambda}) = {info['val_macro_f1']:.4f} | "
              f"cực trị riêng λ_c={info['own_best_lambda_c']} -> {info['own_best_val']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
