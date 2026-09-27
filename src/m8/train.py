"""Vòng huấn luyện cho E1 / E2-L / E2 (mục 3.2, 4.1, 4.2).

Nguyên tắc bám theo đề cương:
  * một epoch = một lượt duyệt D_L;
  * ``L = L_sup + lambda_rot * L_rot`` với hai mất mát được lấy trung bình riêng;
  * E1 **không chạy** nhánh rotation kể cả forward (không truyền ảnh xoay qua BN);
  * B_R lấy từ nguồn ảnh quy định ở bảng 6 và dùng một bộ sinh số ngẫu nhiên riêng
    để việc lấy ảnh/góc xoay không làm thay đổi augmentation của nhánh phân loại;
  * chọn checkpoint bằng Macro-F1 validation cao nhất, hòa thì ưu tiên loss phân
    loại validation thấp hơn, sau đó epoch sớm hơn;
  * không dùng Early Stopping trong bảng kết quả chính.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import (
    BETAS,
    EPS,
    LR,
    PROJ_DIM,
    TEMPERATURE_NTXENT,
    TEMPERATURE_SUPCON,
    WEIGHT_DECAY,
    RunSpec,
)
from .contrastive import contrastive_loss
from .evaluate import ClassificationReport, evaluate_predictions, predict
from .models import MultiTaskNet, build_model, load_shared_init, shared_init_state

try:  # torch.amp có API mới ở torch >= 2.4
    from torch.amp import GradScaler, autocast
except Exception:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast  # type: ignore


@dataclass
class EpochLog:
    epoch: int
    train_sup_loss: float
    train_rot_loss: float
    train_contrastive_loss: float
    train_total_loss: float
    val_sup_loss: float
    val_macro_f1: float
    val_accuracy: float
    lr: float
    seconds: float


@dataclass
class RunResult:
    """Toàn bộ kết quả của một lượt huấn luyện."""

    spec: dict
    best_epoch: int
    best_val: dict
    test: dict
    history: List[dict]
    timing: dict
    params: dict
    # checkpoint không nằm trong dict này để tránh phình file JSON
    checkpoint_path: str = ""

    def as_dict(self) -> dict:
        return {
            "spec": self.spec,
            "best_epoch": self.best_epoch,
            "best_val": self.best_val,
            "test": self.test,
            "history": self.history,
            "timing": self.timing,
            "params": self.params,
            "checkpoint_path": self.checkpoint_path,
        }


def _cycle(loader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


def _fast_forward(iterator: Iterator, steps: int) -> None:
    for _ in range(steps):
        next(iterator)


def make_optimizer(model: nn.Module, lr: float = LR) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
    )


def make_scheduler(optimizer: torch.optim.Optimizer, epochs: int) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)


@torch.no_grad()
def validate(model: MultiTaskNet, loader, device, criterion: nn.Module) -> ClassificationReport:
    model.eval()
    total_loss, n = 0.0, 0
    ys: List[int] = []
    ps: List[int] = []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += float(criterion(logits, y).sum().item())
        n += int(y.numel())
        ys.extend(y.cpu().tolist())
        ps.extend(logits.argmax(dim=1).cpu().tolist())
    return evaluate_predictions(ys, ps, loss=total_loss / max(n, 1))


def train_one_run(
    spec: RunSpec,
    loaders: Dict[str, object],
    device: torch.device,
    lambda_rot: Optional[float] = None,
    lambda_c: Optional[float] = None,
    init_state: Optional[Dict[str, torch.Tensor]] = None,
    amp: bool = True,
    log_every: int = 10,
    verbose: bool = True,
    progress_cb=None,
) -> Tuple[RunResult, MultiTaskNet]:
    """Huấn luyện một lượt và trả về (RunResult, mô hình tại checkpoint tốt nhất)."""
    lam = spec.lambda_rot if lambda_rot is None else lambda_rot
    lam_c = spec.lambda_c if lambda_c is None else lambda_c
    use_rotation = spec.rotation_source != "none"
    con_kind = spec.contrastive_kind
    use_contrastive = con_kind is not None
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(spec.seed)

    model = build_model(
        with_rotation_head=use_rotation,
        proj_dim=(PROJ_DIM if use_contrastive else None),
        seed=spec.seed,
        device=device,
    )
    if init_state is not None:
        load_shared_init(model, init_state)

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer, spec.epochs)
    use_amp = bool(amp and device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp) if use_amp else None

    labeled_loader = loaders["labeled_loader"]
    rotation_loader = loaders.get("rotation_loader") if use_rotation else None
    rot_iter = _cycle(rotation_loader) if rotation_loader is not None else None
    rot_rng = torch.Generator().manual_seed(spec.seed + 7777)
    # Nhánh contrastive dùng hai nguồn minibatch độc lập với nhánh xoay và với nhánh
    # phân loại, để việc thêm nhánh không dịch chuyển chuỗi augmentation của nhánh khác.
    con_main = loaders.get("contrastive_loader_main") if use_contrastive else None
    con_aux = loaders.get("contrastive_loader_aux") if use_contrastive else None
    con_main_iter = _cycle(con_main) if con_main is not None else None
    con_aux_iter = _cycle(con_aux) if con_aux is not None else None
    steps_per_epoch = len(labeled_loader)

    history: List[EpochLog] = []
    best = {"macro_f1": -math.inf, "loss": math.inf, "epoch": -1}
    best_state: Dict[str, torch.Tensor] | None = None
    best_val_report: ClassificationReport | None = None
    t_start = time.perf_counter()
    images_sup = 0
    images_rot = 0
    images_con = 0

    for epoch in range(1, spec.epochs + 1):
        model.train()
        sup_sum, rot_sum, con_sum, total_sum, nb = 0.0, 0.0, 0.0, 0.0, 0
        epoch_start = time.perf_counter()

        # Khởi tạo bộ đếm B_R sao cho số minibatch rotation trùng với số bước
        # được cập nhật thật; mỗi epoch dùng lại cùng chuỗi nhưng RNG riêng đảm
        # bảo góc xoay thay đổi và không ảnh hưởng augmentation của nhánh chính.
        rot_used = 0
        con_used = 0

        for step, (x_sup, y_sup, _) in enumerate(labeled_loader):
            x_sup = x_sup.to(device, non_blocking=True)
            y_sup = y_sup.to(device, non_blocking=True)
            # Minibatch cuối của D_L có thể nhỏ hơn BATCH_SIZE (ta giữ lại, không bỏ mẫu).
            # Khi đó các nhánh phụ phải được cắt về đúng b_sup để hai vế của loss khớp
            # kích thước; nếu không sẽ lỗi "batch_size mismatch" ở minibatch cuối.
            b_sup = int(y_sup.shape[0])

            optimizer.zero_grad(set_to_none=True)
            loss_rot = None
            loss_con = None
            out_rot = None
            z = None
            v1 = v2 = y_con = None

            if use_rotation:
                x_rot, y_rot, _ = next(rot_iter)
                # RNG riêng: hoán vị góc xoay đã được gán trong dataset, ở đây
                # chỉ quyết định thứ tự minibatch để không dùng chung luồng RNG
                # với augmentation của nhánh phân loại.
                perm = torch.randperm(x_rot.shape[0], generator=rot_rng)
                x_rot = x_rot[perm][:b_sup].to(device, non_blocking=True)
                y_rot = y_rot[perm][:b_sup].to(device, non_blocking=True)

            if use_contrastive:
                batch_con = next(con_main_iter if not use_rotation else con_aux_iter)
                v1, v2, y_con, _ = batch_con
                v1 = v1[:b_sup].to(device, non_blocking=True)
                v2 = v2[:b_sup].to(device, non_blocking=True)
                if con_kind == "supcon":
                    y_con = y_con[:b_sup].to(device, non_blocking=True)

            with autocast("cuda") if use_amp else contextlib.nullcontext():
                if use_rotation and use_contrastive:
                    # ba nhánh: ghép B_L với B_R, và hai view của nhánh contrastive
                    joined_rot = torch.cat([x_sup, x_rot], dim=0)
                    joined_con = torch.cat([v1, v2], dim=0)
                    feats = model.encoder(torch.cat([joined_rot, joined_con], dim=0))
                    f_rot = feats[b_sup: 2 * b_sup]
                    f_con = feats[2 * b_sup:]
                    out_sup = model.super_head(feats[:b_sup])
                    out_rot = model.rotation_head(f_rot)
                    z = model.projection_head(f_con)
                elif use_rotation:
                    out_sup, out_rot = model.forward_joined(x_sup, x_rot)
                elif use_contrastive:
                    joined_con = torch.cat([v1, v2], dim=0)
                    feats = model.encoder(torch.cat([x_sup, joined_con], dim=0))
                    out_sup = model.super_head(feats[:b_sup])
                    z = model.projection_head(feats[b_sup:])
                else:
                    out_sup = model(x_sup)

                loss_sup = criterion(out_sup, y_sup)
                loss = loss_sup
                if out_rot is not None:
                    loss_rot = criterion(out_rot, y_rot)
                    loss = loss + lam * loss_rot
                if use_contrastive:
                    labels_con = torch.cat([y_con, y_con], dim=0) if con_kind == "supcon" else None
                    loss_con = contrastive_loss(
                        z, labels_con, con_kind,
                        temperature_supcon=TEMPERATURE_SUPCON,
                        temperature_ntxent=TEMPERATURE_NTXENT,
                    )
                    loss = loss + lam_c * loss_con

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            sup_sum += float(loss_sup.detach())
            if loss_rot is not None:
                rot_sum += float(loss_rot.detach())
                images_rot += int(y_rot.numel())
                rot_used += 1
            if loss_con is not None:
                con_sum += float(loss_con.detach())
                images_con += int(v1.shape[0]) * 2
                con_used += 1
            total_sum += float(loss.detach())
            images_sup += int(y_sup.numel())
            nb += 1

        scheduler.step()
        val = validate(model, loaders["val_loader"], device, criterion)
        epoch_seconds = time.perf_counter() - epoch_start
        log = EpochLog(
            epoch=epoch,
            train_sup_loss=sup_sum / max(nb, 1),
            train_rot_loss=(rot_sum / max(rot_used, 1)) if use_rotation else float("nan"),
            train_contrastive_loss=(con_sum / max(con_used, 1)) if use_contrastive else float("nan"),
            train_total_loss=total_sum / max(nb, 1),
            val_sup_loss=val.loss,
            val_macro_f1=val.macro_f1,
            val_accuracy=val.accuracy,
            lr=float(optimizer.param_groups[0]["lr"]),
            seconds=epoch_seconds,
        )
        history.append(log)
        if progress_cb is not None:
            progress_cb(epoch, log)

        better = (
            log.val_macro_f1 > best["macro_f1"] + 1e-12
            or (
                abs(log.val_macro_f1 - best["macro_f1"]) <= 1e-12
                and (
                    log.val_sup_loss < best["loss"] - 1e-12
                    or (abs(log.val_sup_loss - best["loss"]) <= 1e-12 and epoch < best["epoch"])
                )
            )
        )
        if better:
            best = {"macro_f1": log.val_macro_f1, "loss": log.val_sup_loss, "epoch": epoch}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_report = val

        if verbose and (epoch % log_every == 0 or epoch == 1 or epoch == spec.epochs):
            extra = ""
            if use_rotation:
                extra += f"L_rot={log.train_rot_loss:.4f} "
            if use_contrastive:
                extra += f"L_con={log.train_contrastive_loss:.4f} "
            print(
                f"  [{spec.tag}] epoch {epoch:3d}/{spec.epochs} "
                f"L_sup={log.train_sup_loss:.4f} "
                f"{extra}"
                f"valF1={log.val_macro_f1:.4f} "
                f"valAcc={log.val_accuracy:.4f} "
                f"({epoch_seconds:.1f}s)",
                flush=True,
            )

    train_seconds = time.perf_counter() - t_start
    assert best_state is not None and best_val_report is not None
    model.load_state_dict(best_state)
    best_val_report = validate(model, loaders["val_loader"], device, criterion)

    # Đánh giá test bằng checkpoint tốt nhất theo validation
    test_pred = predict(model, loaders["test_loader"], device)
    test_report = evaluate_predictions(
        test_pred["y_true"], test_pred["y_pred"], loss=float(test_pred["loss"])
    )

    params = {
        "inference": int(model.inference_parameters()),
        "training": int(model.training_parameters()),
    }
    if use_rotation:
        params["rotation_head"] = int(sum(p.numel() for p in model.rotation_head.parameters()))
    else:
        params["rotation_head"] = 0
    if use_contrastive:
        params["projection_head"] = int(model.projection_parameters())
        params["training_with_aux_heads"] = int(
            model.training_parameters() + model.projection_parameters()
        )
    else:
        params["projection_head"] = 0
        params["training_with_aux_heads"] = params["training"]

    result = RunResult(
        spec=spec.as_dict(),
        best_epoch=int(best["epoch"]),
        best_val=best_val_report.as_dict(),
        test=test_report.as_dict(),
        history=[asdict(h) for h in history],
        timing={
            "train_seconds": train_seconds,
            "seconds_per_epoch": train_seconds / spec.epochs,
            "update_steps": int(steps_per_epoch * spec.epochs),
            "sup_images_seen": int(images_sup),
            "rot_images_seen": int(images_rot),
            "contrastive_images_seen": int(images_con),
            "device": str(device),
        },
        params=params,
    )
    return result, model


def save_run(result: RunResult, model: MultiTaskNet, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = result.spec["tag"]
    ckpt = out_dir / f"{tag}.pt"
    torch.save({"state_dict": model.state_dict(), "spec": result.spec}, ckpt)
    result.checkpoint_path = str(ckpt)
    (out_dir / f"{tag}.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ckpt
