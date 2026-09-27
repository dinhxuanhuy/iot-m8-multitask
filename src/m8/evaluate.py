"""Chỉ số đánh giá cho nhiệm vụ siêu lớp Animal/Vehicle (mục 4.3).

Quy ước:
  * mẫu số bằng 0 -> giá trị precision/recall/F1 bằng 0 và dùng thống nhất;
  * chỉ số được tính trên **toàn bộ** dự đoán của tập đánh giá, không lấy trung
    bình không trọng số theo minibatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import torch

from .config import SUPERCLASS_NAMES

__all__ = [
    "ConfusionMatrix",
    "ClassificationReport",
    "evaluate_predictions",
    "macro_f1",
]


@dataclass
class ClassificationReport:
    """Kết quả đánh giá trên một tập."""

    accuracy: float
    macro_f1: float
    precision: List[float]
    recall: List[float]
    f1: List[float]
    support: List[int]
    confusion: List[List[int]]
    loss: float = float("nan")
    rotation_accuracy: float = float("nan")
    n: int = 0
    class_names: Sequence[str] = field(default_factory=lambda: SUPERCLASS_NAMES)

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "precision": {n: float(p) for n, p in zip(self.class_names, self.precision)},
            "recall": {n: float(r) for n, r in zip(self.class_names, self.recall)},
            "f1": {n: float(f) for n, f in zip(self.class_names, self.f1)},
            "support": {n: int(s) for n, s in zip(self.class_names, self.support)},
            "confusion": self.confusion,
            "loss": self.loss,
            "rotation_accuracy": self.rotation_accuracy,
            "n": int(self.n),
        }

    def pretty(self) -> str:
        lines = [
            f"n = {self.n}  Accuracy = {self.accuracy:.4f}  Macro-F1 = {self.macro_f1:.4f}",
            f"{'lớp':<10}{'P':>9}{'R':>9}{'F1':>9}{'support':>9}",
        ]
        for name, p, r, f, s in zip(self.class_names, self.precision, self.recall,
                                    self.f1, self.support):
            lines.append(f"{name:<10}{p:>9.4f}{r:>9.4f}{f:>9.4f}{s:>9d}")
        if not np.isnan(self.rotation_accuracy):
            lines.append(f"Rotation Accuracy = {self.rotation_accuracy:.4f}")
        return "\n".join(lines)


class ConfusionMatrix:
    """Ma trận nhầm lẫn 2 × 2 với quy ước mẫu số 0 -> 0."""

    def __init__(self, n_classes: int = 2) -> None:
        self.n = n_classes
        self.matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        for t, p in zip(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()):
            self.matrix[int(t), int(p)] += 1

    @property
    def tp(self) -> np.ndarray:
        return np.diag(self.matrix).astype(np.float64)

    @property
    def fp(self) -> np.ndarray:
        return self.matrix.sum(axis=0).astype(np.float64) - self.tp

    @property
    def fn(self) -> np.ndarray:
        return self.matrix.sum(axis=1).astype(np.float64) - self.tp

    def precision(self) -> np.ndarray:
        denom = self.tp + self.fp
        return np.divide(self.tp, denom, out=np.zeros_like(self.tp), where=denom > 0)

    def recall(self) -> np.ndarray:
        denom = self.tp + self.fn
        return np.divide(self.tp, denom, out=np.zeros_like(self.tp), where=denom > 0)

    def f1(self) -> np.ndarray:
        p, r = self.precision(), self.recall()
        denom = p + r
        return np.divide(2 * p * r, denom, out=np.zeros_like(p), where=denom > 0)

    def accuracy(self) -> float:
        total = self.matrix.sum()
        return float(np.trace(self.matrix) / total) if total else 0.0

    def macro_f1(self) -> float:
        return float(np.mean(self.f1()))

    def as_list(self) -> List[List[int]]:
        return self.matrix.tolist()


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 2) -> float:
    cm = ConfusionMatrix(n_classes)
    cm.update(y_true, y_pred)
    return cm.macro_f1()


def evaluate_predictions(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    loss: float = float("nan"),
    rotation_true: Sequence[int] | None = None,
    rotation_pred: Sequence[int] | None = None,
) -> ClassificationReport:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = ConfusionMatrix(2)
    cm.update(y_true, y_pred)
    rot_acc = float("nan")
    if rotation_true is not None and rotation_pred is not None:
        rt = np.asarray(rotation_true, dtype=np.int64)
        rp = np.asarray(rotation_pred, dtype=np.int64)
        rot_acc = float(np.mean(rt == rp)) if rt.size else float("nan")
    return ClassificationReport(
        accuracy=cm.accuracy(),
        macro_f1=cm.macro_f1(),
        precision=cm.precision().tolist(),
        recall=cm.recall().tolist(),
        f1=cm.f1().tolist(),
        support=cm.matrix.sum(axis=1).tolist(),
        confusion=cm.as_list(),
        loss=float(loss),
        rotation_accuracy=rot_acc,
        n=int(y_true.size),
    )


@torch.no_grad()
def predict(model, loader, device, return_rotation: bool = False) -> Dict[str, np.ndarray | float]:
    """Chạy suy luận trên ``loader`` và trả về dự đoán cùng loss phân loại.

    ``loader`` phải trả về ``(x, y_super, y_original)``.
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss, n = 0.0, 0
    ys: List[int] = []
    ps: List[int] = []
    rot_preds: List[int] = []
    for batch in loader:
        x, y, _ = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += float(criterion(logits, y).item())
        n += int(y.numel())
        ys.extend(y.detach().cpu().tolist())
        ps.extend(logits.argmax(dim=1).detach().cpu().tolist())
    out: Dict[str, np.ndarray | float] = {
        "y_true": np.asarray(ys, dtype=np.int64),
        "y_pred": np.asarray(ps, dtype=np.int64),
        "loss": total_loss / max(n, 1),
    }
    if return_rotation:
        out["rotation_pred"] = np.asarray(rot_preds, dtype=np.int64)
    return out


@torch.no_grad()
def rotation_accuracy_full(model, loader, device, num_rotations: int = 4) -> float:
    """Đánh giá đủ bốn phép xoay cho mỗi ảnh (giao thức chẩn đoán ở mục 4.3).

    ``loader`` trả về ``(x, y_super, y_original)`` không xoay; ta tự áp bốn góc.
    """
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        x, _, _ = batch
        for k in range(num_rotations):
            xk = torch.rot90(x, k=k, dims=(2, 3)).to(device, non_blocking=True)
            logits = model.forward_rotation(xk)
            pred = logits.argmax(dim=1)
            correct += int((pred == k).sum().item())
            total += int(pred.numel())
    return correct / total if total else float("nan")
