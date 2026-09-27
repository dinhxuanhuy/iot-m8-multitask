"""Dữ liệu CIFAR-10 cho đề tài M8.

Bao gồm:
  * tải CIFAR-10 và ánh xạ sang hai siêu lớp Animal / Vehicle;
  * chia D_train / D_val phân tầng theo 10 lớp gốc với seed 2026 (mục 2.2);
  * tạo các tập có nhãn lồng nhau D_L,10% ⊂ D_L,20% ⊂ D_L,50% theo từng lần lặp s;
  * sinh nhãn góc xoay tự động bằng ``torch.rot90`` (mục 2.3);
  * lưu chỉ mục của mọi tập để kiểm tra lại và để chứng minh nhãn ẩn không rò rỉ.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from .config import (
    ANIMAL_CLASSES,
    ARTIFACT_DIR,
    BATCH_SIZE,
    DATA_DIR,
    LABEL_BUDGET,
    LABEL_FRACTIONS,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    NUM_ROTATIONS,
    ORIGINAL_TO_SUPER,
    PIN_MEMORY,
    REPETITION_SEEDS,
    SPLIT_SEED,
    TEST_SUPER_COUNTS,
    TRAIN_SIZE,
    VAL_SIZE,
    VAL_SUPER_COUNTS,
    VEHICLE_CLASSES,
)

# --------------------------------------------------------------------------
# Biến đổi ảnh
# --------------------------------------------------------------------------


class RandomRotation90:
    """Chọn k đều trong {0,1,2,3} rồi xoay 90k độ ngược chiều kim đồng hồ.

    Nhãn phụ là chính giá trị k. Ảnh vào là tensor 4 chiều ``(B, C, H, W)``
    vì transform chạy bên trong ``torchvision.transforms.v2``.
    """

    def __init__(self, num_rotations: int = NUM_ROTATIONS) -> None:
        self.num_rotations = num_rotations

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = img.shape[0]
        k = torch.randint(0, self.num_rotations, (b,), dtype=torch.long)
        out = img.clone()
        for value in range(self.num_rotations):
            if value == 0:
                continue
            mask = k == value
            if mask.any():
                out[mask] = torch.rot90(img[mask], k=value, dims=(2, 3))
        return out, k


def normalize_transform() -> transforms.Compose:
    """Chuyển tensor [0,1] rồi chuẩn hóa với mean = std = 0,5 (hằng số chọn trước)."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def labeled_train_transform() -> transforms.Compose:
    """Nhánh phân loại khi huấn luyện (mục 2.3)."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


def eval_transform() -> transforms.Compose:
    """Validation / test của nhiệm vụ chính: không augmentation ngẫu nhiên."""
    return normalize_transform()


def contrastive_view_transform() -> transforms.Compose:
    """Tăng cường ảnh **mạnh** cho hai view của nhánh contrastive.

    Chuỗi này tách biệt khỏi augmentation của nhánh phân loại (mục 3.3): nhánh
    contrastive cần hai view "khó" của cùng một ảnh, còn nhánh phân loại cần một
    view giữ được thông tin ngữ nghĩa. Dùng `RandomResizedCrop(32, scale=(0.5,1.0))`,
    lật ngang, đổi màu nhẹ và chuyển xám, rồi chuẩn hóa như mọi nhánh khác.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(32, scale=(0.5, 1.0), ratio=(0.75, 1.333)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
        ]
    )


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


@dataclass
class SuperclassDataset(Dataset):
    """Bọc CIFAR-10 thành dataset hai siêu lớp, giữ nhãn gốc để phân tầng."""

    base: CIFAR10
    indices: np.ndarray
    transform: object
    return_superclass: bool = True

    def __post_init__(self) -> None:
        self.indices = np.asarray(self.indices, dtype=np.int64)
        self._targets = np.asarray(self.base.targets, dtype=np.int64)
        self._super = np.array([ORIGINAL_TO_SUPER[int(t)] for t in self._targets], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        img, original = self.base[idx]
        img = self.transform(img)
        original = int(original)
        if self.return_superclass:
            return img, ORIGINAL_TO_SUPER[original], original
        return img, original

    # ---- tiện ích ----
    @property
    def super_labels(self) -> np.ndarray:
        return self._super[self.indices]

    @property
    def original_labels(self) -> np.ndarray:
        return self._targets[self.indices]


class ContrastiveViewDataset(Dataset):
    """Trả về hai view tăng cường mạnh của cùng một ảnh (cho E3--E6).

    ``__getitem__`` trả ``(view1, view2, y_super, original_idx)``. Hai view được sinh
    bằng cùng một chuỗi biến đổi nhưng gọi hai lần, nên chúng khác nhau; đây chính là
    cặp positive của NT-Xent/SupCon.

    Nhãn siêu lớp chỉ được trả về khi ``return_superclass=True``; với nguồn ảnh là
    ``D_train`` (E3/E5) dataset được tạo với ``return_superclass=False`` nên **không
    thể rò rỉ nhãn thật của phần ẩn nhãn** dù mã gọi có sai đi nữa.
    """

    def __init__(
        self,
        base: CIFAR10,
        indices: np.ndarray,
        return_superclass: bool = True,
        transform: object | None = None,
    ) -> None:
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.return_superclass = return_superclass
        self.transform = transform if transform is not None else contrastive_view_transform()
        self._targets = np.asarray(base.targets, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        img, original = self.base[idx]
        view1 = self.transform(img)
        view2 = self.transform(img)
        original = int(original)
        if self.return_superclass:
            return view1, view2, ORIGINAL_TO_SUPER[original], idx
        return view1, view2, -1, idx


class RotationViewDataset(Dataset):
    """Trả về (ảnh xoay, k, chỉ mục gốc trong CIFAR-10).

    Dùng chung quy ước chuẩn hóa với nhánh phân loại nhưng **không** augmentation
    ngẫu nhiên: chỉ áp dụng ``rot90`` với góc đã gán.
    """

    def __init__(self, base: CIFAR10, indices: np.ndarray) -> None:
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.normalize = normalize_transform()

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        img, _ = self.base[idx]
        tensor = self.normalize(img)
        tensor = tensor.unsqueeze(0)                      # (1, C, H, W)
        rotated, k = RandomRotation90()(tensor)
        return rotated.squeeze(0), int(k.item()), idx


# --------------------------------------------------------------------------
# Chia split
# --------------------------------------------------------------------------


@dataclass
class Benchmark:
    """Toàn bộ chỉ mục của phép chia benchmark và các tập con có nhãn."""

    train_idx: np.ndarray          # D_train (45.000)
    val_idx: np.ndarray            # D_val (5.000)
    test_idx: np.ndarray           # D_test (10.000)
    labeled: Dict[Tuple[int, float], np.ndarray]  # (seed, fraction) -> D_L
    unlabeled: Dict[Tuple[int, float], np.ndarray]  # (seed, fraction) -> D_U

    def dl(self, seed: int, fraction: float) -> np.ndarray:
        return self.labeled[(seed, fraction)]

    def du(self, seed: int, fraction: float) -> np.ndarray:
        return self.unlabeled[(seed, fraction)]

    def to_json(self) -> dict:
        return {
            "train": self.train_idx.tolist(),
            "val": self.val_idx.tolist(),
            "test": self.test_idx.tolist(),
            "labeled": {f"{s}_{int(round(f * 100))}": v.tolist()
                        for (s, f), v in sorted(self.labeled.items())},
            "unlabeled": {f"{s}_{int(round(f * 100))}": v.tolist()
                          for (s, f), v in sorted(self.unlabeled.items())},
        }


def get_cifar10(root: Path = DATA_DIR, download: bool = True) -> Tuple[CIFAR10, CIFAR10]:
    """Trả về (trainset, testset) với ảnh PIL."""
    train = CIFAR10(root=str(root), train=True, download=download)
    test = CIFAR10(root=str(root), train=False, download=download)
    return train, test


def _stratified_counts(labels: np.ndarray, classes: Sequence[int], total: int) -> Dict[int, int]:
    """Chia ``total`` mẫu theo tỷ lệ thực tế của từng lớp trong ``labels``."""
    labels = np.asarray(labels)
    counts: Dict[int, int] = {}
    for c in classes:
        counts[c] = int(np.sum(labels == c))
    n_all = sum(counts.values())
    raw = {c: counts[c] / n_all * total for c in classes}
    alloc = {c: int(np.floor(raw[c])) for c in classes}
    remainder = total - sum(alloc.values())
    # chia phần dư cho các lớp có phần thập phân lớn nhất
    order = sorted(classes, key=lambda c: (-(raw[c] - alloc[c]), c))
    for i in range(remainder):
        alloc[order[i % len(order)]] += 1
    return alloc


def build_benchmark(
    train_targets: Sequence[int],
    split_seed: int = SPLIT_SEED,
    repetitions: Sequence[int] = REPETITION_SEEDS,
    fractions: Sequence[float] = LABEL_FRACTIONS,
    cache: Path | None = ARTIFACT_DIR / "splits.json",
) -> Benchmark:
    """Tạo phép chia benchmark; nếu có cache thì nạp lại."""
    if cache is not None and Path(cache).exists():
        raw = json.loads(Path(cache).read_text(encoding="utf-8"))
        labeled = {(int(k.split("_")[0]), int(k.split("_")[1]) / 100.0): np.asarray(v, dtype=np.int64)
                   for k, v in raw["labeled"].items()}
        unlabeled = {(int(k.split("_")[0]), int(k.split("_")[1]) / 100.0): np.asarray(v, dtype=np.int64)
                     for k, v in raw["unlabeled"].items()}
        return Benchmark(
            train_idx=np.asarray(raw["train"], dtype=np.int64),
            val_idx=np.asarray(raw["val"], dtype=np.int64),
            test_idx=np.asarray(raw["test"], dtype=np.int64),
            labeled=labeled,
            unlabeled=unlabeled,
        )

    targets = np.asarray(train_targets, dtype=np.int64)
    assert targets.size == 50_000, "CIFAR-10 trainset phải có 50.000 ảnh"
    rng = np.random.default_rng(split_seed)

    val_alloc = _stratified_counts(targets, list(range(10)), VAL_SIZE)
    selected: List[int] = []
    for c in range(10):
        idx_c = np.flatnonzero(targets == c)
        perm = rng.permutation(idx_c)
        selected.extend(perm[: val_alloc[c]].tolist())
    selected = np.asarray(sorted(selected), dtype=np.int64)
    mask = np.ones(targets.size, dtype=bool)
    mask[selected] = False
    train_idx = np.flatnonzero(mask)

    val_super = np.array([ORIGINAL_TO_SUPER[int(t)] for t in targets[selected]], dtype=np.int64)
    assert int(np.sum(val_super == 0)) == VAL_SUPER_COUNTS[0], "D_val phải có 3.000 Animal"
    assert int(np.sum(val_super == 1)) == VAL_SUPER_COUNTS[1], "D_val phải có 2.000 Vehicle"

    test_idx = np.arange(10_000, dtype=np.int64)  # đánh dấu D_test ở không gian riêng

    labeled: Dict[Tuple[int, float], np.ndarray] = {}
    unlabeled: Dict[Tuple[int, float], np.ndarray] = {}
    for seed in repetitions:
        rng_s = np.random.default_rng(seed)
        # thứ tự ngẫu nhiên theo từng lớp gốc trên D_train
        per_class_order = {
            c: rng_s.permutation(np.flatnonzero(targets[train_idx] == c)) for c in range(10)
        }
        for fraction in fractions:
            budget = LABEL_BUDGET[fraction]
            want_animal = budget["animal"]
            want_vehicle = budget["vehicle"]
            assert want_animal + want_vehicle == budget["total"]

            def draw(cls_list, need):
                sizes = {c: per_class_order[c].size for c in cls_list}
                total = sum(sizes.values())
                raw = {c: sizes[c] / total * need for c in cls_list}
                alloc = {c: int(np.floor(raw[c])) for c in cls_list}
                rem = need - sum(alloc.values())
                order = sorted(cls_list, key=lambda c: (-(raw[c] - alloc[c]), c))
                for i in range(rem):
                    alloc[order[i % len(order)]] += 1
                return alloc

            pick_animal = draw(ANIMAL_CLASSES, want_animal)
            pick_vehicle = draw(VEHICLE_CLASSES, want_vehicle)

            positions: List[np.ndarray] = []
            for c, n in {**pick_animal, **pick_vehicle}.items():
                order = per_class_order[c]
                # nhãn gốc của D_train tại vị trí cục bộ
                positions.append(order[:n])
            local = np.concatenate(positions)
            n_pool = targets[train_idx].size
            global_idx = np.sort(train_idx[local])
            assert global_idx.size == budget["total"], "D_L phải đúng số lượng bảng 1"
            labeled[(seed, fraction)] = global_idx
            unlabeled[(seed, fraction)] = np.sort(np.setdiff1d(train_idx, global_idx, assume_unique=False))
            assert unlabeled[(seed, fraction)].size == n_pool - budget["total"]

    bench = Benchmark(train_idx=train_idx, val_idx=selected, test_idx=test_idx,
                      labeled=labeled, unlabeled=unlabeled)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(bench.to_json()), encoding="utf-8")
    return bench


def audit_benchmark(bench: Benchmark, train_targets: Sequence[int]) -> Dict[str, object]:
    """Kiểm tra tính đúng đắn của phép chia (mục 2.1 tiêu chí tuần 1)."""
    targets = np.asarray(train_targets, dtype=np.int64)
    report: Dict[str, object] = {}
    report["no_overlap_train_val"] = bool(len(np.intersect1d(bench.train_idx, bench.val_idx)) == 0)
    report["sizes"] = {
        "D_train": int(bench.train_idx.size),
        "D_val": int(bench.val_idx.size),
        "D_test": 10_000,
    }
    nesting = {}
    for seed in sorted({k[0] for k in bench.labeled}):
        sizes = {}
        prev = None
        ok = True
        for frac in sorted({k[1] for k in bench.labeled if k[0] == seed}):
            cur = set(bench.labeled[(seed, frac)].tolist())
            sizes[int(round(frac * 100))] = len(cur)
            if prev is not None and not prev.issubset(cur):
                ok = False
            prev = cur
        nesting[str(seed)] = {"nested": ok, "sizes": sizes}
    report["nesting"] = nesting

    leak = {}
    for (seed, frac), dl in sorted(bench.labeled.items()):
        du = bench.du(seed, frac)
        leak[f"s{seed}_{int(round(frac * 100))}"] = {
            "dl": int(dl.size),
            "du": int(du.size),
            "disjoint": bool(len(np.intersect1d(dl, du)) == 0),
            "no_val_leak": bool(len(np.intersect1d(np.concatenate([dl, du]), bench.val_idx)) == 0),
        }
    report["label_budget"] = leak

    report["train_animal_share"] = float(np.mean([ORIGINAL_TO_SUPER[int(t)] == 0 for t in targets[bench.train_idx]]))
    report["val_animal_share"] = float(np.mean([ORIGINAL_TO_SUPER[int(t)] == 0 for t in targets[bench.val_idx]]))
    return report


# --------------------------------------------------------------------------
# DataLoader
# --------------------------------------------------------------------------


def make_loader(
    dataset: Dataset,
    shuffle: bool,
    batch_size: int = BATCH_SIZE,
    seed: int | None = None,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY and torch.cuda.is_available(),
        drop_last=drop_last,
        generator=generator,
    )


def make_run_loaders(
    train_base: CIFAR10,
    test_base: CIFAR10,
    bench: Benchmark,
    seed: int,
    fraction: float,
    rotation_source: str,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    contrastive_kind: str | None = None,
    contrastive_source: str | None = None,
) -> Dict[str, DataLoader | Dataset]:
    """Sinh các DataLoader cho một lượt chạy.

    ``rotation_source``: ``none`` (không có nhánh xoay), ``labeled`` (chỉ ``D_L``)
    hoặc ``train`` (cả ``D_train``).

    ``contrastive_kind``: ``None``, ``"ntxent"`` hoặc ``"supcon"``.
    ``contrastive_source``: ``"train"`` (E3/E5) hoặc ``"labeled"`` (E4/E6).
    """
    dl_idx = bench.dl(seed, fraction)
    du_idx = bench.du(seed, fraction)

    labeled_ds = SuperclassDataset(train_base, dl_idx, labeled_train_transform())
    val_ds = SuperclassDataset(train_base, bench.val_idx, eval_transform())
    test_ds = SuperclassDataset(test_base, bench.test_idx, eval_transform())

    out: Dict[str, DataLoader | Dataset] = {
        "labeled_ds": labeled_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "labeled_loader": make_loader(labeled_ds, shuffle=True, batch_size=batch_size,
                                      seed=seed, drop_last=False, num_workers=num_workers),
        "val_loader": make_loader(val_ds, shuffle=False, batch_size=512, num_workers=num_workers),
        "test_loader": make_loader(test_ds, shuffle=False, batch_size=512, num_workers=num_workers),
    }
    if rotation_source in ("labeled", "train"):
        rot_idx = dl_idx if rotation_source == "labeled" else bench.train_idx
        rot_ds = RotationViewDataset(train_base, rot_idx)
        out["rotation_ds"] = rot_ds
        out["rotation_loader"] = make_loader(rot_ds, shuffle=True, batch_size=batch_size,
                                             seed=seed + 1000, drop_last=True,
                                             num_workers=num_workers)

    if contrastive_kind is not None:
        if contrastive_source not in ("train", "labeled"):
            raise ValueError("contrastive_source phải là 'train' hoặc 'labeled'")
        con_idx = bench.train_idx if contrastive_source == "train" else dl_idx
        # SupCon cần nhãn siêu lớp; NT-Xent thì không cần nên dataset không mang nhãn
        return_super = contrastive_kind == "supcon"
        con_ds = ContrastiveViewDataset(train_base, con_idx, return_superclass=return_super)
        out["contrastive_ds"] = con_ds
        out["contrastive_loader_main"] = make_loader(
            con_ds, shuffle=True, batch_size=batch_size, seed=seed + 2000,
            drop_last=True, num_workers=num_workers,
        )
        # Bản thứ hai, nguồn ngẫu nhiên riêng: dùng khi cấu hình vừa có nhánh xoay vừa
        # có nhánh contrastive (E5/E6) để hai nhánh không dùng chung chuỗi minibatch.
        out["contrastive_loader_aux"] = make_loader(
            con_ds, shuffle=True, batch_size=batch_size, seed=seed + 3000,
            drop_last=True, num_workers=num_workers,
        )
    return out
