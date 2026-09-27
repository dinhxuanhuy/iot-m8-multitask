"""Hằng số và cấu hình giao thức cho đề tài M8.

Mọi giá trị ở đây bám theo đề cương đã chỉnh sửa (bảng 1, 3, 6, 7).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence, Tuple

# --------------------------------------------------------------------------
# Đường dẫn
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = DATA_DIR / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORT_DIR = PROJECT_ROOT / "report"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
LOG_DIR = RESULTS_DIR / "logs"

for _d in (DATA_DIR, ARTIFACT_DIR, RESULTS_DIR, FIGURES_DIR, CHECKPOINT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

#: Thư mục checkpoint của khối mở rộng contrastive E3--E6 (tách khỏi khối chính)
CHECKPOINT_EXT_DIR = RESULTS_DIR / "checkpoints_ext"
CHECKPOINT_EXT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Ánh xạ siêu lớp (bảng 2 của đề cương)
# Thứ tự nhãn gốc CIFAR-10:
#   0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer,
#   5 dog, 6 frog, 7 horse, 8 ship, 9 truck
# --------------------------------------------------------------------------
CLASS_NAMES: Tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

ANIMAL_CLASSES: Tuple[int, ...] = (2, 3, 4, 5, 6, 7)
VEHICLE_CLASSES: Tuple[int, ...] = (0, 1, 8, 9)

SUPERCLASS_NAMES: Tuple[str, ...] = ("Animal", "Vehicle")

#: nhãn gốc -> nhãn siêu lớp (0 = Animal, 1 = Vehicle)
ORIGINAL_TO_SUPER: Dict[int, int] = {c: 0 for c in ANIMAL_CLASSES}
ORIGINAL_TO_SUPER.update({c: 1 for c in VEHICLE_CLASSES})

# --------------------------------------------------------------------------
# Giao thức
# --------------------------------------------------------------------------
SPLIT_SEED: int = 2026                     # chia D_train / D_val phân tầng theo 10 lớp gốc
REPETITION_SEEDS: Tuple[int, ...] = (42, 43, 44)
LABEL_FRACTIONS: Tuple[float, ...] = (0.10, 0.20, 0.50)

TRAIN_SIZE: int = 45_000                   # D_train trong 50.000 ảnh train chính thức
VAL_SIZE: int = 5_000                      # D_val
NUM_CLASSES_ORIGINAL: int = 10
NUM_CLASSES_SUPER: int = 2
NUM_ROTATIONS: int = 4                     # k in {0, 1, 2, 3}

# Bảng 1: số ảnh có nhãn theo tỷ lệ (tính trên 45.000 ảnh train)
LABEL_BUDGET: Dict[float, Dict[str, int]] = {
    0.10: {"total": 4_500, "animal": 2_700, "vehicle": 1_800, "unlabeled": 40_500},
    0.20: {"total": 9_000, "animal": 5_400, "vehicle": 3_600, "unlabeled": 36_000},
    0.50: {"total": 22_500, "animal": 13_500, "vehicle": 9_000, "unlabeled": 22_500},
}
# Bảng 1: D_val = 3.000 Animal + 2.000 Vehicle; D_test = 6.000 Animal + 4.000 Vehicle
VAL_SUPER_COUNTS: Tuple[int, int] = (3_000, 2_000)
TEST_SUPER_COUNTS: Tuple[int, int] = (6_000, 4_000)

# --------------------------------------------------------------------------
# Huấn luyện (bảng 7)
# --------------------------------------------------------------------------
BATCH_SIZE: int = 128
EPOCHS: int = 100
LR: float = 3e-4
WEIGHT_DECAY: float = 1e-4
BETAS: Tuple[float, float] = (0.9, 0.999)
EPS: float = 1e-8
LAMBDA_ROT: float = 0.5                     # chốt cho thí nghiệm chính
NUM_WORKERS: int = 0                        # Windows: 0 để tránh fork trong notebook/CLI
PIN_MEMORY: bool = True

# --------------------------------------------------------------------------
# Mở rộng contrastive (mục 3.3, bảng 5 của đề cương — E3..E6)
# Mọi siêu tham số ở đây được khóa TRƯỚC khi chạy để không điều chỉnh theo kết quả test.
# --------------------------------------------------------------------------
LAMBDA_CONTRASTIVE: float = 0.5             # lambda_c cho mọi cấu hình mở rộng
PROJ_DIM: int = 128                         # chiều Projection Head dùng khi huấn luyện
TEMPERATURE_SUPCON: float = 0.1             # nhiệt độ cho L_SupCon
TEMPERATURE_NTXENT: float = 0.2             # nhiệt độ cho L_NT-Xent
SUPCON_BATCH_MULTIPLIER: int = 2            # "2N view" của SupCon: hai view mỗi mẫu

# Danh sách mức nhãn dùng cho khối mở rộng (10% là mức ưu tiên theo mục 4.2)
EXTENSION_LABEL_FRACTIONS: Tuple[float, ...] = (0.10, 0.20)

# Chuẩn hóa: hằng số chọn trước, áp dụng chung cho mọi cấu hình
NORMALIZE_MEAN: Tuple[float, float, float] = (0.5, 0.5, 0.5)
NORMALIZE_STD: Tuple[float, float, float] = (0.5, 0.5, 0.5)

# Tên cấu hình đối chứng (bảng 6)
CONFIG_E1 = "E1"        # baseline chỉ học siêu lớp, nguồn rotation: không có
CONFIG_E2L = "E2-L"     # thêm nhánh rotation, nguồn rotation: chỉ D_L
CONFIG_E2 = "E2"        # thêm nhánh rotation, nguồn rotation: D_train = D_L + D_U

# Cấu hình mở rộng contrastive (bảng 5)
CONFIG_E3 = "E3"        # L_sup + lambda_c * L_NT-Xent            (contrastive: D_train)
CONFIG_E4 = "E4"        # L_sup + lambda_c * L_SupCon            (SupCon: chỉ D_L)
CONFIG_E5 = "E5"        # L_sup + lambda_rot*L_rot + lambda_c*L_NT-Xent
CONFIG_E6 = "E6"        # L_sup + lambda_rot*L_rot + lambda_c*L_SupCon
EXTENSION_CONFIGS: Tuple[str, ...] = (CONFIG_E3, CONFIG_E4, CONFIG_E5, CONFIG_E6)

#: cấu hình -> (có nhánh xoay?, loại loss contrastive)
EXTENSION_SPEC: Dict[str, Dict[str, object]] = {
    CONFIG_E3: {"rotation": False, "contrastive": "ntxent"},
    CONFIG_E4: {"rotation": False, "contrastive": "supcon"},
    CONFIG_E5: {"rotation": True, "contrastive": "ntxent"},
    CONFIG_E6: {"rotation": True, "contrastive": "supcon"},
}

#: cấu hình -> nguồn ảnh cho nhánh contrastive ("train" hoặc "labeled")
CONTRASTIVE_SOURCE: Dict[str, str] = {
    CONFIG_E3: "train",
    CONFIG_E4: "labeled",
    CONFIG_E5: "train",
    CONFIG_E6: "labeled",
}
MAIN_CONFIGS: Tuple[str, ...] = (CONFIG_E1, CONFIG_E2L, CONFIG_E2)

# Khảo sát độ nhạy lambda (tùy chọn, mục 4.2)
LAMBDA_GRID: Tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)

# --------------------------------------------------------------------------
# Benchmark suy luận (mục 4.4)
# --------------------------------------------------------------------------
LATENCY_WARMUP: int = 50
LATENCY_ITERS: int = 1_000
LATENCY_SESSIONS: int = 3


@dataclass
class RunSpec:
    """Một lượt huấn luyện: (cấu hình, tỷ lệ nhãn, lần lặp)."""

    config: str
    label_fraction: float
    seed: int
    epochs: int = EPOCHS
    lambda_rot: float = LAMBDA_ROT
    lambda_c: float = LAMBDA_CONTRASTIVE
    tag: str = field(default="")

    def __post_init__(self) -> None:
        if not self.tag:
            self.tag = f"{self.config}_L{int(round(self.label_fraction * 100)):02d}_s{self.seed}"
        if self.config not in MAIN_CONFIGS and self.config not in EXTENSION_CONFIGS:
            raise ValueError(f"cấu hình không hợp lệ: {self.config}")

    # ---- thuộc tính suy ra từ cấu hình ----
    @property
    def rotation_source(self) -> str:
        r"""Nguồn ảnh cho nhánh rotation.

        ``none`` = E1/E3/E4 (không có nhánh xoay),
        ``labeled`` = E2-L (chỉ D_L),
        ``train`` = E2/E5 (cả D_train).
        """
        if self.config in (CONFIG_E1, CONFIG_E3, CONFIG_E4):
            return "none"
        if self.config == CONFIG_E2L:
            return "labeled"
        return "train"

    @property
    def contrastive_kind(self) -> str | None:
        """``"ntxent"``, ``"supcon"`` hoặc ``None`` nếu không có nhánh contrastive."""
        return EXTENSION_SPEC.get(self.config, {}).get("contrastive")  # type: ignore[return-value]

    @property
    def contrastive_source(self) -> str | None:
        """Nguồn ảnh cho nhánh contrastive: ``train``, ``labeled`` hoặc ``None``."""
        return CONTRASTIVE_SOURCE.get(self.config)

    @property
    def uses_rotation_head(self) -> bool:
        return self.rotation_source != "none"

    @property
    def uses_projection_head(self) -> bool:
        return self.contrastive_kind is not None

    @property
    def label_pct(self) -> int:
        return int(round(self.label_fraction * 100))

    def as_dict(self) -> dict:
        return {
            "tag": self.tag,
            "config": self.config,
            "label_fraction": self.label_fraction,
            "label_pct": self.label_pct,
            "seed": self.seed,
            "epochs": self.epochs,
            "lambda_rot": self.lambda_rot,
            "lambda_c": self.lambda_c,
            "rotation_source": self.rotation_source,
            "contrastive_kind": self.contrastive_kind,
            "contrastive_source": self.contrastive_source,
        }


def main_run_specs(lambdas: Sequence[float] | None = None) -> list[RunSpec]:
    """3 cấu hình x 3 mức nhãn x 3 lần lặp = 27 lượt (mục 4.2)."""
    lam = LAMBDA_ROT if lambdas is None else lambdas[0]
    return [
        RunSpec(config=c, label_fraction=f, seed=s, lambda_rot=lam)
        for c in MAIN_CONFIGS
        for f in LABEL_FRACTIONS
        for s in REPETITION_SEEDS
    ]


def extension_run_specs(
    fractions: Sequence[float] | None = None,
    seeds: Sequence[int] | None = None,
    configs: Sequence[str] | None = None,
) -> list[RunSpec]:
    """Khối mở rộng contrastive E3--E6: 4 cấu hình x mức nhãn x lần lặp."""
    return [
        RunSpec(config=c, label_fraction=f, seed=s)
        for c in (configs or EXTENSION_CONFIGS)
        for f in (fractions or EXTENSION_LABEL_FRACTIONS)
        for s in (seeds or REPETITION_SEEDS)
    ]


def device_name() -> str:
    import torch

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def set_global_threads(threads: int | None = None) -> int:
    """Ghim số luồng CPU để phép đo latency có ý nghĩa."""
    import torch

    if threads is None:
        threads = int(os.environ.get("M8_THREADS", "4"))
    torch.set_num_threads(threads)
    return threads
