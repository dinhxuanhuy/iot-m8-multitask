"""Sinh các hình minh họa **dữ liệu** cho mục 2 của báo cáo.

Gồm:
  * ``figA1_dataset_samples``  — lưới ảnh CIFAR-10 theo 10 lớp gốc, tô màu theo siêu lớp;
  * ``figA2_class_distribution`` — phân bố 10 lớp gốc và hai siêu lớp trên 3 tập;
  * ``figA3_views``            — ba loại view của cùng một ảnh (phân loại / xoay / contrastive).

Mọi ảnh đều lấy trực tiếp từ CIFAR-10 đã tải, không sinh giả.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m8.config import (  # noqa: E402
    ANIMAL_CLASSES,
    CLASS_NAMES,
    FIGURES_DIR,
    VEHICLE_CLASSES,
)
from m8.data import (  # noqa: E402
    RotationViewDataset,
    build_benchmark,
    contrastive_view_transform,
    get_cifar10,
    labeled_train_transform,
)

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": False,
        "figure.autolayout": False,
    }
)

ANIMAL_COLOR = "#2ca02c"
VEHICLE_COLOR = "#1f77b4"


def fig_dataset_samples(rng: np.random.Generator, per_class: int = 6) -> Path:
    """Lưới ảnh theo 10 lớp gốc; nền khung thể hiện siêu lớp của lớp đó."""
    train, _ = get_cifar10(download=False)
    targets = np.asarray(train.targets)
    fig, axes = plt.subplots(
        len(CLASS_NAMES), per_class,
        figsize=(per_class * 0.95, len(CLASS_NAMES) * 1.12),
    )
    for row, cls in enumerate(CLASS_NAMES):
        is_animal = row in ANIMAL_CLASSES
        color = ANIMAL_COLOR if is_animal else VEHICLE_COLOR
        idx = np.flatnonzero(targets == row)
        pick = rng.choice(idx, size=per_class, replace=False)
        for col, i in enumerate(pick):
            ax = axes[row, col]
            ax.imshow(np.asarray(train[int(i)][0]))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.6)
            if col == 0:
                ax.set_ylabel(
                    f"{cls}\n({row})", rotation=0, ha="right", va="center",
                    fontsize=8, color=color, labelpad=22,
                )
    handles = [
        Patch(facecolor="none", edgecolor=ANIMAL_COLOR, linewidth=1.6,
              label=f"Animal (nhãn mới 0): {', '.join(CLASS_NAMES[c] for c in ANIMAL_CLASSES)}"),
        Patch(facecolor="none", edgecolor=VEHICLE_COLOR, linewidth=1.6,
              label=f"Vehicle (nhãn mới 1): {', '.join(CLASS_NAMES[c] for c in VEHICLE_CLASSES)}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False,
               bbox_to_anchor=(0.5, -0.015), fontsize=8)
    fig.suptitle(
        f"Ảnh CIFAR-10 theo 10 lớp gốc ({per_class} ảnh mỗi lớp) và phép gộp hai siêu lớp của đề tài",
        y=1.005, fontsize=10,
    )
    out = FIGURES_DIR / "figA1_dataset_samples.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_class_distribution() -> Path:
    """Phân bố lớp gốc và siêu lớp trên D_train / D_val / D_test."""
    train, test = get_cifar10(download=False)
    bench = build_benchmark(train.targets)

    splits = {
        "D_train (45.000)": np.asarray(train.targets)[bench.train_idx],
        "D_val (5.000)": np.asarray(train.targets)[bench.val_idx],
        "D_test (10.000)": np.asarray(test.targets)[bench.test_idx],
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))

    # (a) số ảnh mỗi lớp gốc
    x = np.arange(len(CLASS_NAMES))
    width = 0.26
    for i, (name, t) in enumerate(splits.items()):
        counts = [int(np.sum(t == c)) for c in range(10)]
        axes[0].bar(x + (i - 1) * width, counts, width, label=name)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=7.5)
    axes[0].set_ylabel("Số ảnh")
    axes[0].set_title("(a) Phân bố 10 lớp gốc")
    axes[0].legend(fontsize=7.5)
    axes[0].grid(alpha=0.3, axis="y")

    # (b) tỷ lệ hai siêu lớp
    labels = ["Animal", "Vehicle"]
    for i, (name, t) in enumerate(splits.items()):
        n_animal = int(np.sum([np.sum(t == c) for c in ANIMAL_CLASSES]))
        n_vehicle = int(np.sum([np.sum(t == c) for c in VEHICLE_CLASSES]))
        total = n_animal + n_vehicle
        axes[1].bar(i, n_animal / total, 0.6, color=ANIMAL_COLOR, label="Animal" if i == 0 else None)
        axes[1].bar(i, n_vehicle / total, 0.6, bottom=n_animal / total,
                    color=VEHICLE_COLOR, label="Vehicle" if i == 0 else None)
        axes[1].text(i, n_animal / total / 2, f"{n_animal / total * 100:.1f} %",
                     ha="center", va="center", color="white", fontsize=9)
        axes[1].text(i, n_animal / total + n_vehicle / total / 2,
                     f"{n_vehicle / total * 100:.1f} %", ha="center", va="center",
                     color="white", fontsize=9)
    axes[1].set_xticks(range(len(splits)))
    axes[1].set_xticklabels([n.split(" ")[0] for n in splits], fontsize=8)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Tỷ lệ")
    axes[1].set_title("(b) Tỷ lệ siêu lớp 60/40 không cân bằng")
    axes[1].legend(fontsize=8, loc="lower center", ncol=2)

    # (c) ngân sách nhãn theo ba mức
    from m8.config import LABEL_BUDGET

    pcts = sorted(LABEL_BUDGET)
    dl = [LABEL_BUDGET[p]["total"] for p in pcts]
    du = [LABEL_BUDGET[p]["unlabeled"] for p in pcts]
    xi = np.arange(len(pcts))
    axes[2].bar(xi, dl, 0.5, color=ANIMAL_COLOR, label=r"$D_L$ (có nhãn)")
    axes[2].bar(xi, du, 0.5, bottom=dl, color="#c7c7c7", label=r"$D_U$ (ẩn nhãn)")
    for i, (a, b) in enumerate(zip(dl, du)):
        axes[2].text(i, a / 2, f"{a:,}".replace(",", "."), ha="center", va="center",
                     color="white", fontsize=8.5)
        axes[2].text(i, a + b / 2, f"{b:,}".replace(",", "."), ha="center", va="center", fontsize=8.5)
    axes[2].set_xticks(xi)
    axes[2].set_xticklabels([f"{int(p * 100)} %" for p in pcts])
    axes[2].set_xlabel("Tỷ lệ nhãn")
    axes[2].set_ylabel("Số ảnh trong $D_{train}$")
    axes[2].set_title("(c) Ngân sách nhãn trên 45.000 ảnh train")
    axes[2].legend(fontsize=8)

    fig.suptitle("Thành phần dữ liệu của phép chia benchmark", y=1.03, fontsize=10)
    out = FIGURES_DIR / "figA2_class_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_views(train, bench, rng: np.random.Generator) -> Path:
    """Ba loại view của cùng một ảnh: phân loại / xoay / controstive."""
    idx = int(bench.dl(42, 0.10)[0])
    img, original = train[idx]
    pil = img

    lab_view = labeled_train_transform()(pil)
    con_tf = contrastive_view_transform()
    con1, con2 = con_tf(pil), con_tf(pil)
    rot_ds = RotationViewDataset(train, [idx] * 4)
    rot_views = [rot_ds[k][0] for k in range(4)]

    def to_disp(t: "torch.Tensor"):
        a = t.detach().cpu().numpy().transpose(1, 2, 0)
        return np.clip(a * 0.5 + 0.5, 0, 1)

    fig = plt.figure(figsize=(13.5, 3.4))
    gs = fig.add_gridspec(1, 8, wspace=0.08)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(np.asarray(pil))
    ax.set_title("Ảnh gốc", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("black"); s.set_linewidth(1.2)

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(to_disp(lab_view))
    ax.set_title("Nhánh phân loại\n(crop + flip)", fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(to_disp(con1))
    ax.set_title("Contrastive\nview 1", fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(to_disp(con2))
    ax.set_title("Contrastive\nview 2", fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])

    for k in range(4):
        ax = fig.add_subplot(gs[0, 4 + k])
        ax.imshow(to_disp(rot_views[k]))
        ax.set_title(f"Nhánh xoay\nk = {k}", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#C55A11"); s.set_linewidth(1.3)

    fig.suptitle(
        f"Các loại view của cùng một ảnh — ảnh đầu tiên của $D_L$ tại mức 10 % nhãn, seed 42 "
        f"(lớp gốc: {CLASS_NAMES[original]})",
        y=1.08, fontsize=9.5,
    )
    fig.text(
        0.5, -0.04,
        "Nhánh phân loại dùng RandomCrop + RandomHorizontalFlip; hai view contrastive dùng tăng cường mạnh "
        "độc lập (crop, flip, color jitter, grayscale); nhánh xoay dùng ảnh gốc 32×32 không cắt và "
        "xoay 90k độ ngược chiều kim đồng hồ.",
        ha="center", va="top", fontsize=8.5,
    )
    out = FIGURES_DIR / "figA3_views.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    rng = np.random.default_rng(2026)
    train, _ = get_cifar10(download=False)
    bench = build_benchmark(train.targets)

    made = [
        fig_dataset_samples(rng),
        fig_class_distribution(),
        fig_views(train, bench, rng),
    ]
    for p in made:
        print("  ->", p, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
