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
        # Ảnh CIFAR chỉ 32x32 px. DPI cao + interpolation="nearest" giữ cho các ô
        # pixel sắc nét khi phóng to; nội suy làm mượt sẽ biến ảnh thành nhoè.
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": 9,
        "axes.grid": False,
        "figure.autolayout": False,
        "image.interpolation": "nearest",
    }
)

ANIMAL_COLOR = "#2ca02c"
VEHICLE_COLOR = "#1f77b4"


def _show_cifar(ax, img, edgecolor, lw: float = 1.4) -> None:
    """Hiển thị một ảnh CIFAR giữ nguyên khối pixel (nearest), viền màu siêu lớp."""
    ax.imshow(np.asarray(img), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(edgecolor)
        spine.set_linewidth(lw)


def fig_dataset_samples(rng: np.random.Generator, per_class: int = 6) -> Path:
    """Lưới ảnh theo 10 lớp gốc: 5 hàng x 2 khối 6 cột.

    Bố cục 10 hàng x 6 cột (bản cũ) cho tỉ lệ 0,53 — nhúng ở ``width=\\textwidth``
    thành ~30 cm, VƯỢT chiều cao trang, nên LaTeX cắt mất hàng cuối, legend và cả
    caption. Bố cục 5x12 giữ tỉ lệ ~2,2 nên vừa trang và còn chỗ cho caption.
    """
    train, _ = get_cifar10(download=False)
    targets = np.asarray(train.targets)
    half = len(CLASS_NAMES) // 2          # 5 hàng
    ncols = per_class * 2 + 1             # 13 cột, cột giữa là đệm cho nhãn khối phải

    cell = 0.60
    # Chừa một dải ở đáy cho legend; nếu để axes tràn xuống sát đáy thì legend sẽ
    # ĐÈ LÊN hàng ảnh cuối (lỗi đã gặp khi legend neo ở y âm).
    fig = plt.figure(figsize=(ncols * cell, half * cell + 1.05))
    gs = fig.add_gridspec(
        half, ncols,
        width_ratios=[1] * per_class + [1.9] + [1] * per_class,
        wspace=0.10, hspace=0.10,
        left=0.004, right=0.996, top=0.945, bottom=0.135,
    )

    for row in range(half):
        for block in (0, 1):
            cls = row + block * half
            offset = 0 if block == 0 else per_class + 1
            color = ANIMAL_COLOR if cls in ANIMAL_CLASSES else VEHICLE_COLOR
            idx = np.flatnonzero(targets == cls)
            pick = rng.choice(idx, size=per_class, replace=False)
            for col, i in enumerate(pick):
                ax = fig.add_subplot(gs[row, offset + col])
                _show_cifar(ax, train[int(i)][0], color)
                if col == 0:
                    ax.set_ylabel(
                        f"{CLASS_NAMES[cls]}\n({cls})", rotation=0, ha="right", va="center",
                        fontsize=7.5, color=color, labelpad=14,
                    )

    handles = [
        Patch(facecolor="none", edgecolor=ANIMAL_COLOR, linewidth=1.6,
              label=f"Animal (nhãn mới 0): {', '.join(CLASS_NAMES[c] for c in ANIMAL_CLASSES)}"),
        Patch(facecolor="none", edgecolor=VEHICLE_COLOR, linewidth=1.6,
              label=f"Vehicle (nhãn mới 1): {', '.join(CLASS_NAMES[c] for c in VEHICLE_CLASSES)}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False,
               bbox_to_anchor=(0.5, 0.005), fontsize=8)
    fig.suptitle(
        f"Ảnh CIFAR-10 theo 10 lớp gốc ({per_class} ảnh mỗi lớp) và phép gộp hai siêu lớp của đề tài",
        y=0.99, fontsize=10,
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

    cell = 2.55
    # 2 hàng x 4 cột (thay vì 1 hàng x 8 cột): mỗi panel rộng gấp đôi nên ảnh 32x32
    # nhìn rõ hơn hẳn. Hàng 1 = các view nguồn, hàng 2 = bốn phép xoay.
    fig = plt.figure(figsize=(4 * cell + 0.4, 2 * cell + 1.55))
    gs = fig.add_gridspec(2, 4, wspace=0.07, hspace=0.26,
                          left=0.008, right=0.992, top=0.905, bottom=0.125)

    panels = [
        (np.asarray(pil), "Ảnh gốc", "black", 1.2),
        (to_disp(lab_view), "Nhánh phân loại\n(crop + flip)", "black", 1.0),
        (to_disp(con1), "Contrastive\nview 1", "black", 1.0),
        (to_disp(con2), "Contrastive\nview 2", "black", 1.0),
    ]
    panels += [
        (to_disp(rot_views[k]), f"Nhánh xoay\nk = {k}", "#C55A11", 1.3) for k in range(4)
    ]

    for pos, (arr, title, edge, lw) in enumerate(panels):
        ax = fig.add_subplot(gs[pos // 4, pos % 4])
        # nearest: giữ nguyên khối pixel 32x32, không nội suy làm mượt
        ax.imshow(arr, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(edge)
            s.set_linewidth(lw)

    fig.suptitle(
        f"Các loại view của cùng một ảnh — ảnh đầu tiên của $D_L$ tại mức 10 % nhãn, seed 42 "
        f"(lớp gốc: {CLASS_NAMES[original]})",
        y=0.985, fontsize=10,
    )
    fig.text(
        0.5, 0.012,
        "Nhánh phân loại dùng RandomCrop + RandomHorizontalFlip; hai view contrastive dùng tăng cường mạnh "
        "độc lập (crop, flip, color jitter, grayscale); nhánh xoay dùng ảnh gốc 32×32 không cắt và "
        "xoay 90k độ ngược chiều kim đồng hồ.",
        ha="center", va="bottom", fontsize=9,
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
