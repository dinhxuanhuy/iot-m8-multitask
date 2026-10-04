"""Sinh hình **một số trường hợp dự đoán sai** trên tập test.

Yêu cầu của đề cương (mục 5.2): *"ma trận nhầm lẫn và một số trường hợp dự đoán sai"*.

Đầu vào: ``results/test_predictions/<tag>.npz`` do ``scripts/run_error_analysis.py`` sinh ra
(gồm ``test_index``, ``y_true``, ``y_pred``, ``prob``). Đầu ra:
``results/figures/fig09_misclassified_L10.png`` — mỗi hàng một cấu hình, mỗi cột một ảnh bị
phân loại sai, kèm nhãn thật, nhãn đoán và độ tự tin.

Ảnh CIFAR chỉ 32×32 px nên dùng ``interpolation="nearest"``; hình lưu ở 300 DPI.

Cách dùng::

    python scripts/make_error_figure.py
    python scripts/make_error_figure.py --configs E1 E2-L E2 --per-config 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m8.config import FIGURES_DIR, RESULTS_DIR, SUPERCLASS_NAMES  # noqa: E402
from m8.data import build_benchmark, get_cifar10  # noqa: E402

PRED_DIR = RESULTS_DIR / "test_predictions"

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.grid": False,
})

COLORS = {"E1": "#1f77b4", "E2-L": "#ff7f0e", "E2": "#2ca02c",
          "E3": "#9467bd", "E4": "#d62728", "E5": "#8c564b", "E6": "#e377c2"}


def pick_mistakes(pred: dict, k: int) -> np.ndarray:
    """Chọn ``k`` mẫu sai TRẢI ĐỀU theo độ tự tin.

    Lấy đều từ nhóm sai-chắc-nhất (độ tự tin cao) tới nhóm sai-sát-nước (độ tự tin thấp)
    để hình vừa cho thấy lỗi hệ thống, vừa cho thấy các ca ranh giới — chứ không chỉ
    toàn ca có độ tự tin 1,00.
    """
    wrong = np.flatnonzero(pred["y_true"] != pred["y_pred"])
    if wrong.size == 0:
        return wrong
    conf = pred["prob"][wrong, pred["y_pred"][wrong]]
    order = wrong[np.argsort(-conf)]          # giảm dần theo độ tự tin
    if wrong.size <= k:
        return order
    idx = np.linspace(0, wrong.size - 1, k).round().astype(int)
    return order[np.unique(idx)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Hình các trường hợp dự đoán sai")
    ap.add_argument("--configs", nargs="*", default=["E1", "E2-L", "E2"])
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-config", type=int, default=6)
    ap.add_argument("--name", type=str, default="fig09_misclassified_L10.png")
    args = ap.parse_args()

    tags = [f"{c}_L{int(round(args.fraction*100)):02d}_s{args.seed}" for c in args.configs]
    avail = [(c, t) for c, t in zip(args.configs, tags) if (PRED_DIR / f"{t}.npz").exists()]
    if not avail:
        print(f"  !! chưa có dự đoán nào trong {PRED_DIR}/")
        print("     Chạy trước:  python scripts/run_error_analysis.py")
        return 1

    train, test = get_cifar10(download=False)
    bench = build_benchmark(train.targets)
    # ảnh test gốc, đúng thứ tự 0..9999 mà run_error_analysis.py dùng
    test_images = np.stack([np.asarray(test[i][0]) for i in bench.test_idx])

    k = args.per_config
    fig, axes = plt.subplots(len(avail), k,
                             figsize=(k * 1.30, len(avail) * 1.85),
                             squeeze=False)

    for row, (cfg, tag) in enumerate(avail):
        pred = dict(np.load(PRED_DIR / f"{tag}.npz"))
        sel = pick_mistakes(pred, k)
        n_wrong = int((pred["y_true"] != pred["y_pred"]).sum())
        for col in range(k):
            ax = axes[row][col]
            if col >= sel.size:
                ax.axis("off")
                continue
            i = int(sel[col])
            ax.imshow(test_images[i], interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            true_l, pred_l = SUPERCLASS_NAMES[pred["y_true"][i]], SUPERCLASS_NAMES[pred["y_pred"][i]]
            conf = float(pred["prob"][i, pred["y_pred"][i]])
            ax.set_title(f"#{i}\n{true_l} → {pred_l}\n({conf:.2f})", fontsize=7.5, pad=3)
            for s in ax.spines.values():
                s.set_edgecolor(COLORS.get(cfg, "black"))
                s.set_linewidth(1.5)
            if col == 0:
                ax.set_ylabel(f"{cfg}\n{n_wrong} lỗi", rotation=0, ha="center",
                              va="center", fontsize=8.5,
                              color=COLORS.get(cfg, "black"), labelpad=30)

    fig.suptitle(
        f"Các trường hợp dự đoán sai trên $D_{{test}}$ — "
        f"mức ${int(round(args.fraction*100))}\\,\\%$ nhãn, seed {args.seed}\n"
        f"mỗi ô: chỉ mục ảnh · nhãn thật → nhãn đoán · độ tự tin của dự đoán sai "
        f"(chọn trải đều từ sai-chắc-nhất tới sát-nước)",
        y=1.06, fontsize=9.5,
    )
    fig.tight_layout(rect=(0.02, 0, 1, 0.99))
    out = FIGURES_DIR / args.name
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")
    print(f"     {len(avail)} cấu hình x {k} ví dụ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
