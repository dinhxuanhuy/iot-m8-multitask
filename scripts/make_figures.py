"""Sinh hình cho báo cáo M8 từ kết quả thật trong ``results/``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m8.config import CHECKPOINT_DIR, FIGURES_DIR, RESULTS_DIR, MAIN_CONFIGS  # noqa: E402
from m8.experiments import load_all_runs, summarize_runs  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 160,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.autolayout": False,
    }
)

CONFIG_COLORS = {"E1": "#1f77b4", "E2-L": "#ff7f0e", "E2": "#2ca02c"}
SUPERCLASS_NAMES = ("Animal", "Vehicle")


def _runs_for(runs: Dict[str, dict], config: str, pct: int) -> List[dict]:
    return [r for r in runs.values() if r["spec"]["config"] == config and r["spec"]["label_pct"] == pct]


def _mean_curve(runs: Sequence[dict], key: str) -> np.ndarray:
    series = [np.asarray([h[key] for h in r["history"]], dtype=float) for r in runs]
    n = min(len(s) for s in series)
    return np.mean([s[:n] for s in series], axis=0)


def fig_learning_curves(runs: Dict[str, dict], pct: int = 10) -> Path:
    """Hình 1: đường học train/validation (tổng hợp 3 lần lặp)."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for config in MAIN_CONFIGS:
        sel = _runs_for(runs, config, pct)
        if not sel:
            continue
        color = CONFIG_COLORS[config]
        ep = np.arange(1, len(_mean_curve(sel, "train_total_loss")) + 1)
        axes[0].plot(ep, _mean_curve(sel, "train_total_loss"), color=color, label=config)
        axes[1].plot(ep, _mean_curve(sel, "val_sup_loss"), color=color, label=config)
        axes[2].plot(ep, _mean_curve(sel, "val_macro_f1"), color=color, label=config)

    axes[0].set_title("(a) Loss huấn luyện")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(r"$L_{sup}+\lambda_{rot}L_{rot}$")
    axes[1].set_title("(b) Loss phân loại trên validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(r"$L_{sup}$ (validation)")
    axes[2].set_title("(c) Macro-F1 validation")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].set_ylim(0.5, 1.0)
    for ax in axes:
        ax.legend(loc="best")
    fig.suptitle(f"Đường học tại mức {pct}% nhãn (trung bình 3 lần lặp)", y=1.03)
    out = FIGURES_DIR / f"fig01_learning_curves_L{pct}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_macro_f1_by_budget(summary: Dict[str, object]) -> Path:
    """Hình 2: Macro-F1 theo tỷ lệ nhãn, kèm độ lệch chuẩn qua ba lần lặp."""
    agg = summary["aggregate"]
    pcts = sorted({int(k.split("_")[1]) for k in agg})
    x = np.arange(len(pcts))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for i, config in enumerate(MAIN_CONFIGS):
        means, stds = [], []
        for pct in pcts:
            a = agg.get(f"{config}_{pct}")
            means.append(a["test_macro_f1"]["mean"] if a else np.nan)
            stds.append(a["test_macro_f1"]["std"] if a else 0.0)
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=4,
               color=CONFIG_COLORS[config], label=config, edgecolor="white", linewidth=0.6)
        for xi, m in zip(x + (i - 1) * width, means):
            if not np.isnan(m):
                ax.text(xi, m + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in pcts])
    ax.set_xlabel("Tỷ lệ nhãn huấn luyện")
    ax.set_ylabel("Macro-F1 trên D_test")
    ax.set_ylim(0.0, 1.05)
    ax.axhline(0.60, ls=":", lw=1.0, color="gray")
    ax.text(len(pcts) - 0.55, 0.612, "đoán hết Animal = 0,60", fontsize=7.5, color="gray")
    ax.legend(loc="lower right", title="Cấu hình")
    ax.set_title("Macro-F1 theo ngân sách nhãn (mean ± std, 3 lần lặp)")
    out = FIGURES_DIR / "fig02_macro_f1_by_budget.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_paired_delta(summary: Dict[str, object]) -> Path:
    """Hình 3: chênh lệch ghép cặp Δ_s theo từng lần lặp (điểm phần trăm)."""
    deltas = summary["paired_delta"]
    labels, series = [], []
    for key in sorted(deltas):
        labels.append(key)
        series.append(deltas[key]["deltas_pp"])
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = np.arange(len(labels))
    for i, (lab, vals) in enumerate(zip(labels, series)):
        ax.scatter([i] * len(vals), vals, s=42, zorder=3,
                   color=CONFIG_COLORS["E2"] if "E2_minus" in lab else CONFIG_COLORS["E2-L"])
        ax.plot([i - 0.18, i + 0.18], [np.mean(vals)] * 2, color="black", lw=1.6, zorder=4)
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([lab.replace("_minus_E1_", " − E1 tại ").replace("E2-L", "E2-L").replace("_", " ") + "%"
                        for lab in labels], rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("Δ Macro-F1 (điểm phần trăm)")
    ax.set_title("Chênh lệch ghép cặp so với E1 (mỗi điểm là một lần lặp)")
    out = FIGURES_DIR / "fig03_paired_delta.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_confusion(runs: Dict[str, dict], pct: int = 10) -> Path:
    """Hình 4: ma trận nhầm lẫn của mỗi cấu hình (tổng ba lần lặp, mức 10%)."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))
    im = None
    for ax, config in zip(axes, MAIN_CONFIGS):
        sel = _runs_for(runs, config, pct)
        if not sel:
            ax.axis("off")
            ax.set_title(f"{config} (chưa có kết quả)")
            continue
        cm = np.sum([np.asarray(r["test"]["confusion"]) for r in sel], axis=0)
        norm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({norm[i, j] * 100:.1f}%)",
                        ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1], SUPERCLASS_NAMES)
        ax.set_yticks([0, 1], SUPERCLASS_NAMES)
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Nhãn thật")
        ax.set_title(f"{config}")
        ax.grid(False)
    fig.suptitle(f"Ma trận nhầm lẫn trên D_test, mức {pct}% nhãn (gộp 3 lần lặp)", y=1.05)
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Tỷ lệ theo hàng")
    out = FIGURES_DIR / f"fig04_confusion_L{pct}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_rotation_budget(summary: Dict[str, object]) -> Path:
    """Hình 5: ngân sách nhãn D_L / D_U ở ba mức tỷ lệ."""
    from m8.config import LABEL_BUDGET

    pcts = sorted(LABEL_BUDGET)
    labels = [int(round(p * 100)) for p in pcts]
    dl = [LABEL_BUDGET[p]["total"] for p in pcts]
    du = [LABEL_BUDGET[p]["unlabeled"] for p in pcts]
    x = np.arange(len(pcts))
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.bar(x, dl, 0.5, label=r"$D_L$ (có nhãn)", color="#2ca02c")
    ax.bar(x, du, 0.5, bottom=dl, label=r"$D_U$ (ẩn nhãn)", color="#c7c7c7")
    for xi, a, b in zip(x, dl, du):
        ax.text(xi, a / 2, f"{a:,}".replace(",", "."), ha="center", va="center",
                fontsize=8.5, color="white")
        ax.text(xi, a + b / 2, f"{b:,}".replace(",", "."), ha="center", va="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in labels])
    ax.set_xlabel("Tỷ lệ nhãn")
    ax.set_ylabel("Số ảnh trong D_train (45.000)")
    ax.legend(loc="upper left")
    ax.set_title("Ngân sách nhãn: D_L và D_U theo tỷ lệ")
    out = FIGURES_DIR / "fig05_label_budget.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def table_csv(runs: Dict[str, dict]) -> Path:
    import csv

    rows = []
    for tag, r in sorted(runs.items()):
        spec = r["spec"]
        rows.append(
            {
                "tag": tag,
                "config": spec["config"],
                "label_pct": spec["label_pct"],
                "seed": spec["seed"],
                "best_epoch": r["best_epoch"],
                "val_macro_f1": round(r["best_val"]["macro_f1"], 6),
                "val_accuracy": round(r["best_val"]["accuracy"], 6),
                "test_macro_f1": round(r["test"]["macro_f1"], 6),
                "test_accuracy": round(r["test"]["accuracy"], 6),
                "f1_animal": round(r["test"]["f1"]["Animal"], 6),
                "f1_vehicle": round(r["test"]["f1"]["Vehicle"], 6),
                "prec_animal": round(r["test"]["precision"]["Animal"], 6),
                "rec_animal": round(r["test"]["recall"]["Animal"], 6),
                "prec_vehicle": round(r["test"]["precision"]["Vehicle"], 6),
                "rec_vehicle": round(r["test"]["recall"]["Vehicle"], 6),
                "train_seconds": round(r["timing"]["train_seconds"], 2),
                "inference_params": r["params"]["inference"],
                "training_params": r["params"]["training"],
            }
        )
    out = RESULTS_DIR / "runs_table.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return out


def main() -> int:
    runs = load_all_runs(CHECKPOINT_DIR)
    if not runs:
        print("Chưa có kết quả trong", CHECKPOINT_DIR)
        return 1
    print(f"Đọc {len(runs)} lượt chạy", flush=True)

    summary = summarize_runs(runs)
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    table_csv(runs)

    made = []
    pcts = sorted({r["spec"]["label_pct"] for r in runs.values()})
    made.append(fig_learning_curves(runs, pct=10))
    made.append(fig_macro_f1_by_budget(summary))
    made.append(fig_paired_delta(summary))
    if 10 in pcts:
        made.append(fig_confusion(runs, pct=10))
    made.append(fig_rotation_budget(summary))
    for p in made:
        print("  ->", p, flush=True)

    # số liệu bổ sung cho báo cáo
    extras = {}
    flat = summary["runs"]
    if flat:
        extras["best_overall"] = max(flat, key=lambda r: r["test_macro_f1"])
        extras["worst_overall"] = min(flat, key=lambda r: r["test_macro_f1"])
        extras["total_train_seconds"] = float(sum(r["train_seconds"] for r in flat))
        extras["total_runs"] = len(flat)
    (RESULTS_DIR / "summary_extras.json").write_text(
        json.dumps(extras, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
