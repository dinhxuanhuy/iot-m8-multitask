"""Sinh hình cho khối mở rộng contrastive (E3--E6).

  * ``fig06_extension_macro_f1`` — Macro-F1 của 7 cấu hình theo mức nhãn;
  * ``fig07_extension_delta``   — chênh lệch ghép cặp Δ_s của E3--E6 so với E1;
  * ``fig08_lambda_sweep``      — đường khảo sát λ_c trên Macro-F1 **validation**.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m8.config import CHECKPOINT_DIR, CHECKPOINT_EXT_DIR, FIGURES_DIR, RESULTS_DIR  # noqa: E402
from m8.experiments import load_all_runs_multi, summarize_runs  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 160,
        # 300 DPI cho đủ nét khi nhúng ở bề rộng \textwidth (xem make_figures.py).
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

COLORS = {
    "E1": "#1f77b4",
    "E2-L": "#ff7f0e",
    "E2": "#2ca02c",
    "E3": "#9467bd",
    "E4": "#d62728",
    "E5": "#8c564b",
    "E6": "#e377c2",
}
MAIN = ("E1", "E2-L", "E2")
EXT = ("E3", "E4", "E5", "E6")


def fig_extension_macro_f1(summary: dict) -> Path:
    agg = summary["aggregate"]
    have = {r["config"] for r in summary["runs"]}
    pcts = sorted({int(k.split("_")[1]) for k in agg})
    configs = [c for c in (*MAIN, *EXT) if c in have]
    x = np.arange(len(pcts))
    width = 0.9 / max(len(configs), 1)
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    for i, cfg in enumerate(configs):
        means, stds = [], []
        for pct in pcts:
            a = agg.get(f"{cfg}_{pct}")
            means.append(a["test_macro_f1"]["mean"] if a else np.nan)
            stds.append(a["test_macro_f1"]["std"] if a else 0.0)
        ax.bar(x + (i - (len(configs) - 1) / 2) * width, means, width, yerr=stds,
               capsize=3, color=COLORS.get(cfg, "gray"), label=cfg,
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p} %" for p in pcts])
    ax.set_xlabel("Tỷ lệ nhãn huấn luyện")
    ax.set_ylabel("Macro-F1 trên $D_{test}$")
    ax.set_ylim(0.5, 1.02)
    ax.axhline(0.60, ls=":", lw=1.0, color="gray")
    ax.text(len(pcts) - 0.5, 0.615, "đoán hết Animal = 0,60", fontsize=8, color="gray")
    ax.legend(ncol=4, fontsize=8.5, loc="lower right", title="Cấu hình")
    ax.set_title("Macro-F1 của khối chính và khối mở rộng contrastive (mean ± std, 3 lần lặp)")
    out = FIGURES_DIR / "fig06_extension_macro_f1.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _placeholder(path: Path, message: str) -> Path:
    """Tạo hình tạm khi chưa có dữ liệu — KHÔNG để tệp rỗng (LaTeX không đọc được)."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, wrap=True)
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_extension_delta(summary: dict) -> Path:
    deltas = summary["paired_delta"]
    keys = [k for k in sorted(deltas) if k.split("_minus_")[0] in EXT]
    out = FIGURES_DIR / "fig07_extension_delta.png"
    if not keys:
        return _placeholder(out, "Chưa có kết quả khối mở rộng\n(không có chênh lệch ghép cặp)")
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    x = np.arange(len(keys))
    for i, key in enumerate(keys):
        vals = deltas[key]["deltas_pp"]
        cfg = key.split("_minus_")[0]
        ax.scatter([i] * len(vals), vals, s=48, zorder=3, color=COLORS.get(cfg, "gray"))
        ax.plot([i - 0.2, i + 0.2], [np.mean(vals)] * 2, color="black", lw=1.6, zorder=4)
    ax.axhline(0.0, color="black", lw=0.9)
    labels = []
    for k in keys:
        cfg, rest = k.split("_minus_")
        base, pct = rest.rsplit("_", 1)
        labels.append(f"{cfg} − {base}\n{pct} % nhãn")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Δ Macro-F1 (điểm phần trăm)")
    ax.set_title("Chênh lệch ghép cặp so với E1 của các cấu hình mở rộng (mỗi điểm = một lần lặp)")
    out = FIGURES_DIR / "fig07_extension_delta.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_lambda_sweep() -> Path:
    out = FIGURES_DIR / "fig08_lambda_sweep.png"
    path = RESULTS_DIR / "lambda_c_sweep.json"
    if not path.exists():
        return _placeholder(out, "Chưa có khảo sát λ_c")
    data = json.loads(path.read_text(encoding="utf-8"))
    recs = data.get("records", [])
    if not recs:
        return _placeholder(out, "Chưa có khảo sát λ_c")
    configs = sorted({r["config"] for r in recs})
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for cfg in configs:
        sel = sorted([r for r in recs if r["config"] == cfg], key=lambda r: r["lambda_c"])
        xs = [r["lambda_c"] for r in sel]
        ys = [r["best_val_macro_f1"] for r in sel]
        ax.plot(xs, ys, marker="o", label=cfg, color=COLORS.get(cfg, "gray"))
        for r in sel:
            ax.annotate(f"{r['best_val_macro_f1']:.4f}", (r["lambda_c"], r["best_val_macro_f1"]),
                        textcoords="offset points", xytext=(0, 7), fontsize=7.5, ha="center")
    chosen = data.get("chosen", {})
    for cfg, info in chosen.items():
        ax.axvline(info["lambda_c"], ls="--", lw=1.0, color=COLORS.get(cfg, "gray"), alpha=0.6)
    ax.set_xscale("log")
    ax.set_xticks(sorted({r["lambda_c"] for r in recs}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel(r"$\lambda_c$")
    ax.set_ylabel("Macro-F1 validation tốt nhất")
    ax.set_title(f"Khảo sát $\\lambda_c$ CHỈ trên validation ({data.get('epochs')} epoch, "
                 f"mức {int(data.get('fraction', 0) * 100)} % nhãn, seed {data.get('seed')})")
    ax.legend(title="Cấu hình")
    out2 = FIGURES_DIR / "fig08_lambda_sweep.png"
    fig.savefig(out2)
    plt.close(fig)
    return out2


def main() -> int:
    runs = load_all_runs_multi(CHECKPOINT_DIR, CHECKPOINT_EXT_DIR)
    if not runs:
        print("Chưa có kết quả.")
        return 1
    summary = summarize_runs(runs)
    made = [fig_extension_macro_f1(summary), fig_extension_delta(summary), fig_lambda_sweep()]
    for p in made:
        print("  ->", p, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
