"""Sinh dữ liệu LaTeX cho báo cáo từ kết quả thực nghiệm thật.

Sinh ra:
  * ``report/generated/results_macros.tex`` — các macro số liệu;
  * ``report/generated/tab_*.tex``          — các bảng kết quả.

Mọi con số đều đọc từ ``results/checkpoints/*.json`` và ``results/*.json``.
Không có giá trị minh họa nào được đặt vào vị trí kết quả thực nghiệm.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m8.config import (  # noqa: E402
    CHECKPOINT_DIR, CHECKPOINT_EXT_DIR, LABEL_BUDGET, MAIN_CONFIGS, RESULTS_DIR,
)
from m8.experiments import load_all_runs, load_all_runs_multi, summarize_runs  # noqa: E402

GEN = ROOT / "report" / "generated"
GEN.mkdir(parents=True, exist_ok=True)

NUMBER = "0,000"


def vn(value: float, decimals: int = 4) -> str:
    """Định dạng số theo quy ước Việt Nam: dấu phẩy thập phân, dấu chấm phân cách nghìn."""
    if value is None:
        return "---"
    s = f"{value:,.{decimals}f}"
    return s.replace(",", "\u00a0").replace(".", ",").replace("\u00a0", ".")


def vn_pct(value: float, decimals: int = 2) -> str:
    return vn(value, decimals)


def ms(value: float, decimals: int = 3) -> str:
    return vn(value, decimals)


def tex_escape(text: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$")):
        text = text.replace(a, b)
    return text


def agg_text(a: dict, decimals: int = 4, unit: str = "") -> str:
    """mean ± std (mẫu) theo định dạng Việt Nam."""
    return f"${vn(a['mean'], decimals)} \\pm {vn(a['std'], decimals)}${unit}"


def mean_only(a: dict, decimals: int = 4) -> str:
    return vn(a["mean"], decimals)


# =========================================================================
#  Bảng 1 — số ảnh huấn luyện theo tỷ lệ nhãn
# =========================================================================

def tab_label_budget() -> str:
    lines = [
        r"% Sinh tự động — scripts/make_report_data.py",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{@{}l r r r r r@{}}",
        r"    \toprule",
        r"    Tỷ lệ nhãn & $\DL$ & Animal & Vehicle & $\DU$ & Tổng $\Dtrain$ \\",
        r"    \midrule",
    ]
    total_train = LABEL_BUDGET[0.10]["total"] + LABEL_BUDGET[0.10]["unlabeled"]
    for frac in sorted(LABEL_BUDGET):
        b = LABEL_BUDGET[frac]
        lines.append(
            f"    {int(round(frac * 100))}\\% & {vn(b['total'], 0)} & {vn(b['animal'], 0)} & "
            f"{vn(b['vehicle'], 0)} & {vn(b['unlabeled'], 0)} & {vn(total_train, 0)} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Số ảnh huấn luyện theo từng mức tỷ lệ nhãn. "
        r"$\DL$ là tập có nhãn, $\DU = \Dtrain \setminus \DL$ là phần ẩn nhãn; "
        r"mọi tỷ lệ được tính trên 45.000 ảnh của $\Dtrain$, không phải trên toàn bộ 60.000 ảnh.}",
        r"  \label{tab:label-budget}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def tab_split() -> str:
    lines = [
        r"% Sinh tự động — scripts/make_report_data.py",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{@{}l r r r r@{}}",
        r"    \toprule",
        r"    Tập & Tổng số ảnh & Animal & Vehicle & Tỷ lệ Animal \\",
        r"    \midrule",
        r"    $\Dtrain$ & 45.000 & 27.000 & 18.000 & 60{,}0\,\% \\",
        r"    $\Dval$   & 5.000  & 3.000  & 2.000  & 60{,}0\,\% \\",
        r"    $\Dtest$  & 10.000 & 6.000  & 4.000  & 60{,}0\,\% \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Quy mô và thành phần của ba tập dữ liệu sau phép gộp siêu lớp. "
        r"Tập test được giữ riêng hoàn toàn và chỉ mở ra ở bước đánh giá cuối cùng.}",
        r"  \label{tab:split}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def tab_arch() -> str:
    rows = [
        ("Ảnh đầu vào", r"$3 \times 32 \times 32$"),
        (r"Conv $3\times3$ + BatchNorm + ReLU", r"$64 \times 32 \times 32$"),
        ("Nhóm residual 1: 2 BasicBlock", r"$64 \times 32 \times 32$"),
        ("Nhóm residual 2: 2 BasicBlock", r"$128 \times 16 \times 16$"),
        ("Nhóm residual 3: 2 BasicBlock", r"$256 \times 8 \times 8$"),
        ("Nhóm residual 4: 2 BasicBlock", r"$512 \times 4 \times 4$"),
        ("Global Average Pooling + Flatten", r"vector 512 chiều"),
        (r"Superclass Head (tuyến tính, có bias)", r"$512 \to 2$"),
        (r"Rotation Head (tuyến tính, có bias)", r"$512 \to 4$"),
    ]
    lines = [
        r"% Sinh tự động — scripts/make_report_data.py",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{@{}l l@{}}",
        r"    \toprule",
        r"    Thành phần & Kích thước đầu ra $C \times H \times W$ \\",
        r"    \midrule",
    ]
    for a, b in rows:
        lines.append(f"    {a} & {b} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Chuỗi kích thước tensor của backbone ResNet-18 điều chỉnh cho ảnh "
        r"$32 \times 32$. Các giá trị này được kiểm tra tự động bằng "
        r"\texttt{scripts/self\_test.py} trên môi trường thực nghiệm.}",
        r"  \label{tab:arch}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def tab_configs() -> str:
    lines = [
        r"% Sinh tự động — scripts/make_report_data.py",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{@{}l l l l@{}}",
        r"    \toprule",
        r"    ID & Cấu hình & Ảnh cho phân loại & Ảnh cho nhiệm vụ xoay \\",
        r"    \midrule",
        r"    \cfg{E1}   & Baseline chỉ học siêu lớp & $\DL$ & Không có \\",
        r"    \cfg{E2-L} & Thêm nhánh xoay, cùng tập nguồn & $\DL$ & Chỉ $\DL$ \\",
        r"    \cfg{E2}   & Thêm nhánh xoay, mở rộng nguồn ảnh & $\DL$ & $\Dtrain = \DL \cup \DU$ \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Ba cấu hình chính. \cfg{E1} và \cfg{E2} là cặp đối chứng chính; "
        r"\cfg{E2-L} tách riêng tác dụng của việc mở rộng nguồn ảnh cho nhánh phụ.}",
        r"  \label{tab:configs}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def tab_hyperparams() -> str:
    rows = [
        ("Optimizer", r"AdamW"),
        ("Learning rate", r"$3 \times 10^{-4}$"),
        (r"Weight decay", r"$10^{-4}$"),
        ("Betas", r"$(0{,}9;\ 0{,}999)$"),
        ("Eps", r"$10^{-8}$"),
        (r"Kích thước minibatch $B_L$", r"tối đa 128 ảnh"),
        (r"Kích thước minibatch $B_R$", r"bằng $B_L$ tại mỗi bước"),
        ("Ngân sách học", r"100 epoch ($1$ epoch $=$ một lượt duyệt $\DL$)"),
        ("Scheduler", r"CosineAnnealingLR, $T_{\max}=100$, $\eta_{\min}=0$"),
        (r"Hệ số $\lamrot$", r"$0{,}5$ cho thí nghiệm chính"),
        ("Chọn checkpoint", r"Macro-F1 validation cao nhất; hòa thì loss thấp hơn, rồi epoch sớm hơn"),
        ("Early stopping", r"Không dùng trong bảng kết quả chính"),
        ("Seed", r"chia split 2026; lần lặp 42, 43, 44"),
        ("Chuẩn hóa", r"mean $=$ std $= (0{,}5;\ 0{,}5;\ 0{,}5)$"),
    ]
    lines = [
        r"% Sinh tự động — scripts/make_report_data.py",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{@{}l l@{}}",
        r"    \toprule",
        r"    Thành phần & Thiết lập cho thí nghiệm chính \\",
        r"    \midrule",
    ]
    for a, b in rows:
        lines.append(f"    {a} & {b} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Cấu hình huấn luyện dùng chung cho cả ba cấu hình. Ba cấu hình ghép cặp "
        r"dùng cùng trọng số khởi tạo của encoder và Superclass Head.}",
        r"  \label{tab:hyperparams}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def tab_experiment_matrix() -> str:
    """Bảng 5 của đề cương: bốn cấu hình mở rộng contrastive E3--E6."""
    rows = [
        (r"\cfg{E3}", r"$\Lsup + \lamc \Lntxent$",
         r"Không có nhánh xoay",
         r"$\Dtrain$; không dùng nhãn siêu lớp"),
        (r"\cfg{E4}", r"$\Lsup + \lamc \Lsupcon$",
         r"Không có nhánh xoay",
         r"Chỉ $\DL$ có nhãn hợp lệ"),
        (r"\cfg{E5}", r"$\Lsup + \lamrot \Lrot + \lamc \Lntxent$",
         r"$\Dtrain$",
         r"NT-Xent: $\Dtrain$; rotation: $\Dtrain$"),
        (r"\cfg{E6}", r"$\Lsup + \lamrot \Lrot + \lamc \Lsupcon$",
         r"$\Dtrain$",
         r"SupCon: $\DL$; rotation: $\Dtrain$"),
    ]
    lines = [
        r"% Sinh tự động — scripts/make_report_data.py",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{@{}l l l l@{}}",
        r"    \toprule",
        r"    ID & Hàm mục tiêu & Nguồn ảnh cho xoay & Nguồn ảnh cho loss bổ sung \\",
        r"    \midrule",
    ]
    for a, b, c, d in rows:
        lines.append(f"    {a} & {b} & {c} & {d} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Bốn cấu hình mở rộng contrastive. $\lamc$ dùng chung cho mọi cấu hình và "
        r"được khóa bằng validation trước khi đánh giá test (\secref{sec:results-ext-lambda}). "
        r"SupCon chỉ dùng $\DL$; NT-Xent dùng $\Dtrain$ vì không cần nhãn siêu lớp.}",
        r"  \label{tab:extensions}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def tab_main_results(summary: dict, pct: Optional[int] = None) -> str:
    """Bảng kết quả chính: Macro-F1 / Accuracy trên D_test, mean ± std qua 3 lần lặp."""
    agg = summary["aggregate"]
    pcts = sorted({int(k.split("_")[1]) for k in agg})
    if pct is not None:
        pcts = [pct]
    cfg_order = ["E1", "E2-L", "E2"]
    present = [c for c in cfg_order if any(k.startswith(c + "_") for k in agg)]

    ncol = 2 + 3 * len(pcts)
    colspec = "@{}l" + " " + " ".join([r"r r r"] * len(pcts)) + "@{}"
    header1 = r"    \toprule" + "\n"
    header1 += r"    \multirow{2}{*}{\makecell[l]{Cấu hình}}"
    for p in pcts:
        header1 += f" & \\multicolumn{{3}}{{c}}{{{p}\\,\\% nhãn}}"
    header1 += r" \\" + "\n"
    cmid = r"    "
    for i in range(len(pcts)):
        lo = 2 + 3 * i
        cmid += f"\\cmidrule(lr){{{lo}-{lo + 2}}}"
    cmid += "\n"
    header2 = (r"     & " + " & ".join([r"Macro-F1 & Acc. & $\Delta$"] * len(pcts)) + r" \\" + "\n")

    lines = [
        r"% Sinh tự động — scripts/make_report_data.py",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4.5pt}",
        r"  \begin{tabular}{" + colspec + "}",
        header1 + cmid + header2,
        r"    \midrule",
    ]
    for cfg in present:
        row = f"    \\cfg{{{cfg}}}"
        for p in pcts:
            a = agg.get(f"{cfg}_{p}")
            delta = summary["paired_delta"].get(f"{cfg}_minus_E1_{p}")
            if cfg == "E1":
                d = r"---"
            elif delta:
                # Dấu âm phải là '-' trần rồi bọc $...$ MỘT lần. Bản cũ đặt sign = "$-$"
                # rồi bọc thêm $...$ nữa, sinh ra "$$-$0,06$" — tức một cặp $$ (display
                # math rỗng) chèn vào giữa bảng.
                sign = "+" if delta["mean"] >= 0 else "-"
                d = f"${sign}{vn(abs(delta['mean']), 2)}$"
            else:
                d = "---"
            if not a:
                row += r" & --- & --- & ---"
            else:
                row += (f" & {vn(a['test_macro_f1']['mean'], 4)} & "
                        f"{vn(a['test_accuracy']['mean'], 4)} & {d}")
        row += r" \\"
        lines.append(row)
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Kết quả trên $\Dtest$: Macro-F1 và Accuracy (\enquote{Acc.}) là trung bình "
        r"qua ba lần lặp. $\Delta$ là chênh lệch Macro-F1 trung bình so với \cfg{E1} ở cùng mức "
        r"nhãn, tính bằng điểm phần trăm; giá trị dương nghĩa là cấu hình đa nhiệm tốt hơn "
        r"baseline. Độ lệch chuẩn qua ba lần lặp được báo cáo ở \tabref{tab:main-std}.}",
        r"  \label{tab:main-results}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def tab_main_std(summary: dict) -> str:
    agg = summary["aggregate"]
    pcts = sorted({int(k.split("_")[1]) for k in agg})
    cfg_order = ["E1", "E2-L", "E2"]
    present = [c for c in cfg_order if any(k.startswith(c + "_") for k in agg)]
    lines = [
        r"% Sinh tự động — scripts/make_report_data.py",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \begin{tabular}{@{}l " + " ".join([r"c"] * len(pcts)) + r"@{}}",
        r"    \toprule",
        r"    Cấu hình & " + " & ".join([f"{p}\\,\\% nhãn" for p in pcts]) + r" \\",
        r"    \midrule",
    ]
    for cfg in present:
        row = f"    \\cfg{{{cfg}}}"
        for p in pcts:
            a = agg.get(f"{cfg}_{p}")
            row += " & " + (agg_text(a["test_macro_f1"]) if a else "---")
        row += r" \\"
        lines.append(row)
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Macro-F1 trên $\Dtest$ dạng trung bình $\pm$ độ lệch chuẩn mẫu qua ba lần "
        r"lặp. Ba lần lặp chỉ cung cấp bằng chứng ban đầu về độ ổn định, không đủ để kết luận "
        r"về ý nghĩa thống kê.}",
        r"  \label{tab:main-std}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def tab_per_class(summary: dict, pct: int = 10) -> str:
    agg = summary["aggregate"]
    cfg_order = ["E1", "E2-L", "E2"]
    present = [c for c in cfg_order if f"{c}_{pct}" in agg]
    lines = [
        r"% Sinh tự động — scripts/make_report_data.py",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{6pt}",
        r"  \begin{tabular}{@{}l r r r r r@{}}",
        r"    \toprule",
        r"    & \multicolumn{2}{c}{Animal} & \multicolumn{2}{c}{Vehicle} & \\",
        r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"    Cấu hình & Precision & Recall & Precision & Recall & F1 (Animal / Vehicle) \\",
        r"    \midrule",
    ]
    for cfg in present:
        sel = [r for r in summary["runs"] if r["config"] == cfg and r["label_pct"] == pct]
        if not sel:
            continue
        pa = statistics.fmean(r["test_precision_animal"] for r in sel)
        ra = statistics.fmean(r["test_recall_animal"] for r in sel)
        pv = statistics.fmean(r["test_precision_vehicle"] for r in sel)
        rv = statistics.fmean(r["test_recall_vehicle"] for r in sel)
        fa = statistics.fmean(r["test_f1_animal"] for r in sel)
        fv = statistics.fmean(r["test_f1_vehicle"] for r in sel)
        lines.append(
            f"    \\cfg{{{cfg}}} & {vn(pa)} & {vn(ra)} & {vn(pv)} & {vn(rv)} & "
            f"{vn(fa)} / {vn(fv)} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Precision, recall và F1 từng siêu lớp tại mức " + str(pct) + r"\,\% nhãn "
        r"(trung bình ba lần lặp). Vì tỷ lệ Animal/Vehicle là 60/40, Accuracy đơn lẻ không "
        r"phản ánh đầy đủ chất lượng của từng lớp.}",
        # Nhãn phải PHỤ THUỘC mức nhãn: bản cũ dùng chung "tab:per-class" cho cả bảng 10%
        # và 50%, gây "multiply defined labels" và làm \ref trỏ không xác định.
        f"  \\label{{tab:per-class-{pct}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def tab_per_run(summary: dict) -> str:
    runs = sorted(summary["runs"], key=lambda r: (r["label_pct"], r["config"], r["seed"]))
    lines = [
        r"% Sinh tự động — scripts/make_report_data.py",
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{@{}l l r r r r r r@{}}",
        r"  \caption{Chi tiết từng lượt chạy trong khối thực nghiệm chính. "
        r"\enquote{Epoch tốt} là epoch được chọn theo Macro-F1 validation.}",
        r"  \label{tab:per-run} \\",
        r"  \toprule",
        r"  Cấu hình & Nhãn & Lần lặp & Epoch tốt & Val Macro-F1 & Test Macro-F1 & Test Acc & Giây \\",
        r"  \midrule",
        r"  \endfirsthead",
        r"  \multicolumn{8}{l}{\small\itshape (tiếp theo \tabref{tab:per-run})} \\",
        r"  \toprule",
        r"  Cấu hình & Nhãn & Lần lặp & Epoch tốt & Val Macro-F1 & Test Macro-F1 & Test Acc & Giây \\",
        r"  \midrule",
        r"  \endhead",
        r"  \bottomrule",
        r"  \endfoot",
    ]
    for r in runs:
        lines.append(
            f"  \\cfg{{{r['config']}}} & {r['label_pct']}\\,\\% & {r['seed']} & "
            f"{r['best_epoch']} & {vn(r['val_macro_f1'])} & {vn(r['test_macro_f1'])} & "
            f"{vn(r['test_accuracy'])} & {vn(r['train_seconds'], 1)} \\\\"
        )
    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")
    return "\n".join(lines)


def tab_training_cost(summary: dict) -> str:
    agg = summary["aggregate"]
    pcts = sorted({int(k.split("_")[1]) for k in agg})
    cfg_order = ["E1", "E2-L", "E2"]
    present = [c for c in cfg_order if any(k.startswith(c + "_") for k in agg)]
    lines = [
        r"% Sinh tự động — scripts/make_report_data.py",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{@{}l " + " ".join([r"r"] * len(pcts)) + r"@{}}",
        r"    \toprule",
        r"    Cấu hình & " + " & ".join([f"{p}\\,\\% (giây/lượt)" for p in pcts]) + r" \\",
        r"    \midrule",
    ]
    for cfg in present:
        row = f"    \\cfg{{{cfg}}}"
        for p in pcts:
            a = agg.get(f"{cfg}_{p}")
            row += " & " + (f"{vn(a['train_seconds']['mean'], 1)}" if a else "---")
        row += r" \\"
        lines.append(row)
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Thời gian huấn luyện trung bình cho một lượt chạy 100 epoch, " \
        r"trung bình qua BA lần lặp (độ lệch chuẩn xem \tabref{tab:main-std}). "
        r"\cfg{E2-L} và \cfg{E2} xử lý thêm minibatch cho nhiệm vụ xoay ở mỗi bước cập nhật, "
        r"nên chi phí học cao hơn \cfg{E1} một cách có hệ thống.}",
        r"  \label{tab:training-cost}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def tab_inference_cost() -> str:
    rows = [
        (r"Tham số encoder $\enc$", r"11.168.832"),
        (r"Tham số Superclass Head", r"1.026"),
        (r"Tham số Rotation Head", r"2.052"),
        (r"Tổng tham số khi suy luận", r"\RParamInference"),
        (r"Tổng tham số khi huấn luyện", r"\RParamTraining"),
        (r"Dung lượng FP32 của mô hình suy luận", r"\RParamInferenceMiB{} MiB"),
        (r"Dung lượng tệp trọng số suy luận đo được", r"\RParamInferenceFileMiB{} MiB"),
    ]
    lines = [
        r"% Sinh tự động — scripts/make_report_data.py",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{@{}l r@{}}",
        r"    \toprule",
        r"    Thành phần & Giá trị \\",
        r"    \midrule",
    ]
    for a, b in rows:
        lines.append(f"    {a} & {b} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Chi phí tham số của mô hình tại cấu hình \secref{sec:arch}. Đây là đại lượng " 
        r"\emph{xác định} (đếm trên kiến trúc), \textbf{không} phải trung bình qua nhiều lần "
        r"chạy. Rotation Head "
        r"chỉ tồn tại trong giai đoạn huấn luyện; khi suy luận chỉ cần encoder và Superclass Head. "
        r"Dung lượng tệp được đo bằng cách lưu riêng trọng số suy luận, không gộp optimizer state.}",
        r"  \label{tab:inference-cost}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def tab_latency() -> str:
    rows = [
        ("CPU (4 luồng), FP32", r"\RLatMeanCpu", r"\RLatMedianCpu", r"\RLatPNinetyFiveCpu"),
        ("GPU (RTX 5070 Laptop), FP32", r"\RLatMeanGpu", r"\RLatMedianGpu", r"\RLatPNinetyFiveGpu"),
    ]
    lines = [
        r"% Sinh tự động — scripts/make_report_data.py",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{@{}l r r r@{}}",
        r"    \toprule",
        r"    Điều kiện đo & Mean (ms/ảnh) & Median (ms/ảnh) & p95 (ms/ảnh) \\",
        r"    \midrule",
    ]
    for a, b, c, d in rows:
        lines.append(f"    {a} & {b} & {c} & {d} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Độ trễ suy luận với batch $=1$, tensor $1 \times 3 \times 32 \times 32$, "
        r"FP32, \code{model.eval()} và \code{torch.inference\_mode()}. Mỗi điều kiện khởi động "
        r"50 lượt rồi đo 1.000 lượt, lặp ba phiên. Cách tổng hợp ba phiên KHÔNG giống nhau: "
        r"\emph{mean} và \emph{p95} là trung bình cộng của ba phiên, còn \emph{median} là trung "
        r"vị của ba trung vị phiên (xem \code{src/m8/benchmark.py}). Phép đo chỉ tính thời gian "
        r"tính toán của mô hình, không gồm đọc ảnh và tiền xử lý.}",
        r"  \label{tab:latency}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def tab_convergence(summary: dict) -> str:
    """Bảng: thời điểm đạt checkpoint tốt nhất và mức quá khớp cuối quá trình."""
    raw = load_all_runs(CHECKPOINT_DIR)
    if not raw:
        return "% (chưa có dữ liệu hội tụ)"
    lines = [
        r"% Sinh tự động — scripts/make_report_data.py",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{@{}l c c c r@{}}",
        r"    \toprule",
        r"    \makecell[l]{Cấu hình} & \makecell{Epoch\\tốt nhất} "
        r"& \makecell{Macro-F1 val\\tại epoch tốt nhất} "
        r"& \makecell{Macro-F1 val\\tại epoch cuối} "
        r"& \makecell[r]{Khoảng\\cách} \\",
        r"    \midrule",
    ]
    for cfg in ("E1", "E2-L", "E2"):
        sel = [r for r in raw.values() if r["spec"]["config"] == cfg]
        if not sel:
            continue
        best_epochs = [r["best_epoch"] for r in sel]
        best_f1 = [r["best_val"]["macro_f1"] for r in sel]
        final_f1 = [r["history"][-1]["val_macro_f1"] for r in sel]
        gaps = [b - f for b, f in zip(best_f1, final_f1)]
        lines.append(
            f"    \\cfg{{{cfg}}} & {vn(statistics.fmean(best_epochs), 1)} & "
            f"{vn(statistics.fmean(best_f1))} & {vn(statistics.fmean(final_f1))} & "
            f"{_signed(statistics.fmean(gaps), 4)} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Hành vi hội tụ. \enquote{Khoảng cách} là Macro-F1 validation tại epoch tốt "
        r"nhất trừ giá trị tại epoch cuối; giá trị dương nghĩa là mô hình bắt đầu suy giảm trên "
        r"validation trước khi kết thúc 100 epoch. Vì không dùng Early Stopping, quá trình vẫn "
        r"chạy tới epoch 100 như thiết kế.}",
        r"  \label{tab:convergence}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def tab_extension_results(summary: dict) -> str:
    """Bảng kết quả khối mở rộng contrastive E3--E6 so với E1 (trên D_test)."""
    agg = summary["aggregate"]
    runs = summary["runs"]
    have = {r["config"] for r in runs}
    present = [c for c in ("E1", "E2-L", "E2", "E3", "E4", "E5", "E6") if c in have]
    ext = [c for c in ("E3", "E4", "E5", "E6") if c in have]
    if not ext:
        return "% (chưa chạy khối mở rộng contrastive)"
    pcts = sorted({int(k.split("_")[1]) for k in agg if k.split("_")[0] in present})
    lines = [
        r"% Sinh tự động — scripts/make_report_data.py",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4.5pt}",
        r"  \begin{tabular}{@{}l " + " ".join([r"r r"] * len(pcts)) + r"@{}}",
        r"    \toprule",
        r"    & " + " & ".join([rf"\multicolumn{{2}}{{c}}{{{p}\,\% nhãn}}" for p in pcts]) + r" \\",
    ]
    cmid = "    "
    for i in range(len(pcts)):
        lo = 2 + 2 * i
        cmid += f"\\cmidrule(lr){{{lo}-{lo + 1}}}"
    lines.append(cmid)
    lines.append(r"    Cấu hình & " + " & ".join([r"Macro-F1 & $\Delta$"] * len(pcts)) + r" \\")
    lines.append(r"    \midrule")
    for cfg in present:
        if cfg in ("E2-L", "E2"):
            lines.append(r"    \addlinespace[2pt]")
        row = f"    \\cfg{{{cfg}}}"
        for p in pcts:
            a = agg.get(f"{cfg}_{p}")
            d = summary["paired_delta"].get(f"{cfg}_minus_E1_{p}")
            if not a:
                row += r" & --- & ---"
                continue
            delta = "---" if cfg == "E1" or d is None else _signed(d["mean"])
            row += f" & {vn(a['test_macro_f1']['mean'], 4)} & {delta}"
        row += r" \\"
        lines.append(row)
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Kết quả trên $\Dtest$, trung bình qua ba lần lặp. $\Delta$ là chênh lệch "
        r"Macro-F1 ghép cặp so với \cfg{E1} ở cùng mức nhãn, đơn vị điểm phần trăm. Hai nhóm "
        r"được ngăn bằng khoảng trắng: \cfg{E1}/\cfg{E2-L}/\cfg{E2} là khối chính; "
        r"\cfg{E3}--\cfg{E6} là khối mở rộng contrastive.}",
        r"  \label{tab:extension-results}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def tab_extension_cost(summary: dict) -> str:
    """Chi phí của khối mở rộng: tham số head phụ, thời gian học, lượt ảnh."""
    runs = summary["runs"]
    have = {r["config"] for r in runs}
    present = [c for c in ("E1", "E2-L", "E2", "E3", "E4", "E5", "E6") if c in have]
    ext = [c for c in present if c in ("E3", "E4", "E5", "E6")]
    if not ext:
        return "% (chưa có dữ liệu chi phí mở rộng)"
    lines = [
        r"% Sinh tự động — scripts/make_report_data.py",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{@{}l r r r r r@{}}",
        r"    \toprule",
        r"    Cấu hình & \makecell[r]{Tham số\\suy luận} & \makecell[r]{Rotation\\Head} "
        r"& \makecell[r]{Projection\\Head} & \makecell[r]{Giây/epoch\\(mức 10\,\%)} "
        r"& \makecell[r]{Lượt ảnh phụ\\mỗi epoch\\(mức 10\,\%)} \\",
        r"    \midrule",
    ]
    for cfg in present:
        sel = [r for r in runs if r["config"] == cfg and r["label_pct"] == 10]
        if not sel:
            sel = [r for r in runs if r["config"] == cfg]
        if not sel:
            continue
        inf = sel[0]["inference_params"]
        proj = sel[0].get("projection_params", 0)
        rot_head = sel[0].get("training_params", 0) - inf
        sec = statistics.fmean(r["seconds_per_epoch"] for r in sel)
        # lượt ảnh phụ mỗi epoch = (xoay + contrastive) / số epoch
        epochs = sel[0]["epochs"]
        aux = (statistics.fmean(r.get("rot_images_seen", 0) for r in sel)
               + statistics.fmean(r.get("contrastive_images_seen", 0) for r in sel)) / epochs
        lines.append(
            f"    \\cfg{{{cfg}}} & {vn(inf, 0)} & {vn(max(rot_head, 0), 0)}"
            f" & {vn(proj, 0)} & {vn(sec, 1)} & {vn(aux, 0)} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Chi phí của khối mở rộng. Số tham số suy luận \textbf{không} thay đổi so với "
        r"khối chính vì cả Rotation Head lẫn Projection Head đều bị bỏ khi suy luận. "
        r"\enquote{Lượt ảnh phụ} là tổng số lượt ảnh mà các nhánh phụ xử lý trong một epoch; "
        r"đây là lý do định lượng khiến các cấu hình mở rộng chậm hơn \cfg{E1}.}",
        r"  \label{tab:extension-cost}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def tab_lambda_sweep() -> str:
    """Bảng khảo sát λ_c (chỉ dùng validation)."""
    path = RESULTS_DIR / "lambda_c_sweep.json"
    if not path.exists():
        return "% (chưa có dữ liệu khảo sát λ_c)"
    data = json.loads(path.read_text(encoding="utf-8"))
    recs = data.get("records", [])
    if not recs:
        return "% (chưa có dữ liệu khảo sát λ_c)"
    configs = sorted({r["config"] for r in recs})
    lambdas = sorted({r["lambda_c"] for r in recs})
    lines = [
        r"% Sinh tự động — scripts/make_report_data.py",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{@{}l " + " ".join([r"c"] * len(lambdas)) + r"@{}}",
        r"    \toprule",
        r"    Cấu hình & " + " & ".join([f"$\\lamc = {vn(l, 2)}$" for l in lambdas]) + r" \\",
        r"    \midrule",
    ]
    for cfg in configs:
        row = f"    \\cfg{{{cfg}}}"
        for lam in lambdas:
            sel = [r for r in recs if r["config"] == cfg and r["lambda_c"] == lam]
            row += " & " + (vn(sel[0]["best_val_macro_f1"], 4) if sel else "---")
        row += r" \\"
        lines.append(row)
    chosen = data.get("chosen", {})
    shared = data.get("shared_lambda_c")
    if shared is not None and shared in lambdas:
        row = r"    \midrule" + "\n" + r"    \textit{được khóa}"
        for lam in lambdas:
            row += " & " + (r"$\bullet$" if lam == shared else "")
        row += r" \\"
        lines.append(row)
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Khảo sát $\lamc$ bằng \textbf{Macro-F1 validation} với "
        + vn(data.get("epochs", 0), 0) + r" epoch, mức "
        + vn(data.get("fraction", 0) * 100, 0) + r"\,\% nhãn, seed "
        + str(data.get("seed", "")) + r". Tập test \textbf{không} được đánh giá trong bước này. "
        r"Quy tắc khóa được khai báo trước: chọn \emph{một} $\lamc$ dùng chung cho mọi cấu "
        r"hình mở rộng sao cho trung bình Macro-F1 validation qua hai cấu hình khảo sát là lớn "
        r"nhất; giá trị được chọn là "
        + vn(data.get("shared_lambda_c", 0), 2) + r" (trung bình "
        + vn(data.get("shared_mean_val", 0), 4) + r").}",
        r"  \label{tab:lambda-sweep}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def tab_rotation_data_cost() -> str:
    """Bảng: chi phí dựng view xoay khi nguồn ảnh là D_L so với D_train."""
    path = RESULTS_DIR / "rotation_data_cost.json"
    if not path.exists():
        return "% (chưa có dữ liệu chi phí dữ liệu nhánh xoay)"
    data = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        r"% Sinh tự động — scripts/make_report_data.py",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{6pt}",
        r"  \begin{tabular}{@{}r r r r r@{}}",
        r"    \toprule",
        r"    Mức nhãn & $|\DL|$ (view) & $|\Dtrain|$ (view) "
        r"& Giây/epoch khi nguồn $= \DL$ & Giây/epoch khi nguồn $= \Dtrain$ \\",
        r"    \midrule",
    ]
    for row in data["rows"]:
        lines.append(
            f"    {row['label_pct']}\\,\\% & {vn(row['dl_images'], 0)} & "
            f"{vn(row['train_images'], 0)} & "
            f"{vn(row['seconds_per_epoch_rotation_from_dl'], 1)} & "
            f"{vn(row['seconds_per_epoch_rotation_from_train'], 1)} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Chi phí \emph{dựng view} cho nhánh xoay, đo riêng trên CPU, không gồm "
        r"forward/backward. Bảng đo theo GIẢ ĐỊNH nhánh phụ duyệt hết tập nguồn mỗi epoch; "
        r"trong mã nguồn thì $B_R$ được cắt về đúng $b_L$, nên số view \emph{thực dùng} mỗi "
        r"epoch bằng $|\DL|$ ở cả \cfg{E2-L} và \cfg{E2}. Vì vậy chi phí dựng view thực tế của "
        r"hai cấu hình gần như bằng nhau, và tỷ lệ nêu trên là chi phí của kịch bản giả định.}",
        r"  \label{tab:rotation-data-cost}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def tab_rotation_diag(summary: dict) -> str:
    """Rotation Accuracy chẩn đoán (nếu đã đo được)."""
    rot_path = RESULTS_DIR / "rotation_diagnostic.json"
    if not rot_path.exists():
        return "% (chưa có dữ liệu Rotation Accuracy chẩn đoán)"
    data = json.loads(rot_path.read_text(encoding="utf-8"))
    lines = [
        r"% Sinh tự động — scripts/make_report_data.py",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{@{}l r r r@{}}",
        r"    \toprule",
        r"    Cấu hình & Mức nhãn & Lần lặp & Rotation Accuracy (validation) \\",
        r"    \midrule",
    ]
    for row in data.get("rows", []):
        lines.append(
            f"    \\cfg{{{row['config']}}} & {row['label_pct']}\\,\\% & {row['seed']} & "
            f"{vn(row['rotation_accuracy'])} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \caption{Rotation Accuracy chẩn đoán trên $\Dval$, đánh giá đủ bốn phép xoay cho "
        r"mỗi ảnh. Đây chỉ là chỉ số chẩn đoán nhánh phụ và không thay thế bằng chứng về "
        r"Macro-F1 của nhiệm vụ chính.}",
        r"  \label{tab:rotation-diag}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# =========================================================================
#  Macro số liệu
# =========================================================================

def _delta_stats(summary: dict, cfg: str, pct: int, base: str = "E1") -> Optional[dict]:
    """Thống kê chênh lệch ghép cặp của ``cfg`` so với ``base`` tại mức ``pct``."""
    runs = summary["runs"]
    deltas: List[float] = []
    for seed in sorted({r["seed"] for r in runs}):
        a = [r for r in runs if r["config"] == cfg and r["label_pct"] == pct and r["seed"] == seed]
        b = [r for r in runs if r["config"] == base and r["label_pct"] == pct and r["seed"] == seed]
        if a and b:
            deltas.append((a[0]["test_macro_f1"] - b[0]["test_macro_f1"]) * 100.0)
    if not deltas:
        return None
    arr = np.asarray(deltas, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "values": deltas,
        "n": int(arr.size),
        "n_positive": int((arr > 0).sum()),
    }


def _signed(value: float, decimals: int = 2) -> str:
    """Chuỗi số có dấu, định dạng Việt Nam, sẵn sàng chèn vào LaTeX."""
    sign = "+" if value >= 0 else "$-$"
    return f"{sign}{vn(abs(value), decimals)}"


def _clean(text: str) -> str:
    """Bỏ các lệnh LaTeX khỏi câu sinh tự động, để dùng được cả trong notebook."""
    text = re.sub(r"\\cfg\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:mathrm|text|mathbf|mathsf)\{([^}]*)\}", r"\1", text)
    text = text.replace(r"\,\%", " %").replace(r"\%", "%")
    text = text.replace(r"$\times$", "x").replace(r"\\", "")
    text = re.sub(r"\\[A-Za-z]+\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _finding_sentence(summary: dict, tex: bool = True) -> str:
    """Một câu kết luận được sinh từ chính kết quả, không viết cứng.

    ``tex=True`` trả về chuỗi sẵn sàng chèn vào LaTeX; ``tex=False`` trả về văn bản thuần.
    """
    def finish(s: str) -> str:
        return s if tex else _clean(s)

    best_cfg, best_pct, best = None, None, None
    for pct in sorted({r["label_pct"] for r in summary["runs"]}):
        for cfg in ("E2", "E2-L"):
            d = _delta_stats(summary, cfg, pct)
            if d is not None and (best is None or d["mean"] > best["mean"]):
                best_cfg, best_pct, best = cfg, pct, d
    if best is None:
        return "Không đủ dữ liệu để kết luận về hiệu ứng của nhiệm vụ phụ."
    if best["mean"] > 0 and best["n_positive"] == best["n"]:
        return finish(
            f"Ở cả ba lần lặp, cấu hình \\cfg{{{best_cfg}}} tại mức {best_pct}\\,\\% nhãn đều cho "
            f"Macro-F1 test cao hơn \\cfg{{E1}} (chênh lệch trung bình {_signed(best['mean'])} "
            f"điểm phần trăm, độ lệch chuẩn {vn(best['std'], 2)})."
        )
    if best["mean"] < 0:
        return finish(
            f"Kết quả không ủng hộ giả thuyết: hiệu ứng dương lớn nhất quan sát được chỉ là "
            f"{_signed(best['mean'])} điểm phần trăm (\\cfg{{{best_cfg}}}, {best_pct}\\,\\% nhãn) "
            f"và {best['n'] - best['n_positive']}/{best['n']} lần lặp cho kết quả thấp hơn \\cfg{{E1}}."
        )
    return finish(
        f"Hiệu ứng dương lớn nhất quan sát được là {_signed(best['mean'])} điểm phần trăm "
        f"(\\cfg{{{best_cfg}}}, {best_pct}\\,\\% nhãn) nhưng chỉ {best['n_positive']}/{best['n']} "
        f"lần lặp là dương, tức chênh lệch này nằm trong phạm vi biến động giữa các lần lặp."
    )


def _cost_sentence(summary: dict, tex: bool = True) -> str:
    runs = summary["runs"]
    pcts = sorted({r["label_pct"] for r in runs})
    parts = []
    for pct in pcts:
        times = {c: [r["train_seconds"] for r in runs if r["config"] == c and r["label_pct"] == pct]
                 for c in ("E1", "E2-L", "E2")}
        if all(times[c] for c in times):
            ratio = statistics.fmean(times["E2"]) / statistics.fmean(times["E1"])
            parts.append(f"{pct}\\,\\%: " + (f"$\\times${vn(ratio, 2)}" if tex else f"x{vn(ratio, 2)}"))
    if not parts:
        return ""
    s = ("Tỷ lệ thời gian huấn luyện \\cfg{E2}/\\cfg{E1} là " if tex
         else "Tỷ lệ thời gian huấn luyện E2/E1 là ")
    s += ", ".join(parts)
    s += " (cùng 100 epoch, khác số lượt ảnh được xử lý mỗi bước)."
    return s if tex else _clean(s)


def _data_source_sentence(summary: dict, tex: bool = True) -> str:
    runs = summary["runs"]
    parts = []
    for pct in sorted({r["label_pct"] for r in runs}):
        d = _delta_stats(summary, "E2", pct, base="E2-L")
        if d is None:
            continue
        verdict = "cao hơn" if d["mean"] > 0 else "thấp hơn"
        cfg_e2 = "\\cfg{E2}" if tex else "E2"
        cfg_e2l = "\\cfg{E2-L}" if tex else "E2-L"
        pct_tex = f"{pct}\\,\\%" if tex else f"{pct} %"
        parts.append(
            f"tại {pct_tex} nhãn, {cfg_e2} {verdict} {cfg_e2l} trung bình "
            f"{vn(abs(d['mean']), 2)} điểm phần trăm ({d['n_positive']}/{d['n']} lần lặp dương)"
        )
    if not parts:
        return "Chưa đủ dữ liệu để so sánh hai nguồn ảnh cho nhánh phụ."
    return "Mở rộng nguồn ảnh cho nhánh phụ: " + "; ".join(parts) + "."


def _extension_text(summary: dict) -> tuple[str, str]:
    """Sinh (phân tích, kết luận) cho khối mở rộng, dựa trên kết quả thật.

    Nguyên tắc: chỉ được nói điều mà số liệu chống đỡ; nêu rõ số lần lặp cùng dấu.
    """
    runs = summary["runs"]
    have = {r["config"] for r in runs}
    ext = [c for c in ("E3", "E4", "E5", "E6") if c in have]
    if not ext:
        msg = ("Khối mở rộng contrastive chưa được chạy tại thời điểm sinh báo cáo; "
               "mục này chỉ mô tả giao thức.")
        return msg, msg

    pcts = sorted({r["label_pct"] for r in runs})
    # 1) cấu hình mở rộng tốt nhất ở mỗi mức nhãn
    lines = []
    for pct in pcts:
        best_cfg, best_delta = None, None
        for cfg in ext:
            d = _delta_stats(summary, cfg, pct)
            if d is None:
                continue
            if best_delta is None or d["mean"] > best_delta["mean"]:
                best_cfg, best_delta = cfg, d
        if best_cfg is None:
            continue
        verdict = "cao hơn" if best_delta["mean"] > 0 else "thấp hơn"
        lines.append(
            f"tại {pct}\\,\\% nhãn, cấu hình mở rộng tốt nhất là \\cfg{{{best_cfg}}}, "
            f"{verdict} \\cfg{{E1}} trung bình {vn(abs(best_delta['mean']), 2)} điểm phần trăm "
            f"({best_delta['n_positive']}/{best_delta['n']} lần lặp dương)"
        )

    # 2) so sánh SupCon với NT-Xent cùng điều kiện có/không nhánh xoay (E6 vs E5, E4 vs E3)
    pairs = []
    for hi, lo in (("E6", "E5"), ("E4", "E3")):
        if hi in ext and lo in ext:
            for pct in pcts:
                d = _delta_stats(summary, hi, pct, base=lo)
                if d is None:
                    continue
                better = hi if d["mean"] > 0 else lo
                pairs.append(
                    f"{pct}\\,\\% nhãn: SupCon (\\cfg{{E{hi[-1]}}}) so với NT-Xent "
                    f"(\\cfg{{E{lo[-1]}}}) chênh {_signed(d['mean'])} điểm phần trăm, "
                    f"nghiêng về \\cfg{{{better}}}"
                )

    # 3) vai trò của nhánh xoay khi đã có contrastive (E6 vs E4, E5 vs E3)
    rot_notes = []
    for hi, lo in (("E6", "E4"), ("E5", "E3")):
        if hi in ext and lo in ext:
            for pct in pcts:
                d = _delta_stats(summary, hi, pct, base=lo)
                if d is None:
                    continue
                better = hi if d["mean"] > 0 else lo
                rot_notes.append(
                    f"{pct}\\,\\% nhãn: thêm nhánh xoay (\\cfg{{{hi}}} so với \\cfg{{{lo}}}) "
                    f"chênh {_signed(d['mean'])} điểm phần trăm về phía \\cfg{{{better}}}"
                )

    best_overall = None
    for cfg in ext:
        for pct in pcts:
            d = _delta_stats(summary, cfg, pct)
            if d is not None and (best_overall is None or d["mean"] > best_overall[2]["mean"]):
                best_overall = (cfg, pct, d)

    analysis = "Kết quả khối mở rộng: " + "; ".join(lines) + "."
    if pairs:
        analysis += " Đối chiếu hai loại loss contrastive — " + "; ".join(pairs) + "."
    if rot_notes:
        analysis += " Khi đã có nhánh contrastive — " + "; ".join(rot_notes) + "."

    if best_overall is not None:
        cfg, pct, d = best_overall
        if d["mean"] > 0 and d["n_positive"] == d["n"]:
            concl = (
                f"Trong điều kiện đã khảo sát, cấu hình mở rộng \\textbf{{có}} cải thiện so với "
                f"\\cfg{{E1}}: \\cfg{{{cfg}}} tại {pct}\\,\\% nhãn đạt chênh lệch trung bình "
                f"{_signed(d['mean'])} điểm phần trăm và dương ở cả {d['n']}/{d['n']} lần lặp. "
                f"Tuy nhiên mức cải thiện này vẫn nằm trong cùng bậc độ lớn với độ lệch chuẩn "
                f"giữa các lần lặp, nên chỉ được coi là bằng chứng thăm dò."
            )
        elif d["mean"] > 0:
            concl = (
                f"Không có cấu hình mở rộng nào cải thiện \\cfg{{E1}} một cách nhất quán: hiệu "
                f"ứng dương lớn nhất là {_signed(d['mean'])} điểm phần trăm (\\cfg{{{cfg}}}, "
                f"{pct}\\,\\% nhãn) nhưng chỉ {d['n_positive']}/{d['n']} lần lặp là dương."
            )
        else:
            concl = (
                f"Trong điều kiện đã khảo sát, contrastive learning \\textbf{{không}} cải thiện "
                f"phân loại siêu lớp: cấu hình mở rộng tốt nhất (\\cfg{{{cfg}}}, {pct}\\,\\% nhãn) "
                f"vẫn thấp hơn \\cfg{{E1}} trung bình {vn(abs(d['mean']), 2)} điểm phần trăm, và "
                f"{d['n'] - d['n_positive']}/{d['n']} lần lặp cho kết quả thấp hơn. Đây là một "
                f"kết quả âm và được báo cáo nguyên trạng."
            )
        concl += (
            " Cần nhấn mạnh ba lần lặp chỉ cho bằng chứng ban đầu: không có tuyên bố ý nghĩa "
            "thống kê nào được đưa ra, và kết luận chỉ áp dụng cho kiến trúc, hai siêu lớp, "
            "các mức nhãn và giao thức đã khảo sát."
        )
    else:
        concl = "Không đủ dữ liệu để kết luận cho khối mở rộng."
    return analysis, concl


def build_finding_macros(summary: dict) -> Dict[str, str]:
    """Các macro nội dung được sinh từ kết quả (câu kết luận, chênh lệch ghép cặp)."""
    out: Dict[str, str] = {}
    for pct in (10, 20, 50):
        tag = {10: "Ten", 20: "Twenty", 50: "Fifty"}[pct]
        for cfg, cfgname in (("E2", "Etwo"), ("E2-L", "EtwoL")):
            d = _delta_stats(summary, cfg, pct)
            out[f"{cfgname}{tag}Delta"] = _signed(d["mean"]) if d else "n/a"
            out[f"{cfgname}{tag}DeltaStd"] = vn(d["std"], 2) if d else "n/a"
            out[f"{cfgname}{tag}DeltaPos"] = f"{d['n_positive']}/{d['n']}" if d else "n/a"
        a = summary["aggregate"].get(f"E1_{pct}")
        out[f"Eone{tag}Val"] = vn(a["test_macro_f1"]["mean"], 4) if a else "n/a"
        out[f"Eone{tag}Acc"] = vn(a["test_accuracy"]["mean"], 4) if a else "n/a"
        b = summary["aggregate"].get(f"E2_{pct}")
        out[f"Etwo{tag}Val"] = vn(b["test_macro_f1"]["mean"], 4) if b else "n/a"
        c = summary["aggregate"].get(f"E2-L_{pct}")
        out[f"EtwoL{tag}Val"] = vn(c["test_macro_f1"]["mean"], 4) if c else "n/a"

    out["FindingSentence"] = _finding_sentence(summary)
    out["FindingCone"] = _finding_sentence(summary)
    out["FindingCtwo"] = _data_source_sentence(summary)
    out["FindingCthree"] = _cost_sentence(summary)
    analysis, conclusion = _extension_text(summary)
    out["ExtensionAnalysis"] = analysis
    out["ExtensionConclusion"] = conclusion
    return out


REQUIRED_MACROS = {
    "Runs", "ExtRuns", "TotalHours", "Incomplete",
    "EnvPython", "EnvTorch", "EnvTorchvision", "EnvCudaBuild", "EnvGpu", "EnvVram",
    "EnvCpu", "EnvCores", "EnvRam",
    "ParamInference", "ParamTraining", "ParamRotationHead", "ParamInferenceMiB",
    "ParamInferenceFileMiB",
    "LatThreads", "PreprocessMs",
    "LatMeanCpu", "LatMedianCpu", "LatPNinetyFiveCpu",
    "LatMeanGpu", "LatMedianGpu", "LatPNinetyFiveGpu",
    "ParamTraining",
    "EoneTenVal", "EoneTenAcc", "EoneTenConf",
    "EoneTenVal", "EoneTwentyVal", "EoneFiftyVal",
    "EtwoTenVal", "EtwoTwentyVal", "EtwoFiftyVal",
    "EtwoLTenVal", "EtwoLTwentyVal", "EtwoLFiftyVal",
    "EtwoTenDelta", "EtwoTwentyDelta", "EtwoFiftyDelta",
    "EtwoLTenDelta", "EtwoLTwentyDelta", "EtwoLFiftyDelta",
    "FindingSentence", "FindingCone", "FindingCtwo", "FindingCthree",
}


def _check_required(macros: Dict[str, str]) -> List[str]:
    return sorted(REQUIRED_MACROS - set(macros))


def _prepare_data_ok(summary: dict) -> bool:
    """Kiểm tra khối thực nghiệm chính đã đủ 27 lượt chưa."""
    need = {(c, p, s) for c in ("E1", "E2-L", "E2") for p in (10, 20, 50) for s in (42, 43, 44)}
    have = {(r["config"], r["label_pct"], r["seed"]) for r in summary["runs"]}
    return need.issubset(have)


def build_macros(summary: dict, extras: dict, stub: bool = False) -> str:
    env = {}
    env_path = RESULTS_DIR / "environment.json"
    if env_path.exists():
        env = json.loads(env_path.read_text(encoding="utf-8"))

    bench_path = RESULTS_DIR / "inference_benchmark.json"
    bench = json.loads(bench_path.read_text(encoding="utf-8")) if bench_path.exists() else {}

    param = bench.get("parameters", {})
    cpu = bench.get("latency_cpu", {})
    gpu = bench.get("latency_gpu", {})

    flat = summary["runs"]
    # ``\RRuns`` và ``\RTotalHours`` được dùng trong văn xuôi để nói về KHỐI CHÍNH (mục 5.1,
    # mục 9.1, phần tóm tắt). Khi hàm này nhận ``summary_all`` (đã gộp khối mở rộng E3--E6),
    # nếu không lọc lại thì hai macro sẽ cộng dồn cả hai khối và mâu thuẫn với chính câu
    # "3 cấu hình × 3 mức nhãn × 3 lần lặp" ngay cạnh đó.
    main_flat = [r for r in flat if r["config"] in MAIN_CONFIGS]
    total_seconds = sum(r["train_seconds"] for r in main_flat)

    def g(d, *keys, default=0.0):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    macros = {
        "Runs": str(len(main_flat)),
        "ExtRuns": str(len(flat) - len(main_flat)),
        "TotalHours": vn(total_seconds / 3600.0, 2),
        "EnvPython": tex_escape(str(env.get("python", "---"))),
        "EnvTorch": tex_escape(str(env.get("torch", "---"))),
        "EnvTorchvision": tex_escape(str(env.get("torchvision", "---"))),
        "EnvCudaBuild": tex_escape(str(env.get("torch_cuda_build", "---"))),
        "EnvGpu": tex_escape(str(env.get("gpu") or "không sử dụng GPU")),
        "EnvVram": vn(float(env.get("gpu_memory_mib") or 0) / 1024.0, 3),
        "EnvCpu": tex_escape(str(env.get("cpu", "---"))[:60]),
        "EnvCores": str(env.get("physical_cores") or env.get("cpu_count", "---")),
        "EnvRam": vn(float(env.get("ram_gib") or 0), 1),
        "ParamInference": vn(param.get("inference_total", 0), 0),
        "ParamTraining": vn(param.get("training_total", 0), 0),
        "ParamRotationHead": vn(param.get("rotation_head", 0), 0),
        "ParamInferenceMiB": vn(param.get("inference_fp32_mib", 0.0), 2),
        "ParamInferenceFileMiB": vn(
            param.get("inference_file_mib_measured", param.get("inference_file_mib", 0.0)), 2
        ),
        "LatThreads": str(cpu.get("cpu_threads", 4)),
        "PreprocessMs": vn(bench.get("preprocess_ms_per_image", 0.0), 3),
    }
    # Lưu ý: tên lệnh LaTeX chỉ được chứa chữ cái. Vì vậy "p95" phải viết thành "PNinetyFive"
    # và "E2-L" thành "EtwoL"; dùng chữ số trong tên \newcommand sẽ gây lỗi
    # "Missing \begin{document}" rất khó đoán nguyên nhân.
    stat_names = {"mean_ms": "Mean", "median_ms": "Median", "p95_ms": "PNinetyFive"}
    for name, d in (("Cpu", cpu), ("Gpu", gpu)):
        for stat, label in stat_names.items():
            macros["Lat" + label + name] = vn(float(d.get(stat, 0.0)), 3)

    # giá trị mốc E1 tại 10%
    e1 = summary["aggregate"].get("E1_10")
    macros["EoneTenVal"] = vn(e1["val_macro_f1"]["mean"], 4) if e1 else NUMBER
    macros["EoneTenAcc"] = vn(e1["test_accuracy"]["mean"], 4) if e1 else NUMBER
    macros["EoneTenConf"] = str(extras.get("e1_10_mean_of_diagonal", 0))

    # câu kết luận sinh từ kết quả
    macros.update(build_finding_macros(summary))

    missing = sorted(REQUIRED_MACROS - set(macros) - {"Incomplete"})
    if missing:
        print("CẢNH BÁO: thiếu macro bắt buộc:", ", ".join(missing))
    out = [
        "% =====================================================================",
        "%  SINH TỰ ĐỘNG bởi scripts/make_report_data.py — không sửa tay",
        "% =====================================================================",
        "",
    ]
    for k, v in macros.items():
        if v == "n/a":
            v = "0{,}000" if stub else "$\\cdot$"
        out.append(f"\\newcommand{{\\R{k}}}{{{v}}}")
    if not _prepare_data_ok(summary):
        out.append("")
        out.append("% CẢNH BÁO: khối thực nghiệm chính chưa hoàn thành đủ 27 lượt.")
        out.append("\\newcommand{\\RIncomplete}{\\textbf{CẢNH BÁO: khối thực nghiệm chính chưa "
                   "hoàn thành đủ 27 lượt chạy; các bảng kết quả dưới đây chỉ là một phần.}}")
    else:
        out.append("\\newcommand{\\RIncomplete}{}")
    out.append("")
    return "\n".join(out)


def main() -> int:
    stub = "--stub" in sys.argv
    runs = load_all_runs(CHECKPOINT_DIR)
    runs_ext = load_all_runs(CHECKPOINT_EXT_DIR)
    all_runs = load_all_runs_multi(CHECKPOINT_DIR, CHECKPOINT_EXT_DIR)
    print(f"khối chính: {len(runs)} lượt | khối mở rộng: {len(runs_ext)} lượt", flush=True)
    if not runs:
        print("Chưa có kết quả trong", CHECKPOINT_DIR)
        return 1
    summary = summarize_runs(runs)
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # tổng hợp gộp khối chính + khối mở rộng (dùng cho bảng và hình của cả hai khối)
    summary_all = summarize_runs(all_runs) if runs_ext else summary
    (RESULTS_DIR / "summary_all.json").write_text(
        json.dumps(summary_all, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    extras: Dict[str, object] = {}
    extras_path = RESULTS_DIR / "summary_extras.json"
    if extras_path.exists():
        try:
            extras = json.loads(extras_path.read_text(encoding="utf-8"))
        except Exception:
            extras = {}

    (GEN / "results_macros.tex").write_text(build_macros(summary_all, extras, stub=stub), encoding="utf-8")
    (GEN / "tab_label_budget.tex").write_text(tab_label_budget(), encoding="utf-8")
    (GEN / "tab_split.tex").write_text(tab_split(), encoding="utf-8")
    (GEN / "tab_arch.tex").write_text(tab_arch(), encoding="utf-8")
    (GEN / "tab_configs.tex").write_text(tab_configs(), encoding="utf-8")
    (GEN / "tab_experiment_matrix.tex").write_text(tab_experiment_matrix(), encoding="utf-8")
    (GEN / "tab_hyperparams.tex").write_text(tab_hyperparams(), encoding="utf-8")
    (GEN / "tab_main_results.tex").write_text(tab_main_results(summary), encoding="utf-8")
    (GEN / "tab_main_std.tex").write_text(tab_main_std(summary), encoding="utf-8")
    (GEN / "tab_per_class.tex").write_text(tab_per_class(summary, pct=10), encoding="utf-8")
    (GEN / "tab_per_class_50.tex").write_text(tab_per_class(summary, pct=50), encoding="utf-8")
    (GEN / "tab_per_run.tex").write_text(tab_per_run(summary), encoding="utf-8")
    (GEN / "tab_training_cost.tex").write_text(tab_training_cost(summary), encoding="utf-8")
    (GEN / "tab_inference_cost.tex").write_text(tab_inference_cost(), encoding="utf-8")
    (GEN / "tab_latency.tex").write_text(tab_latency(), encoding="utf-8")
    (GEN / "tab_rotation_diag.tex").write_text(tab_rotation_diag(summary), encoding="utf-8")
    (GEN / "tab_convergence.tex").write_text(tab_convergence(summary), encoding="utf-8")
    (GEN / "tab_rotation_data_cost.tex").write_text(tab_rotation_data_cost(), encoding="utf-8")
    # bảng của khối mở rộng
    (GEN / "tab_extension_results.tex").write_text(
        tab_extension_results(summary_all), encoding="utf-8")
    (GEN / "tab_extension_cost.tex").write_text(tab_extension_cost(summary_all), encoding="utf-8")
    (GEN / "tab_lambda_sweep.tex").write_text(tab_lambda_sweep(), encoding="utf-8")
    print(f"Đã sinh dữ liệu LaTeX cho {len(runs)} lượt khối chính + "
          f"{len(runs_ext)} lượt khối mở rộng vào {GEN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
