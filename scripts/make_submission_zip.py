"""Đóng gói gói nộp bài: `<HọVàTên>_final.zip` gồm PDF, notebook, mã nguồn LaTeX và hình.

Cấu trúc gói::

    DinhXuanHuy_final.zip
    |- DinhXuanHuy_final.pdf              báo cáo (bản dựng mới nhất)
    |- DinhXuanHuy_final.ipynb            notebook tổng hợp toàn bộ đề tài
    |- DinhXuanHuy_final_latex/           mã nguồn LaTeX đầy đủ, biên dịch được
    |  |- main.tex                        bản Windows (Times New Roman)
    |  |- main-linux.tex                  bản Linux (font đi kèm)
    |  |- sections/ generated/ assets/
    |- DinhXuanHuy_final_figures/         toàn bộ hình PNG 300 DPI
    |- DOC_GOI_NOP.txt                    mô tả gói + địa chỉ kho mã nguồn

Cách dùng::

    python scripts/make_submission_zip.py
    python scripts/make_submission_zip.py --name "NguyenVanA"     # đổi tên khác
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "report-build"
REPORT = ROOT / "report"
FIGDIR = ROOT / "results" / "figures"
NOTEBOOK = ROOT / "notebooks" / "M8_Final_All.ipynb"
PDF = BUILD / "main-linux.pdf"

REPO_URL = "https://github.com/dinhxuanhuy/iot-m8-multitask"

# Không đưa vào gói nộp: tệp tạm, cache, và mọi thứ chỉ để dựng báo cáo tại chỗ
LATEX_EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints"}
LATEX_EXCLUDE_EXT = {".aux", ".log", ".out", ".toc", ".lof", ".lot", ".fls", ".fdb_latexmk"}


def collect_latex_files() -> list[tuple[Path, Path]]:
    """(đường dẫn nguồn, đường dẫn trong zip) cho cây nguồn LaTeX.

    Nguồn LaTeX nằm trong thư mục con ``report/`` chứ không phải gốc gói, vì `main*.tex` tham
    chiếu hình bằng đường dẫn TƯƠNG ĐỐI ``../results/figures/<ten>.png`` (tính từ ``report/``).
    Bản sao hình vì vậy phải nằm ở ``results/figures/`` NGANG CẤP với ``report/``; nếu để hình
    ở một thư mục rời thì khi giải nén ra, XeTeX báo
    ``Unable to load picture or PDF file '../results/figures/...'`` và dừng.
    """
    out: list[tuple[Path, Path]] = []
    for p in sorted(REPORT.rglob("*")):
        if not p.is_file():
            continue
        if any(part in LATEX_EXCLUDE_DIRS for part in p.parts):
            continue
        if p.suffix in LATEX_EXCLUDE_EXT:
            continue
        if p.name.startswith("main.") and p.suffix == ".pdf":
            continue                      # PDF đã có ở gốc gói, không nhân bản
        out.append((p, Path("report") / p.relative_to(REPORT)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Đóng gói gói nộp bài")
    ap.add_argument("--name", default="DinhXuanHuy",
                    help="Họ và tên không dấu, dùng làm tên tệp (mặc định DinhXuanHuy)")
    args = ap.parse_args()
    stem = f"{args.name}_final"

    for need, label in ((PDF, "PDF báo cáo"), (NOTEBOOK, "notebook"),
                        (FIGDIR, "thư mục hình")):
        if not need.exists():
            print(f"  !! thiếu {label}: {need}")
            return 1

    figs = sorted(FIGDIR.glob("*.png"))
    latex = collect_latex_files()
    print(f"  PDF        : {PDF.name} ({PDF.stat().st_size/1048576:.1f} MB)")
    print(f"  notebook   : {NOTEBOOK.name} ({NOTEBOOK.stat().st_size/1048576:.1f} MB)")
    print(f"  hình       : {len(figs)} tệp")
    print(f"  nguồn LaTeX: {len(latex)} tệp")

    info = f"""GÓI NỘP BÀI — ĐỀ TÀI M8
============================================================

Sinh viên : Đinh Xuân Huy
MSSV      : 23110102
Học phần  : Trí tuệ nhân tạo cho IoT
GVHD      : Hồ Nhựt Minh
Đề tài    : Học đa nhiệm trên CIFAR-10 — phân loại siêu lớp kết hợp
            dự đoán góc xoay ảnh tự giám sát

MÃ NGUỒN CÔNG KHAI
------------------------------------------------------------
{REPO_URL}

Kho chứa toàn bộ mã nguồn, kết quả thô của 39 lượt huấn luyện,
chỉ mục split đã đóng băng, dự đoán test theo chỉ mục và hai
checkpoint đã huấn luyện.

NỘI DUNG GÓI
------------------------------------------------------------
{stem}.pdf                 Báo cáo kỹ thuật, 47 trang
{stem}.ipynb               Notebook tổng hợp toàn bộ đề tài
{stem}_latex/              Mã nguồn LaTeX đầy đủ, biên dịch được
    main.tex               Bản Windows (Times New Roman)
    main-linux.tex         Bản Linux (font đóng gói kèm)
    sections/              Nội dung từng chương
    generated/             Bảng và macro sinh tự động từ kết quả
    assets/                Logo, huy hiệu và font
{stem}_figures/            Toàn bộ hình trong báo cáo, PNG 300 DPI

CÁCH BIÊN DỊCH LẠI BÁO CÁO
------------------------------------------------------------
Windows:
    cd {stem}_latex
    tectonic -X compile main.tex

Linux / macOS:
    cd {stem}_latex
    tectonic -X compile main-linux.tex

LƯU Ý VỀ SỐ LIỆU
------------------------------------------------------------
Mọi chỉ số trong báo cáo đều sinh tự động từ kết quả chạy thật
(27 lượt khối chính + 12 lượt khối mở rộng), không có giá trị
nào được nhập tay. Bảng và macro LaTeX được tạo bởi
scripts/make_report_data.py.
"""
    (BUILD / "DOC_GOI_NOP.txt").write_text(info, encoding="utf-8")

    zip_path = BUILD / f"{stem}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(PDF, f"{stem}.pdf")
        z.write(NOTEBOOK, f"{stem}.ipynb")
        z.write(BUILD / "DOC_GOI_NOP.txt", "DOC_GOI_NOP.txt")
        for p in figs:
            z.write(p, f"{stem}_figures/{p.name}")
            # bản sao thứ hai, đặt đúng chỗ để `../results/figures/` trong main*.tex phân giải
            z.write(p, f"{stem}_latex/results/figures/{p.name}")
        for src, rel in latex:
            z.write(src, f"{stem}_latex/{rel}")
        n = len(z.namelist())

    print(f"\n  -> {zip_path}")
    print(f"     {n} tệp, {zip_path.stat().st_size/1048576:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
